// falcon.rs — Falcon/RISC-V engine HAL + NVIDIA ucode descriptor parsing.
//
// Ports nova-core's falcon.rs + firmware.rs for the bare-metal UEFI GSP boot.
// Register offsets are relative to a falcon's PFalconBase / PFalcon2Base:
//   GSP:  PFalcon 0x110000  PFalcon2 0x111000
//   SEC2: PFalcon 0x840000  PFalcon2 0x841000
//
// This module owns: the FalconUCodeDescV3 descriptor (native NVIDIA format used
// by linux-firmware booter/bootloader blobs), and the falcon load/boot ops.

#![allow(dead_code)]

/// Falcon register offsets relative to PFalconBase.
pub mod fb_off {
    pub const MAILBOX0: u64 = 0x040;
    pub const MAILBOX1: u64 = 0x044;
    pub const HWCFG2: u64 = 0x0f4;   // b31 reset_ready, b12 mem_scrubbing(0=done), b10 riscv
    pub const CPUCTL: u64 = 0x100;
    pub const BOOTVEC: u64 = 0x104;
    pub const DMACTL: u64 = 0x10c;
    pub const DMATRFBASE: u64 = 0x110;
    pub const DMATRFMOFFS: u64 = 0x114;
    pub const DMATRFCMD: u64 = 0x118;
    pub const DMATRFFBOFFS: u64 = 0x11c;
    pub const DMATRFBASE1: u64 = 0x128;
    pub const ENGINE: u64 = 0x3c0;   // b0 reset
    pub const IMEMC: u64 = 0x180;    // [4] stride 16
    pub const IMEMD: u64 = 0x184;
    pub const IMEMT: u64 = 0x188;
    pub const DMEMC: u64 = 0x1c0;    // [8] stride 8
    pub const DMEMD: u64 = 0x1c4;
    pub const FBIF_CTL: u64 = 0x624;
    pub const FBIF_TRANSCFG: u64 = 0x600;
}

/// RISC-V (Peregrine) register offsets relative to PFalcon2Base.
pub mod f2_off {
    pub const BROM_PARAADDR: u64 = 0x210;
    pub const BROM_ENGIDMASK: u64 = 0x19c;
    pub const BROM_CURR_UCODE_ID: u64 = 0x198;
    pub const MOD_SEL: u64 = 0x180;
    pub const RISCV_CORE_STATUS: u64 = 0x240; // b0 active
    pub const RISCV_CPUCTL: u64 = 0x388;      // b7 active, b4 halted
    pub const RISCV_BCR_CTRL: u64 = 0x668;    // b4 core_select(0=falcon,1=riscv), b0 valid
}

/// Engine bases.
pub const GSP_PFALCON: u64 = 0x110000;
pub const GSP_PFALCON2: u64 = 0x111000;
pub const SEC2_PFALCON: u64 = 0x840000;
pub const SEC2_PFALCON2: u64 = 0x841000;

/// NVIDIA-native Falcon ucode descriptor V3 (linux-firmware booter/bootloader).
/// Verified from nova-core firmware.rs FalconUCodeDescV3 (#[repr(C)]).
/// `hdr` high 16 bits = descriptor size; the code/data segments follow.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FalconUCodeDescV3 {
    pub hdr: u32,
    pub stored_size: u32,
    pub pkc_data_offset: u32,   // signature offset in DMEM
    pub interface_offset: u32,
    pub imem_phys_base: u32,
    pub imem_load_size: u32,
    pub imem_virt_base: u32,
    pub dmem_phys_base: u32,
    pub dmem_load_size: u32,
    pub engine_id_mask: u16,
    pub ucode_id: u8,
    pub signature_count: u8,
    pub signature_versions: u16,
    pub _reserved: u16,
}

impl FalconUCodeDescV3 {
    /// Descriptor size, encoded in hdr[31:16].
    pub fn hdr_size(&self) -> u32 {
        (self.hdr & 0xffff_0000) >> 16
    }
    /// Parse from the head of a descriptor slice (little-endian, packed).
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < core::mem::size_of::<Self>() {
            return None;
        }
        let rd32 = |o: usize| u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]);
        let rd16 = |o: usize| u16::from_le_bytes([b[o], b[o+1]]);
        Some(Self {
            hdr: rd32(0),
            stored_size: rd32(4),
            pkc_data_offset: rd32(8),
            interface_offset: rd32(12),
            imem_phys_base: rd32(16),
            imem_load_size: rd32(20),
            imem_virt_base: rd32(24),
            dmem_phys_base: rd32(28),
            dmem_load_size: rd32(32),
            engine_id_mask: rd16(36),
            ucode_id: b[38],
            signature_count: b[39],
            signature_versions: rd16(40),
            _reserved: rd16(42),
        })
    }
}

// ---- nvfw container parser (nouveau raw-struct format, magic 0x10de) --------
// Verified against booter_load-570.144.bin. Layout:
//   nvfw_bin_hdr @0 -> nvfw_hs_header_v2 @header_offset -> nvfw_hs_load_header_v2
// Load recipe from nouveau ga102_gsp_booter_ctor.

fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Falcon HS firmware load parameters extracted from an nvfw booter/bootloader.
#[derive(Debug, Clone, Copy, Default)]
pub struct HsFwLoad {
    /// The ucode data blob within the file: [data_off .. data_off+data_size).
    pub data_off: usize,
    pub data_size: usize,
    // IMEM (code): copy data[imem_base_img .. +imem_size] to falcon IMEM.
    pub imem_base_img: u32,
    pub imem_size: u32,
    // DMEM (data): copy data[dmem_base_img .. +dmem_size] to falcon DMEM.
    pub dmem_base_img: u32,
    pub dmem_size: u32,
    /// Offset of the signature within DMEM (loc - os_data_offset).
    pub dmem_sign: u32,
    /// Falcon boot vector.
    pub boot_addr: u32,
    /// Signature patch location (file offset) + per-sig size + count.
    pub patch_loc: u32,
    pub sig_prod_offset: u32,
    pub sig_prod_size: u32,
    pub num_sig: u32,
    /// Signature source base in the FILE: sig_prod_offset + patch_sig-value.
    pub sig_base: u32,
    /// BROM anti-rollback selectors (meta_data u32[3]).
    pub fuse_ver: u32,
    pub engine_id: u32,
    pub ucode_id: u32,
}

/// PIO-write a byte slice into falcon IMEM (ports at 0x180/0x184/0x188).
/// Ports gm200_flcn_pio_imem_wr. `tag_base` = imem virtual base >> 8.
///
/// Nouveau advertises both `.min` and `.max` as 0x100 for IMEM PIO, so every
/// tag is accompanied by exactly one 256-byte / 64-word block.  Pad a partial
/// final block with zeroes rather than emitting a short tagged block.
pub fn pio_imem_wr<W>(pfalcon: u64, data: &[u8], imem_base: u32, sec: bool, tag_base: u16, w: &W)
where W: Fn(u64, u32) {
    // init: 0x180 = (sec?BIT28:0)|BIT24|imem_base
    let init = (if sec { 1 << 28 } else { 0 }) | (1 << 24) | imem_base;
    w(pfalcon + 0x180, init);
    let mut tag = tag_base;
    let mut block_base = 0usize;
    while block_base < data.len() {
        // Each tag gets exactly 64 data writes, matching min=max=0x100.
        w(pfalcon + 0x188, tag as u32);
        for word_idx in 0..64usize {
            let i = block_base + word_idx * 4;
            let mut word = [0u8; 4];
            if i < data.len() {
                let n = core::cmp::min(4, data.len() - i);
                word[..n].copy_from_slice(&data[i..i + n]);
            }
            w(pfalcon + 0x184, u32::from_le_bytes(word));
        }
        block_base += 256;
        tag = tag.wrapping_add(1);
    }
}

/// PIO-write a byte slice into falcon DMEM (ports at 0x1c0/0x1c4).
pub fn pio_dmem_wr<W>(pfalcon: u64, data: &[u8], dmem_base: u32, w: &W)
where W: Fn(u64, u32) {
    // init: 0x1c0 = BIT24|dmem_base
    w(pfalcon + 0x1c0, (1 << 24) | dmem_base);
    let mut i = 0usize;
    while i < data.len() {
        let mut word = [0u8; 4];
        let n = core::cmp::min(4, data.len() - i);
        word[..n].copy_from_slice(&data[i..i + n]);
        w(pfalcon + 0x1c4, u32::from_le_bytes(word));
        i += 4;
    }
}

/// Boot a falcon that has firmware loaded. Ports gm200_flcn_fw_boot.
/// Sets mailboxes, bootvec, starts the CPU, waits for halt, returns (mbox0, ok).
pub fn gm200_fw_boot<R, W>(
    pfalcon: u64, mbox0: u32, mbox1: u32, boot_addr: u32, mbox0_ok: u32, r: &R, w: &W,
) -> (u32, bool)
where R: Fn(u64) -> u32, W: Fn(u64, u32) {
    w(pfalcon + fb_off::MAILBOX0, mbox0);
    w(pfalcon + fb_off::MAILBOX1, mbox1);
    w(pfalcon + fb_off::BOOTVEC, boot_addr); // 0x104
    w(pfalcon + fb_off::CPUCTL, 0x2);        // 0x100 startcpu
    // wait for halt: rd32(0x100) & 0x10
    let mut halted = false;
    for _ in 0..2_000_000 {
        if r(pfalcon + fb_off::CPUCTL) & 0x10 != 0 { halted = true; break; }
    }
    let m0 = r(pfalcon + fb_off::MAILBOX0);
    (m0, halted && m0 == mbox0_ok)
}

/// Reset a falcon engine and select the Falcon core, generic over base pair.
/// `r`/`w` are MMIO read/write closures (absolute BAR0+offset addresses).
/// Returns (scrub_done, core_select_ok, riscv_cpuctl_value).
/// `d(us)` = a real microsecond delay closure (e.g. wrapping boot::stall) —
/// required: the ENGINE reset pulse and scrub wait need real time, not spin-reads.
pub fn falcon_reset_and_select<R, W, D>(pfalcon: u64, pfalcon2: u64, r: &R, w: &W, d: &D) -> (bool, bool, u32)
where
    R: Fn(u64) -> u32,
    W: Fn(u64, u32),
    D: Fn(u64),
{
    // ga102_flcn_reset_prep: sample HWCFG2 and allow reset_ready up to 150us.
    let _ = r(pfalcon + fb_off::HWCFG2);
    for _ in 0..15 {
        if r(pfalcon + fb_off::HWCFG2) & 0x8000_0000 != 0 { break; }
        d(10);
    }
    // ENGINE reset pulse (offset 0x3c0 bit0): set, wait ~10us, clear (matches M1.4b).
    let eng = r(pfalcon + fb_off::ENGINE);
    w(pfalcon + fb_off::ENGINE, eng | 1);
    d(20);
    w(pfalcon + fb_off::ENGINE, eng & !1);
    d(20);
    // Wait mem scrubbing done (HWCFG2 bit12 -> 0), up to ~20ms.
    let mut scrub = false;
    for _ in 0..200 {
        if (r(pfalcon + fb_off::HWCFG2) >> 12) & 1 == 0 { scrub = true; break; }
        d(100);
    }
    // ga102_flcn_select: only switch when RISC-V is selected.  Writing zero
    // requests Falcon; hardware asserts bit0 once the switch is valid.
    let mut sel = true;
    if r(pfalcon2 + f2_off::RISCV_BCR_CTRL) & 0x10 != 0 {
        w(pfalcon2 + f2_off::RISCV_BCR_CTRL, 0);
        sel = false;
        for _ in 0..200 {
            if r(pfalcon2 + f2_off::RISCV_BCR_CTRL) & 1 != 0 { sel = true; break; }
            d(50);
        }
    }
    let cpuctl = r(pfalcon2 + f2_off::RISCV_CPUCTL);
    (scrub, sel, cpuctl)
}

/// Full runtime reset used by the GSP CPU sequencer's CORE_RESET opcode.
/// This matches nvkm_falcon_reset (gm200 disable + enable) rather than the
/// lighter reset_eng operation used during initial preparation.
pub fn falcon_runtime_reset_to_falcon<R, W, D>(
    pfalcon: u64,
    pfalcon2: u64,
    boot0: u32,
    r: &R,
    w: &W,
    d: &D,
) -> (bool, bool)
where
    R: Fn(u64) -> u32,
    W: Fn(u64, u32),
    D: Fn(u64),
{
    let select_falcon = || -> bool {
        if r(pfalcon2 + f2_off::RISCV_BCR_CTRL) & 0x10 == 0 {
            return true;
        }
        w(pfalcon2 + f2_off::RISCV_BCR_CTRL, 0);
        for _ in 0..200 {
            if r(pfalcon2 + f2_off::RISCV_BCR_CTRL) & 1 != 0 {
                return true;
            }
            d(50);
        }
        false
    };
    let reset_eng = || -> bool {
        let _ = r(pfalcon + fb_off::HWCFG2);
        for _ in 0..15 {
            if r(pfalcon + fb_off::HWCFG2) & 0x8000_0000 != 0 { break; }
            d(10);
        }
        let eng = r(pfalcon + fb_off::ENGINE);
        w(pfalcon + fb_off::ENGINE, eng | 1);
        d(10);
        w(pfalcon + fb_off::ENGINE, eng & !1);
        for _ in 0..200 {
            if r(pfalcon + fb_off::HWCFG2) & 0x1000 == 0 {
                return true;
            }
            d(100);
        }
        false
    };

    // gm200_flcn_disable: select Falcon, mask/clear IRQs, reset engine.
    let mut selected = select_falcon();
    let irqen = r(pfalcon + 0x048);
    w(pfalcon + 0x048, irqen & !0x3);
    w(pfalcon + 0x014, 0xffff_ffff);
    let scrub_disable = reset_eng();

    // gm200_flcn_enable: reset again, keep Falcon selected, wait for scrub,
    // and restore FALCON_SCTL from PMC_BOOT_0.
    let scrub_enable = reset_eng();
    selected &= select_falcon();
    w(pfalcon + 0x084, boot0);
    (scrub_disable && scrub_enable, selected)
}

/// Reset a GA102-class GSP falcon and select its RISC-V core, matching
/// `ga102_gsp_reset`: engine reset/scrub followed by
/// `mask(RISCV_BCR_CTRL, 0x111, 0x111)`.
pub fn falcon_reset_into_riscv<R, W, D>(
    pfalcon: u64,
    pfalcon2: u64,
    r: &R,
    w: &W,
    d: &D,
) -> (bool, u32)
where
    R: Fn(u64) -> u32,
    W: Fn(u64, u32),
    D: Fn(u64),
{
    let _ = r(pfalcon + fb_off::HWCFG2);
    for _ in 0..15 {
        if r(pfalcon + fb_off::HWCFG2) & 0x8000_0000 != 0 { break; }
        d(10);
    }
    let eng = r(pfalcon + fb_off::ENGINE);
    w(pfalcon + fb_off::ENGINE, eng | 1);
    d(20);
    w(pfalcon + fb_off::ENGINE, eng & !1);
    d(20);
    let mut scrub = false;
    for _ in 0..200 {
        if r(pfalcon + fb_off::HWCFG2) & 0x1000 == 0 { scrub = true; break; }
        d(100);
    }
    let bcr = r(pfalcon2 + f2_off::RISCV_BCR_CTRL);
    w(pfalcon2 + f2_off::RISCV_BCR_CTRL, (bcr & !0x111) | 0x111);
    d(20);
    (scrub, r(pfalcon2 + f2_off::RISCV_BCR_CTRL))
}

/// DMA-load a firmware segment into IMEM or DMEM via the ga102 DMA engine.
/// Ports nouveau ga102_flcn_dma_init/xfer/done + nvkm_falcon_dma_wr.
///  - `pfalcon`: falcon PFalconBase (abs BAR0+base)
///  - `fw_dma`: bus address of the ucode data blob in sysmem
///  - `img_off`: offset within the blob to start copying (imem_base_img/dmem_base_img)
///  - `mem_base`: destination base in IMEM/DMEM (usually 0)
///  - `size`: bytes to copy
///  - `to_imem`: true=IMEM (secure), false=DMEM
///  - `sec`: secure flag (true for IMEM code)
/// DMATRFCMD (0x118) bits: b0 write(0)/read, imem(b4), sec(b16..), size in b8..10 (log2-2).
pub fn dma_wr<R, W>(
    pfalcon: u64, fw_dma: u64, img_off: u32, mem_base: u32, size: u32,
    to_imem: bool, sec: bool, r: &R, w: &W,
) -> bool where R: Fn(u64) -> u32, W: Fn(u64, u32) {
    const XFER: u32 = 256; // 256-byte DMA blocks (nouveau uses this granularity)
    if size == 0 || size & (XFER - 1) != 0 {
        return false;
    }

    // Match nvkm_falcon_dma_wr exactly:
    //  * IMEM keeps fw_dma as the base and starts FBOFFS at img_off.  The
    //    source offset is also the IMEM virtual/tag address.
    //  * DMEM folds img_off into the DMA base and starts FBOFFS at zero.
    let (dma_addr, src_start) = if to_imem {
        (fw_dma, img_off)
    } else {
        (fw_dma + img_off as u64, 0)
    };
    if dma_addr & (XFER as u64 - 1) != 0 {
        return false;
    }
    w(pfalcon + fb_off::DMATRFBASE, ((dma_addr >> 8) & 0xffff_ffff) as u32);
    w(pfalcon + fb_off::DMATRFBASE1, (dma_addr >> 40) as u32);
    // cmd template: (log2(XFER)-2)<<8, plus imem/sec/write flags.
    let size_bits = (ilog2(XFER) - 2) << 8;
    let mut cmd = size_bits;
    if to_imem { cmd |= 1 << 4; }      // IMEM select
    if sec { cmd |= 0x0000_0004; }     // sec=1 in DMATRFCMD bits 3:2
    // transfer loop, XFER bytes at a time.
    let mut off = 0u32;
    while off < size {
        // xfer: 0x114 = mem_base+off (dest), 0x11c = src offset within base window, 0x118 = cmd
        w(pfalcon + fb_off::DMATRFMOFFS, mem_base + off);
        w(pfalcon + fb_off::DMATRFFBOFFS, src_start + off);
        w(pfalcon + fb_off::DMATRFCMD, cmd);
        // wait dma_done: rd32(0x118) & 0x2
        let mut done = false;
        for _ in 0..100000 {
            if r(pfalcon + fb_off::DMATRFCMD) & 0x2 != 0 { done = true; break; }
        }
        if !done { return false; }
        off += XFER;
    }
    true
}

fn ilog2(mut v: u32) -> u32 { let mut n = 0; while v > 1 { v >>= 1; n += 1; } n }

/// Program the ga102 BROM (Boot ROM signature verify) registers, then leave the
/// falcon ready to start. Ports ga102_flcn_fw_boot's BROM writes.
pub fn program_brom<W>(pfalcon2: u64, dmem_sign: u32, engine_id: u32, ucode_id: u32, w: &W)
where W: Fn(u64, u32) {
    w(pfalcon2 + f2_off::BROM_PARAADDR, dmem_sign);
    w(pfalcon2 + f2_off::BROM_ENGIDMASK, engine_id);
    w(pfalcon2 + f2_off::BROM_CURR_UCODE_ID, ucode_id);
    w(pfalcon2 + f2_off::MOD_SEL, 1);
}

/// Prepare FBIF + DMACTL before a DMA load (ga102_flcn_fw_load preamble).
pub fn fw_load_prep<R, W>(pfalcon: u64, r: &R, w: &W)
where R: Fn(u64) -> u32, W: Fn(u64, u32) {
    // mask(0x624, 0x80, 0x80): FBIF_CTL allow_phys_no_ctx
    let ctl = r(pfalcon + fb_off::FBIF_CTL);
    w(pfalcon + fb_off::FBIF_CTL, ctl | 0x80);
    // DMACTL = 0
    w(pfalcon + fb_off::DMACTL, 0);
    // mask(0x600, 0x10007, (0<<16)|(1<<2)|1): FBIF_TRANSCFG target/mem-type
    let tc = r(pfalcon + fb_off::FBIF_TRANSCFG);
    w(pfalcon + fb_off::FBIF_TRANSCFG, (tc & !0x10007) | (1 << 2) | 1);
}

/// Parse an nvfw booter/bootloader blob into load parameters.
/// Returns None if the magic/structure doesn't match.
pub fn parse_hs_fw(blob: &[u8]) -> Option<HsFwLoad> {
    if blob.len() < 24 || rd32(blob, 0) != 0x0000_10de {
        return None;
    }
    // nvfw_bin_hdr
    let header_offset = rd32(blob, 0x0c) as usize;
    let data_off = rd32(blob, 0x10) as usize;
    let data_size = rd32(blob, 0x14) as usize;
    if header_offset + 36 > blob.len() {
        return None;
    }
    // nvfw_hs_header_v2 @ header_offset
    let h = header_offset;
    let sig_prod_offset = rd32(blob, h);
    let sig_prod_size = rd32(blob, h + 4);
    let patch_loc = rd32(blob, h + 8);
    let patch_sig = rd32(blob, h + 12);
    let meta_off = rd32(blob, h + 16) as usize;
    let num_sig_off = rd32(blob, h + 24) as usize;
    // nouveau: cnt = *(u32*)(blob + num_sig_off); sig = *(u32*)(blob + patch_sig).
    let num_sig = if num_sig_off + 4 <= blob.len() { rd32(blob, num_sig_off) } else { 1 };
    let sig = if (patch_sig as usize) + 4 <= blob.len() { rd32(blob, patch_sig as usize) } else { 0 };
    let load_hdr_off = rd32(blob, h + 28) as usize;
    if load_hdr_off + 20 + 16 > blob.len() {
        return None;
    }
    // nvfw_hs_load_header_v2 @ load_hdr_off
    let l = load_hdr_off;
    let os_data_offset = rd32(blob, l + 8);
    let os_data_size = rd32(blob, l + 12);
    let num_apps = rd32(blob, l + 16);
    if num_apps < 1 {
        return None;
    }
    // app[0] immediately follows the 5 u32 header words.
    let app0_offset = rd32(blob, l + 20);
    let app0_size = rd32(blob, l + 24);
    // loc = *(u32*)(blob + patch_loc)
    let loc = if (patch_loc as usize) + 4 <= blob.len() { rd32(blob, patch_loc as usize) } else { 0 };
    // meta_data u32[3] = {fuse_ver, engine_id, ucode_id}
    let (fuse_ver, engine_id, ucode_id) = if meta_off + 12 <= blob.len() {
        (rd32(blob, meta_off), rd32(blob, meta_off + 4), rd32(blob, meta_off + 8))
    } else { (0, 0, 0) };

    Some(HsFwLoad {
        data_off, data_size,
        imem_base_img: app0_offset,
        imem_size: app0_size,
        dmem_base_img: os_data_offset,
        dmem_size: os_data_size,
        dmem_sign: loc.wrapping_sub(os_data_offset),
        boot_addr: app0_offset,
        patch_loc, sig_prod_offset, sig_prod_size, num_sig,
        sig_base: sig_prod_offset + sig,
        fuse_ver, engine_id, ucode_id,
    })
}
