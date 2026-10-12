//! FWSEC-FRTS: run NVIDIA's signed FWSEC ucode (extracted from the board's own
//! VBIOS) on the GSP falcon to establish the FRTS region and open WPR2.
//!
//! WHY THIS EXISTS
//! ---------------
//! Our SEC2 "Booter Load" run returns mailbox0 = 0x91
//! (FLCN_ERR_INIT_CRYPTO_DEVICE_FAILED).  A deep source audit established that
//! FWSEC-FRTS is an UNCONDITIONAL predecessor of Booter Load on GA104:
//!
//!   * OpenRM binds the "skip FRTS" stub (kgspGetFrtsSize returning 0) ONLY to
//!     ChipHal GA100 (g_kernel_gsp_nvoc.c:988-994, chip mask 0x00000400).  Our
//!     GA104 falls to the else branch = kgspGetFrtsSize_TU102 = hard-coded 1MB,
//!     so the `if (frtsSize > 0)` gate in kgspBootstrap_TU102:472 is ALWAYS true.
//!   * kgspBootstrap_TU102:471-498 runs FWSEC then Booter Load in that order, in
//!     the same function.
//!   * nouveau agrees: upstream-tu102.c:376 sets frts.size = 0x100000 for every
//!     chipset except GA100, and comments that GA100 is the only chip where "the
//!     booter firmware handles WPR2 setup directly" - implying ours does not.
//!   * Our probe reads WPR2_ADDR_HI (0x1fa828) == 0, which by OpenRM's own
//!     success test (kernel_gsp_frts_tu102.c:503-509) is positive proof that
//!     FWSEC has never run on this GPU in our boot path.
//!
//! We use NVIDIA's own signed FWSEC image, unmodified.  The only bytes we change
//! are the command arguments in its DMEM, exactly as the driver is meant to.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

// --- VBIOS container constants (kernel_gsp_fwsec.c:40-200) -----------------

const BIT_HEADER_ID: u16 = 0xB8FF;
const BIT_HEADER_SIGNATURE: u32 = 0x0054_4942; // "BIT\0"
const BIT_TOKEN_FALCON_DATA: u8 = 0x70;
const FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: u8 = 0x85;

/// NVGI IFR signature that prefixes a raw ROM dump before the PCI image.
const IFR_SIGNATURE_NVGI: u32 = 0x4947_564E; // "NVGI"

// --- FWSEC application-interface constants (kernel_gsp_frts_tu102.c:60-102) -

const FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER: u32 = 0x4;
const DMEM_MAPPER_V3_CMD_FRTS: u32 = 0x15;
const FWSECLIC_READ_VBIOS_STRUCT_FLAGS: u32 = 2;
const FWSECLIC_FRTS_REGION_MEDIA_FB: u32 = 2;
const FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K: u32 = 0x100;

/// Offset of `init_cmd` inside FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.
/// Layout: signature u32, version u16, size u16, then 8 u32s
/// (cmd_in_buffer_offset, cmd_in_buffer_size, cmd_out_buffer_offset,
///  cmd_out_buffer_size, nvf_img_data_buffer_offset, nvf_img_data_buffer_size,
///  printfBufferHdr, ucode_build_time_stamp, ucode_signature) then init_cmd.
const DMEM_MAPPER_INIT_CMD_OFF: usize = 4 + 2 + 2 + (9 * 4);
/// Offset of `cmd_in_buffer_offset` in the same struct.
const DMEM_MAPPER_CMD_IN_OFF: usize = 4 + 2 + 2;
/// Offset of `cmd_in_buffer_size`.
const DMEM_MAPPER_CMD_IN_SIZE_OFF: usize = DMEM_MAPPER_CMD_IN_OFF + 4;

#[inline]
fn rd8(b: &[u8], o: usize) -> Option<u8> { b.get(o).copied() }
#[inline]
fn rd16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]))
}
#[inline]
fn rd32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes([*b.get(o)?, *b.get(o + 1)?, *b.get(o + 2)?, *b.get(o + 3)?]))
}

/// A parsed FWSEC ucode ready to be loaded onto a falcon.
pub struct FwsecUcode {
    /// Whole stored image (IMEM part followed by DMEM part).
    pub image: Vec<u8>,
    pub imem_phys_base: u32,
    pub imem_load_size: u32,
    pub imem_virt_base: u32,
    pub dmem_phys_base: u32,
    pub dmem_load_size: u32,
    /// Byte offset of the application interface header within the DMEM blob.
    pub interface_offset: u32,
    pub engine_id_mask: u16,
    pub ucode_id: u8,
    pub signature_count: u8,
    /// V3 only: offset of the PKC signature data.
    pub pkc_data_offset: u32,
    pub desc_version: u8,
    /// V3 only: bitmap of which signature versions the descriptor carries.
    pub signature_versions: u16,
    /// The descriptor's trailing signature area (descSize - 44 bytes for V3).
    pub signatures: Vec<u8>,
}

impl FwsecUcode {
    /// The DMEM portion of the stored image (everything after IMEM).
    fn dmem_slice_mut(&mut self) -> Option<&mut [u8]> {
        let start = self.imem_load_size as usize;
        let end = start.checked_add(self.dmem_load_size as usize)?;
        self.image.get_mut(start..end)
    }
}

/// Locate the PCI expansion ROM inside a raw VBIOS dump.
///
/// A dump read from the board (or from `NV_PROM`) may be prefixed by an NVGI
/// "IFR" block, in which case the PCI image (0x55AA) starts later.  OpenRM does
/// the same thing in s_romImgFindPciHeader_TU102 (kernel_gsp_vbios_tu102.c:204)
/// and thereafter treats that offset as index 0 of the image.
pub fn find_pci_image_base(rom: &[u8]) -> Option<usize> {
    if rd16(rom, 0) == Some(0xAA55) {
        return Some(0);
    }
    // NVGI IFR present: scan for the first 0x55AA that has a valid PCIR.
    let _ = rd32(rom, 0).filter(|&s| s == IFR_SIGNATURE_NVGI);
    let limit = core::cmp::min(rom.len(), 0x20_0000);
    let mut o = 0usize;
    while o + 0x1a < limit {
        if rd16(rom, o) == Some(0xAA55) {
            let p = o + rd16(rom, o + 0x18)? as usize;
            if rom.get(p..p + 4) == Some(b"PCIR") {
                return Some(o);
            }
        }
        o += 512; // PCI ROM images are 512-byte aligned
    }
    None
}

/// Walk the PCI expansion ROM chain to find the offset of the extended
/// (type != base) image, which is the base for falcon-ucode-table pointers.
///
/// Mirrors s_locateExpansionRoms (kernel_gsp_vbios_tu102.c:278).  We terminate
/// on a missing PCIR signature rather than trusting the NPDE "last image" bit,
/// because on some dumps that bit walks the parser off into garbage.
pub fn locate_expansion_rom_offset(img: &[u8]) -> u32 {
    let mut cur = 0usize;
    let mut ext_rom_offset: u32 = 0;
    let mut base_rom_size: u32 = 0;

    for _ in 0..16 {
        if rd16(img, cur) != Some(0xAA55) {
            break;
        }
        let pcir = match rd16(img, cur + 0x18) {
            Some(v) => cur + v as usize,
            None => break,
        };
        if img.get(pcir..pcir + 4) != Some(b"PCIR") {
            break;
        }
        let ds_len = rd16(img, pcir + 0x0a).unwrap_or(0) as usize;
        let img_len = rd16(img, pcir + 0x10).unwrap_or(0) as u32;
        let code_type = rd8(img, pcir + 0x14).unwrap_or(0);
        let mut last = (rd8(img, pcir + 0x15).unwrap_or(0) & 0x80) != 0;
        let mut sub_len = img_len;

        // Optional NVIDIA PCI Data Extension ("NPDE"), 16-byte aligned.
        let ext = (pcir + ds_len + 0xF) & !0xF;
        if img.get(ext..ext + 4) == Some(b"NPDE") {
            let ext_len = rd16(img, ext + 6).unwrap_or(0) as usize;
            sub_len = rd16(img, ext + 8).unwrap_or(sub_len as u16) as u32;
            if 0x0a + 1 <= ext_len {
                last = (rd8(img, ext + 0x0a).unwrap_or(0) & 0x80) != 0;
            } else if sub_len < img_len {
                last = false;
            }
        }

        let block_offset = cur as u32;
        let block_size = sub_len.saturating_mul(512);
        // Base image is code type 0; the first non-base image is the extended
        // region that falcon-ucode pointers are relative to.  We want that
        // image's SIZE, not its offset (verified below).
        if ext_rom_offset == 0 && code_type != 0 {
            ext_rom_offset = block_size;
            let _ = block_offset;
        } else if base_rom_size == 0 && code_type == 0 {
            base_rom_size = block_size;
        }
        if last || sub_len == 0 {
            break;
        }
        cur += (sub_len as usize) * 512;
    }
    let _ = base_rom_size;
    // VERIFIED against the real VBIOS (vbios-94.06.25.00.80.rom): the chain is
    //   [0x00000 type 0  size 0xFE00]  [0xFE00 type 3  size 0x16A00]
    // and stops at 0x26800, which holds an NVIDIA "VN" extended header rather
    // than another 0x55AA+PCIR image.  The falcon ucode table pointer
    // (0x72BEF) and every entry DescPtr only resolve when biased by 0x16A00 -
    // i.e. the type-3 image's block size.  With that bias the table lands at
    // image-rel 0x895EF (file 0x927EF) and FWSEC_PROD's descriptor at file
    // 0x4C5B8, whose IMEMLoadSize+DMEMLoadSize exactly equals its StoredSize.
    ext_rom_offset
}

/// Find the BIT header inside the PCI image, validating its checksum.
fn find_bit_header(img: &[u8]) -> Option<usize> {
    let limit = img.len().saturating_sub(8);
    for a in 0..limit {
        if rd16(img, a) == Some(BIT_HEADER_ID) && rd32(img, a + 2) == Some(BIT_HEADER_SIGNATURE) {
            let hdr_size = rd8(img, a + 8)? as usize;
            let mut sum: u8 = 0;
            for j in 0..hdr_size {
                sum = sum.wrapping_add(*img.get(a + j)?);
            }
            if sum == 0 {
                return Some(a);
            }
        }
    }
    None
}

/// Parse FWSEC_PROD out of a raw VBIOS image.
///
/// `rom` is the raw dump; everything internal is done against the PCI image
/// slice so that pointer arithmetic matches OpenRM exactly.
pub fn parse_fwsec_from_vbios(rom: &[u8], log: &mut Vec<String>) -> Option<FwsecUcode> {
    let pci_base = find_pci_image_base(rom)?;
    let img = rom.get(pci_base..)?;
    log.push(format!("  FWSEC: pci image base=0x{:x} (image len 0x{:x})", pci_base, img.len()));

    let ero = locate_expansion_rom_offset(img);
    log.push(format!("  FWSEC: expansionRomOffset=0x{:x}", ero));

    let bit = find_bit_header(img)?;
    let hdr_size = rd8(img, bit + 8)? as usize;
    let token_size = rd8(img, bit + 9)? as usize;
    let token_count = rd8(img, bit + 10)? as usize;
    log.push(format!(
        "  FWSEC: BIT@0x{:x} hdr={} tokSz={} tokens={}",
        bit, hdr_size, token_size, token_count
    ));

    // Find the FALCON_DATA token (id 0x70, DataVersion 2).
    let mut table_ptr: Option<u32> = None;
    for i in 0..token_count {
        let t = bit + hdr_size + i * token_size;
        if rd8(img, t)? != BIT_TOKEN_FALCON_DATA || rd8(img, t + 1)? != 2 {
            continue;
        }
        if (rd16(img, t + 2)? as usize) < 4 {
            continue;
        }
        // DataPtr is u16 when TokenSize == 6, u32 when >= 8.
        let data_ptr = if token_size >= 8 {
            rd32(img, t + 4)? as usize
        } else {
            rd16(img, t + 4)? as usize
        };
        table_ptr = rd32(img, data_ptr); // BIT_DATA_FALCON_DATA_V2
        log.push(format!(
            "  FWSEC: falconData@0x{:x} tablePtr=0x{:x}",
            data_ptr,
            table_ptr.unwrap_or(0)
        ));
        break;
    }
    let table_ptr = table_ptr?;

    // The ucode table (and every DescPtr) is relative to expansionRomOffset.
    let tbl = (ero as usize).checked_add(table_ptr as usize)?;
    let version = rd8(img, tbl)?;
    let t_hdr = rd8(img, tbl + 1)? as usize;
    let t_ent = rd8(img, tbl + 2)? as usize;
    let t_cnt = rd8(img, tbl + 3)? as usize;
    if version != 1 || t_hdr < 6 || t_ent < 6 {
        log.push(format!(
            "  FWSEC: bad ucode table @0x{:x} (ver={} hdr={} ent={})",
            tbl, version, t_hdr, t_ent
        ));
        return None;
    }
    log.push(format!("  FWSEC: ucode table@0x{:x} entries={}", tbl, t_cnt));

    // Locate the FWSEC_PROD entry.
    let mut desc_off: Option<usize> = None;
    for i in 0..t_cnt {
        let e = tbl + t_hdr + i * t_ent;
        if rd8(img, e)? == FALCON_UCODE_ENTRY_APPID_FWSEC_PROD {
            let dp = rd32(img, e + 2)? as usize;
            desc_off = Some((ero as usize).checked_add(dp)?);
            log.push(format!("  FWSEC: FWSEC_PROD entry[{}] descPtr=0x{:x}", i, dp));
            break;
        }
    }
    let desc = desc_off?;

    // Descriptor header: bit0 verAvail, bit2 encrypted, 15:8 version, 31:16 size.
    let vdesc = rd32(img, desc)?;
    let desc_version = ((vdesc >> 8) & 0xff) as u8;
    let desc_size = ((vdesc >> 16) & 0xffff) as usize;
    let encrypted = (vdesc >> 2) & 1;
    log.push(format!(
        "  FWSEC: desc@0x{:x} vDesc=0x{:08x} V{} size={} enc={}",
        desc, vdesc, desc_version, desc_size, encrypted
    ));

    let (stored_size, pkc_data_offset, interface_offset, imem_phys_base, imem_load_size,
         imem_virt_base, dmem_phys_base, dmem_load_size, engine_id_mask, ucode_id, sig_count,
         sig_versions) =
        match desc_version {
            3 => {
                // "9d1w2b2w"
                let stored = rd32(img, desc + 4)?;
                let pkc = rd32(img, desc + 8)?;
                let iface = rd32(img, desc + 12)?;
                let ipb = rd32(img, desc + 16)?;
                let ils = rd32(img, desc + 20)?;
                let ivb = rd32(img, desc + 24)?;
                let dpb = rd32(img, desc + 28)?;
                let dls = rd32(img, desc + 32)?;
                let eim = rd16(img, desc + 36)?;
                let uid = rd8(img, desc + 38)?;
                let sc = rd8(img, desc + 39)?;
                let sv = rd16(img, desc + 40)?;
                (stored, pkc, iface, ipb, ils, ivb, dpb, dls, eim, uid, sc, sv)
            }
            2 => {
                // "15d" - no PKC/engine fields
                let stored = rd32(img, desc + 4)?;
                let iface = rd32(img, desc + 16)?;
                let ipb = rd32(img, desc + 20)?;
                let ils = rd32(img, desc + 24)?;
                let ivb = rd32(img, desc + 28)?;
                let dpb = rd32(img, desc + 44)?;
                let dls = rd32(img, desc + 48)?;
                (stored, 0, iface, ipb, ils, ivb, dpb, dls, 0, 0, 0, 0)
            }
            _ => {
                log.push(format!("  FWSEC: unsupported desc version {}", desc_version));
                return None;
            }
        };

    // For V3 the signatures follow the fixed 44-byte descriptor body and run to
    // the end of the descriptor's declared size (kernel_gsp_fwsec.c:~990).
    let signatures: Vec<u8> = if desc_version == 3 && desc_size > 44 {
        img.get(desc + 44..desc + desc_size)?.to_vec()
    } else {
        Vec::new()
    };

    // The payload begins after the descriptor's *size field* (not sizeof(struct)).
    let payload = desc.checked_add(desc_size)?;
    let image = img.get(payload..payload.checked_add(stored_size as usize)?)?.to_vec();
    log.push(format!(
        "  FWSEC: payload@0x{:x} len=0x{:x} (IMEM 0x{:x} + DMEM 0x{:x}) iface=0x{:x} ucodeId={} sigs={}B v=0x{:x}",
        payload, stored_size, imem_load_size, dmem_load_size, interface_offset, ucode_id,
        signatures.len(), sig_versions
    ));

    Some(FwsecUcode {
        image,
        imem_phys_base,
        imem_load_size,
        imem_virt_base,
        dmem_phys_base,
        dmem_load_size,
        interface_offset,
        engine_id_mask,
        ucode_id,
        signature_count: sig_count,
        pkc_data_offset,
        desc_version,
        signature_versions: sig_versions,
        signatures,
    })
}

/// Patch the FRTS command into FWSEC's DMEM image.
///
/// Mirrors s_vbiosPatchInterfaceData (kernel_gsp_frts_tu102.c:157-262): walk the
/// application interface table, find the DMEMMAPPER entry, set `init_cmd` to
/// CMD_FRTS and copy the FRTS command block into the mapper's input buffer.
///
/// `frts_offset` is the desired FB byte offset of the FRTS region.
pub fn patch_frts_command(
    uc: &mut FwsecUcode,
    frts_offset: u64,
    log: &mut Vec<String>,
) -> bool {
    let iface = uc.interface_offset as usize;
    let dmem = match uc.dmem_slice_mut() {
        Some(d) => d,
        None => {
            log.push(String::from("  FWSEC: DMEM slice out of range"));
            return false;
        }
    };
    if iface + 4 > dmem.len() {
        log.push(String::from("  FWSEC: interfaceOffset beyond DMEM"));
        return false;
    }

    // FALCON_APPLICATION_INTERFACE_HEADER_V1 { u8 version, headerSize, entrySize, entryCount }
    let hdr_size = dmem[iface + 1] as usize;
    let entry_size = dmem[iface + 2] as usize;
    let entry_count = dmem[iface + 3] as usize;
    if entry_count < 2 {
        log.push(format!("  FWSEC: too few interface entries ({})", entry_count));
        return false;
    }

    // Find the DMEMMAPPER entry: { u32 id, u32 dmemOffset }
    let mut mapper_off: Option<usize> = None;
    let mut cur = iface + hdr_size;
    for _ in 0..entry_count {
        if cur + 8 > dmem.len() {
            break;
        }
        let id = u32::from_le_bytes([dmem[cur], dmem[cur + 1], dmem[cur + 2], dmem[cur + 3]]);
        let off = u32::from_le_bytes([dmem[cur + 4], dmem[cur + 5], dmem[cur + 6], dmem[cur + 7]])
            as usize;
        if id == FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER {
            mapper_off = Some(off);
        }
        cur += entry_size;
    }
    let mapper = match mapper_off {
        Some(m) if m + DMEM_MAPPER_INIT_CMD_OFF + 4 <= dmem.len() => m,
        _ => {
            log.push(String::from("  FWSEC: DMEMMAPPER entry not found"));
            return false;
        }
    };

    // init_cmd = CMD_FRTS
    dmem[mapper + DMEM_MAPPER_INIT_CMD_OFF..mapper + DMEM_MAPPER_INIT_CMD_OFF + 4]
        .copy_from_slice(&DMEM_MAPPER_V3_CMD_FRTS.to_le_bytes());

    let cmd_in_off = u32::from_le_bytes([
        dmem[mapper + DMEM_MAPPER_CMD_IN_OFF],
        dmem[mapper + DMEM_MAPPER_CMD_IN_OFF + 1],
        dmem[mapper + DMEM_MAPPER_CMD_IN_OFF + 2],
        dmem[mapper + DMEM_MAPPER_CMD_IN_OFF + 3],
    ]) as usize;
    let cmd_in_size = u32::from_le_bytes([
        dmem[mapper + DMEM_MAPPER_CMD_IN_SIZE_OFF],
        dmem[mapper + DMEM_MAPPER_CMD_IN_SIZE_OFF + 1],
        dmem[mapper + DMEM_MAPPER_CMD_IN_SIZE_OFF + 2],
        dmem[mapper + DMEM_MAPPER_CMD_IN_SIZE_OFF + 3],
    ]) as usize;

    // FWSECLIC_FRTS_CMD = READ_VBIOS_DESC { u32 version, u32 size, u64 gfwImageOffset,
    // u32 gfwImageSize, u32 flags } followed by FRTS_REGION_DESC { u32 version,
    // u32 size, u32 frtsRegionOffset4K, u32 frtsRegionSize, u32 frtsRegionMediaType }.
    let mut cmd = [0u8; 24 + 20];
    // readVbiosDesc
    cmd[0..4].copy_from_slice(&1u32.to_le_bytes());   // version
    cmd[4..8].copy_from_slice(&24u32.to_le_bytes());  // size
    cmd[8..16].copy_from_slice(&0u64.to_le_bytes());  // gfwImageOffset
    cmd[16..20].copy_from_slice(&0u32.to_le_bytes()); // gfwImageSize
    cmd[20..24].copy_from_slice(&FWSECLIC_READ_VBIOS_STRUCT_FLAGS.to_le_bytes());
    // frtsRegionDesc
    cmd[24..28].copy_from_slice(&1u32.to_le_bytes());  // version
    cmd[28..32].copy_from_slice(&20u32.to_le_bytes()); // size
    cmd[32..36].copy_from_slice(&(((frts_offset >> 12) as u32)).to_le_bytes()); // offset in 4K
    cmd[36..40].copy_from_slice(&FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K.to_le_bytes());
    cmd[40..44].copy_from_slice(&FWSECLIC_FRTS_REGION_MEDIA_FB.to_le_bytes());

    if cmd_in_size < cmd.len() {
        log.push(format!(
            "  FWSEC: cmd_in_buffer too small ({} < {})",
            cmd_in_size,
            cmd.len()
        ));
    }
    if cmd_in_off + cmd.len() > dmem.len() {
        log.push(String::from("  FWSEC: cmd_in_buffer_offset out of range"));
        return false;
    }
    dmem[cmd_in_off..cmd_in_off + cmd.len()].copy_from_slice(&cmd);

    log.push(format!(
        "  FWSEC: patched FRTS cmd (mapper@0x{:x} cmd_in@0x{:x} size={}) frts4K=0x{:x}",
        mapper,
        cmd_in_off,
        cmd_in_size,
        (frts_offset >> 12) as u32
    ));
    true
}

/// Read WPR2 and report whether FWSEC succeeded in establishing it.
/// Per kernel_gsp_frts_tu102.c:503-509 a nonzero WPR2_ADDR_HI means success.
pub fn wpr2_is_up<R: Fn(u64) -> u32>(bar0: u64, rd: &R) -> (bool, u32, u32) {
    let lo = rd(bar0 + 0x1fa824);
    let hi = rd(bar0 + 0x1fa828);
    (hi != 0, lo, hi)
}

/// RSA3K signature size used by every BCRT30 falcon ucode.
pub const RSA3K_SIG_SIZE: usize = 384;

/// Select the correct signature for this board and patch it into the ucode's
/// DMEM at `PKCDataOffset`.
///
/// The V3 descriptor is followed immediately by `descSize - 44` bytes of
/// signatures (`SignatureCount` of them, 384 bytes each).  Which one applies
/// depends on the board's fuse version for this ucode id:
///
///   OpenRM  kernel_gsp_frts_tu102.c:357-397
///   nouveau ga102_flcn_fw_signature (nouveau-ga102.c:99-125)
///
/// both do: read fuse at 0x8241c0 + (ucodeId-1)*4, take BIT(fls(fuse)), then
/// walk the descriptor's SignatureVersions bitmap accumulating sigSize for each
/// present-but-not-matching version, and finally copy that signature to
/// `DMEM + hsSigDmemAddr`.  Skipping this leaves the BROM without a signature to
/// verify against - the ucode then loads and halts without doing its work,
/// which is exactly the "mbox0 = 0 but WPR2 never appears" we observed.
///
/// `sig_blob` is the descriptor's trailing signature area.
/// Returns the selected index on success.
pub fn patch_signature<R: Fn(u64) -> u32>(
    uc: &mut FwsecUcode,
    sig_blob: &[u8],
    bar0: u64,
    rd: &R,
    log: &mut Vec<String>,
) -> Option<usize> {
    if uc.desc_version != 3 {
        log.push(String::from("  FWSEC sig: only V3 descriptors carry PKC signatures"));
        return None;
    }
    // Fuse version for this ucode id (1-indexed register array).
    let fuse_reg = bar0 + 0x8241c0 + ((uc.ucode_id as u64).saturating_sub(1)) * 4;
    let fuse_raw = rd(fuse_reg);
    // BIT(fls(x)): highest set bit index, then 1 << (idx+1) per nouveau's fls()
    // (fls returns a 1-based bit position, so fls(0)=0 -> BIT(0)=1).
    let fls = 32 - fuse_raw.leading_zeros(); // 1-based position of the top set bit
    let mut reg_version: u32 = 1u32 << fls;
    let mut sig_versions: u32 = uc.signature_versions as u32;

    if reg_version & sig_versions == 0 {
        log.push(format!(
            "  FWSEC sig: fuse@0x{:x}=0x{:08x} -> BIT({})=0x{:x} not present in sigVersions=0x{:x} (UNSUPPORTED)",
            fuse_reg, fuse_raw, fls, reg_version, sig_versions));
        return None;
    }

    let mut idx = 0usize;
    let mut sig_offset = 0usize;
    while (reg_version & sig_versions & 1) == 0 {
        if sig_versions & 1 != 0 {
            sig_offset += RSA3K_SIG_SIZE;
            idx += 1;
        }
        sig_versions >>= 1;
        reg_version >>= 1;
    }

    if sig_offset + RSA3K_SIG_SIZE > sig_blob.len() {
        log.push(format!(
            "  FWSEC sig: offset 0x{:x}+0x{:x} exceeds signature blob 0x{:x}",
            sig_offset, RSA3K_SIG_SIZE, sig_blob.len()));
        return None;
    }

    let dst = uc.pkc_data_offset as usize;
    let imem = uc.imem_load_size as usize;
    let dmem_len = uc.dmem_load_size as usize;
    if dst + RSA3K_SIG_SIZE > dmem_len {
        log.push(format!(
            "  FWSEC sig: PKCDataOffset 0x{:x}+0x{:x} exceeds DMEM 0x{:x}",
            dst, RSA3K_SIG_SIZE, dmem_len));
        return None;
    }
    let base = imem + dst;
    if base + RSA3K_SIG_SIZE > uc.image.len() {
        log.push(String::from("  FWSEC sig: signature destination outside image"));
        return None;
    }
    uc.image[base..base + RSA3K_SIG_SIZE]
        .copy_from_slice(&sig_blob[sig_offset..sig_offset + RSA3K_SIG_SIZE]);

    log.push(format!(
        "  FWSEC sig: fuse@0x{:x}=0x{:08x} fls={} sigVersions=0x{:x} -> idx={} off=0x{:x}; patched 384B to DMEM+0x{:x}",
        fuse_reg, fuse_raw, fls, uc.signature_versions, idx, sig_offset, dst));
    Some(idx)
}
