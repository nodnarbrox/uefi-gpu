// uefi-gpu — M1.12: load and run the signed GSP booter on the SEC2 falcon.
//
// No OS. Ports nouveau's GA102 Falcon firmware path for SEC2 (base 0x840000):
//   reset -> GA102 FBIF/DMA preamble -> secure IMEM DMA -> DMEM DMA ->
//   program BROM -> gm200 mailbox/bootvec/start/wait sequence.

#![no_main]
#![no_std]

extern crate alloc;
mod gsp;
mod falcon;
mod fwsec;
mod m3;
mod m4;
mod llm;
mod frontier_perf;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::fmt::Write as _;

use gsp::{DmaAllocator, DmaRegion};

use uefi::prelude::*;
use uefi::proto::unsafe_protocol;
use uefi::{Identify, boot};
use uefi_raw::Status as RawStatus;

#[repr(C)]
struct RootBridgeIoAccess {
    read: unsafe extern "efiapi" fn(*mut PciRootBridgeIoProtocol, u32, u64, usize, *mut c_void) -> RawStatus,
    write: unsafe extern "efiapi" fn(*mut PciRootBridgeIoProtocol, u32, u64, usize, *mut c_void) -> RawStatus,
}
#[repr(C)]
#[unsafe_protocol("2f707ebb-4a1a-11d4-9a38-0090273fc14d")]
struct PciRootBridgeIoProtocol {
    parent_handle: *mut c_void, poll_mem: *const c_void, poll_io: *const c_void,
    mem: RootBridgeIoAccess, io: RootBridgeIoAccess, pci: RootBridgeIoAccess,
    copy_mem: *const c_void,
    // Map(This, Operation:u32, HostAddress:*mut, NumberOfBytes:*mut usize, DeviceAddress:*mut u64, Mapping:*mut *mut) -> Status
    map: unsafe extern "efiapi" fn(*mut PciRootBridgeIoProtocol, u32, *mut c_void, *mut usize, *mut u64, *mut *mut c_void) -> RawStatus,
    unmap: *const c_void,
    // AllocateBuffer(This, AllocType:u32, MemType:u32, Pages:usize, HostAddress:*mut *mut, Attributes:u64) -> Status
    allocate_buffer: unsafe extern "efiapi" fn(*mut PciRootBridgeIoProtocol, u32, u32, usize, *mut *mut c_void, u64) -> RawStatus,
    free_buffer: *const c_void, flush: *const c_void,
    get_attributes: *const c_void, set_attributes: *const c_void, configuration: *const c_void,
    segment_number: u32,
}

#[repr(C)]
struct BlockIoMedia {
    media_id: u32,
    removable_media: u8,
    media_present: u8,
    logical_partition: u8,
    read_only: u8,
    write_caching: u8,
    _pad0: [u8; 3],
    block_size: u32,
    io_align: u32,
    last_block: u64,
    lowest_aligned_lba: u64,
    logical_blocks_per_physical_block: u32,
    optimal_transfer_length_granularity: u32,
}

#[repr(C)]
#[unsafe_protocol("964e5b21-6459-11d2-8e39-00a0c969723b")]
struct BlockIoProtocol {
    revision: u64,
    media: *mut BlockIoMedia,
    reset: *const c_void,
    read_blocks: unsafe extern "efiapi" fn(
        *mut BlockIoProtocol, u32, u64, usize, *mut c_void,
    ) -> RawStatus,
    write_blocks: *const c_void,
    flush_blocks: *const c_void,
}

#[repr(C, align(4096))]
struct AlignedProbeBlock([u8; 4096]);
const WIDTH_U32: u32 = 2;
fn pci_addr(b: u8, d: u8, f: u8, r: u32) -> u64 {
    ((b as u64) << 24) | ((d as u64) << 16) | ((f as u64) << 8) | (r as u64 & 0xff) | (((r as u64) & 0xf00) << 24)
}
unsafe fn pci_resource_addr(
    rb: &mut PciRootBridgeIoProtocol,
    b: u8,
    d: u8,
    f: u8,
    index: u32,
) -> u64 {
    let reg = 0x10 + index * 4;
    let lo = unsafe { rb.pci_read32(b, d, f, reg) }.unwrap_or(0);
    if lo & 1 != 0 {
        return (lo & !3) as u64;
    }
    let mut addr = (lo & !0xf) as u64;
    if (lo & 6) == 4 && index < 5 {
        let hi = unsafe { rb.pci_read32(b, d, f, reg + 4) }.unwrap_or(0);
        addr |= (hi as u64) << 32;
    }
    addr
}
impl PciRootBridgeIoProtocol {
    unsafe fn pci_read32(&mut self, b: u8, d: u8, f: u8, r: u32) -> Option<u32> {
        let mut v = 0u32;
        let st = (self.pci.read)(self as *mut _, WIDTH_U32, pci_addr(b,d,f,r), 1, &mut v as *mut u32 as *mut c_void);
        if st == RawStatus::SUCCESS { Some(v) } else { None }
    }
    unsafe fn pci_write32(&mut self, b: u8, d: u8, f: u8, r: u32, mut v: u32) -> bool {
        let st = (self.pci.write)(self as *mut _, WIDTH_U32, pci_addr(b,d,f,r), 1,
                                  &mut v as *mut u32 as *mut c_void);
        st == RawStatus::SUCCESS
    }
    unsafe fn r32(&mut self, phys: u64) -> u32 {
        let mut v = 0u32;
        let st = (self.mem.read)(self as *mut _, WIDTH_U32, phys, 1, &mut v as *mut u32 as *mut c_void);
        if st == RawStatus::SUCCESS { v } else { 0xDEAD_DEAD }
    }
    unsafe fn w32(&mut self, phys: u64, mut val: u32) -> bool {
        let st = (self.mem.write)(self as *mut _, WIDTH_U32, phys, 1, &mut val as *mut u32 as *mut c_void);
        st == RawStatus::SUCCESS
    }
    /// Raw read returning 0xDEADDEAD on failure (for closure use).
    unsafe fn mem_read_raw(&mut self, phys: u64) -> u32 { self.r32(phys) }
    unsafe fn mem_write_raw(&mut self, phys: u64, v: u32) { let _ = self.w32(phys, v); }
}
const NVIDIA: u16 = 0x10DE;

/// DmaAllocator over EFI_PCI_ROOT_BRIDGE_IO (AllocateBuffer + Map, Operation=2).
/// Holds a raw protocol pointer; valid for the lifetime of the probe.
struct RbioDma {
    rb: *mut PciRootBridgeIoProtocol,
}
impl DmaAllocator for RbioDma {
    fn alloc_pages(&mut self, pages: usize) -> Option<DmaRegion> {
        unsafe {
            let mut host: *mut c_void = core::ptr::null_mut();
            // AllocateBuffer(AllocateAnyPages=0, EfiBootServicesData=4, pages, &host, attrs=0)
            let ab = ((*self.rb).allocate_buffer)(self.rb, 0, 4, pages, &mut host, 0);
            if ab != RawStatus::SUCCESS || host.is_null() {
                return None;
            }
            core::ptr::write_bytes(host as *mut u8, 0, pages * 4096);
            let mut nbytes: usize = pages * 4096;
            let mut dev: u64 = 0;
            let mut mapping: *mut c_void = core::ptr::null_mut();
            // Operation 2 = EfiPciOperationBusMasterCommonBuffer
            let mp = ((*self.rb).map)(self.rb, 2, host, &mut nbytes, &mut dev, &mut mapping);
            if mp != RawStatus::SUCCESS || dev == 0 {
                return None;
            }
            Some(DmaRegion { host: host as *mut u8, bus: dev, size: pages * 4096 })
        }
    }
}

/// Find the raw second disk attached by probe.sh (identified by ELF magic) and
/// read it into page-aligned memory.  The returned region contains the entire
/// padded block device; the ELF's own section table supplies exact lengths.
fn read_gsp_disk(dma: &mut RbioDma, log: &mut Vec<String>) -> Option<DmaRegion> {
    let handles = boot::locate_handle_buffer(boot::SearchType::ByProtocol(&BlockIoProtocol::GUID)).ok()?;
    for &handle in handles.iter() {
        let mut bio = match unsafe { boot::open_protocol::<BlockIoProtocol>(
            boot::OpenProtocolParams {
                handle,
                agent: boot::image_handle(),
                controller: None,
            },
            boot::OpenProtocolAttributes::GetProtocol,
        ) } {
            Ok(p) => p,
            Err(_) => continue,
        };
        let media = unsafe { bio.media.as_ref()? };
        let block_size = media.block_size as usize;
        if media.media_present == 0 || block_size == 0 || block_size > 4096 {
            continue;
        }
        let mut probe = AlignedProbeBlock([0; 4096]);
        let status = unsafe {
            (bio.read_blocks)(
                &mut *bio as *mut _,
                media.media_id,
                0,
                block_size,
                probe.0.as_mut_ptr() as *mut c_void,
            )
        };
        if status != RawStatus::SUCCESS || &probe.0[..4] != b"\x7fELF" {
            continue;
        }

        let bytes_u64 = media.last_block.checked_add(1)?.checked_mul(media.block_size as u64)?;
        let bytes = usize::try_from(bytes_u64).ok()?;
        // Reject an accidental tiny ELF or an unrelated large block device.
        if !(60 * 1024 * 1024..=70 * 1024 * 1024).contains(&bytes) {
            log.push(format!("  GSP raw disk has unexpected size {}", bytes));
            continue;
        }
        let region = dma.alloc_pages((bytes + 4095) / 4096)?;
        const CHUNK: usize = 4 * 1024 * 1024;
        let mut done = 0usize;
        while done < bytes {
            let mut amount = core::cmp::min(CHUNK, bytes - done);
            amount -= amount % block_size;
            if amount == 0 {
                amount = bytes - done;
            }
            let st = unsafe {
                (bio.read_blocks)(
                    &mut *bio as *mut _,
                    media.media_id,
                    (done / block_size) as u64,
                    amount,
                    region.host.add(done) as *mut c_void,
                )
            };
            if st != RawStatus::SUCCESS {
                log.push(format!("  GSP raw disk read failed at 0x{:x}: 0x{:x}", done, st.0));
                return None;
            }
            done += amount;
        }
        log.push(format!("  GSP raw disk: {} bytes, block={} io_align={}", bytes, block_size, media.io_align));
        return Some(DmaRegion { size: bytes, ..region });
    }
    None
}

// GSP register bases (from nova-core falcon/gsp.rs): PFalconBase=0x110000, PFalcon2Base=0x111000.
const GSP_BASE: u64      = 0x110000; // PFalconBase
const GSP2_BASE: u64     = 0x111000; // PFalcon2Base (RISC-V/BROM)
const F_HWCFG2: u64      = 0x0f4; // bit31 reset_ready, bit12 mem_scrubbing(0=done), bit10 riscv
const F_ENGINE: u64      = 0x3c0; // bit0 reset
const F_MAILBOX0: u64    = 0x040;
// RISC-V (Peregrine) regs, relative to PFalcon2Base:
const R_BCR_CTRL: u64    = 0x668; // bit4 core_select(0=Falcon,1=RISCV), bit0 valid, bit8 br_fetch
const R_CPUCTL: u64      = 0x388; // bit7 active_stat, bit4 halted
const R_CORE_STATUS: u64 = 0x240; // bit0 active_stat

fn diagnostic_ascii(bytes: &[u8], len: usize) -> String {
    let mut out = String::new();
    for &byte in bytes.iter().take(core::cmp::min(len, bytes.len())) {
        if byte == 0 { break; }
        out.push(if (0x20..=0x7e).contains(&byte) { byte as char } else { '.' });
    }
    out
}

fn diagnostic_hex(bytes: &[u8], len: usize) -> String {
    let mut out = String::new();
    for &byte in bytes.iter().take(core::cmp::min(len, bytes.len())) {
        let _ = write!(&mut out, "{:02x}", byte);
    }
    out
}

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();
    // This application remains in boot services for long inference runs.
    // The boot manager arms a five-minute watchdog before StartImage; the
    // external, GPU-scoped probe timeout remains responsible for hung runs.
    if let Err(error) = boot::set_watchdog_timer(0, 0x10000, None) {
        uefi::println!("FRONTIER_WATCHDOG disable=FAILED error={:?}", error);
        return Status::ABORTED;
    }
    uefi::println!("FRONTIER_WATCHDOG disabled=true");
    let mut log: Vec<String> = Vec::new();
    log.push(String::from("uefi-gpu M1.12: GA102 SEC2 booter load + run"));

    let handles = match boot::locate_handle_buffer(boot::SearchType::ByProtocol(&PciRootBridgeIoProtocol::GUID)) {
        Ok(h)=>h, Err(_)=>{ log.push(String::from("no PciRootBridgeIo")); dump(&log); return Status::ABORTED; }
    };
    for &h in handles.iter() {
        let mut rb = match unsafe { boot::open_protocol::<PciRootBridgeIoProtocol>(
            boot::OpenProtocolParams{handle:h,agent:boot::image_handle(),controller:None},
            boot::OpenProtocolAttributes::GetProtocol) } { Ok(p)=>p, Err(_)=>continue };
        for bus in 0u16..=255 { for dev in 0u8..32 { for func in 0u8..8 {
            let id = match unsafe { rb.pci_read32(bus as u8,dev,func,0x00) } { Some(v)=>v, None=>continue };
            if id==0xFFFF_FFFF || id==0 || (id&0xffff) as u16 != NVIDIA { continue; }
            let class = unsafe { rb.pci_read32(bus as u8,dev,func,0x08) }.unwrap_or(0) >> 24;
            if class != 0x03 { continue; }
            let bar0 = (unsafe { rb.pci_read32(bus as u8,dev,func,0x10) }.unwrap_or(0) & 0xFFFF_FFF0) as u64;
            if bar0==0 { continue; }
            log.push(format!("GPU {:04x} BAR0=0x{:08x} SEC2 falcon @ +0x{:x}",
                (id>>16) as u16, bar0, falcon::SEC2_PFALCON));

            // Confirm that the physical option ROM is visible to this UEFI
            // environment; FWSEC-FRTS is stored in the board's VBIOS.
            let rom_cfg = unsafe { rb.pci_read32(bus as u8, dev, func, 0x30) }.unwrap_or(0);
            let rom_base = (rom_cfg & 0xffff_f800) as u64;
            let rom_enabled = unsafe { rb.pci_write32(bus as u8, dev, func, 0x30, rom_cfg | 1) };
            let rom_magic = if rom_base != 0 { unsafe { rb.r32(rom_base) } } else { 0 };
            log.push(format!("  VBIOS ROM cfg=0x{:08x} base=0x{:x} enable={} head=0x{:08x}",
                rom_cfg, rom_base, rom_enabled, rom_magic));
            log.push(format!("  PROM aperture: [0]=0x{:08x} [0x9200]=0x{:08x} [0x93b0]=0x{:08x}",
                unsafe { rb.r32(bar0 + 0x300000) },
                unsafe { rb.r32(bar0 + 0x300000 + 0x9200) },
                unsafe { rb.r32(bar0 + 0x300000 + 0x93b0) }));

            // FWSEC needs the board's VBIOS.  The file dump we parsed offline has an
            // NVGI/IFR wrapper with the PCI image at 0x9200, but the PROM aperture
            // appears to expose the PCI image directly at offset 0 (PROM[0] reads
            // 0xeb7faa55 = a 0x55AA signature in the low half).  Dump a small window
            // so we can see the real on-card layout before wiring the parser to it.
            {
                let prom = bar0 + 0x300000;
                let mut line = String::from("  PROM[0x00..0x20]:");
                for i in 0..8 { line.push_str(&format!(" {:08x}", unsafe { rb.r32(prom + i * 4) })); }
                log.push(line);
                // The PROM aperture exposes the PCI expansion ROM DIRECTLY at offset 0
                // (PROM[0] = 0x55AA and the "VIDEO" marker follow), unlike our offline
                // file dump which is prefixed by an NVGI/IFR wrapper putting the PCI
                // image at 0x9200.  So on hardware pci_base == 0 and the BIT header
                // should sit near 0x1b0 (0x93b0 - 0x9200).
                //
                // Scan byte-by-byte for the 16-bit id 0xB8FF followed by "BIT\0".
                // Reads are 32-bit, so assemble bytes from dwords rather than assuming
                // any alignment.
                let rbp2: *mut PciRootBridgeIoProtocol = &mut *rb as *mut _;
                let rb8 = |o: u32| -> u8 {
                    let w = unsafe { (*rbp2).mem_read_raw(prom + (o & !3) as u64) };
                    ((w >> ((o & 3) * 8)) & 0xff) as u8
                };
                let mut bit_at: Option<u32> = None;
                for off in 0..0x10000u32 {
                    if rb8(off) == 0xff && rb8(off + 1) == 0xb8
                        && rb8(off + 2) == b'B' && rb8(off + 3) == b'I'
                        && rb8(off + 4) == b'T' && rb8(off + 5) == 0
                    {
                        bit_at = Some(off);
                        break;
                    }
                }
                match bit_at {
                    Some(a) => {
                        let hdr_size = rb8(a + 8);
                        let tok_size = rb8(a + 9);
                        let tok_cnt = rb8(a + 10);
                        log.push(format!(
                            "  PROM: BIT @0x{:x} hdrSize={} tokSize={} tokens={} (file dump: image-rel 0x1b0)",
                            a, hdr_size, tok_size, tok_cnt));
                    }
                    None => log.push(String::from("  PROM: BIT header NOT found in first 64KB")),
                }
            }

            // ROOT-CAUSE CHECK for booter mbox0=0x91 (FLCN_ERR_INIT_CRYPTO_DEVICE_FAILED,
            // a LibCCC/SE crypto-init failure). Per OpenRM kernel_gsp_frts_tu102.c:531-540,
            // a falcon's crypto engine can only init once GPU firmware boot (GFW/DEVINIT)
            // has COMPLETED and the SE priv-level-mask is lowered. If devinit never ran on
            // our bare-metal path, GFW_BOOT_PROGRESS != 0xFF and the booter's crypto init
            // fails => 0x91. dev_gc6_island(_addendum).h (tu102, authoritative):
            //   NV_PGC6_AON_SECURE_SCRATCH_GROUP_05(0) = 0x118234, PROGRESS=[7:0], COMPLETED=0xFF
            //   NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK = 0x118128, RD_L0=bit0(1=lowered)
            let gfw_boot = unsafe { rb.r32(bar0 + 0x118234) };
            let gfw_progress = gfw_boot & 0xff;
            let gfw_plm = unsafe { rb.r32(bar0 + 0x118128) };
            log.push(format!("  GFW_BOOT(0x118234)=0x{:08x} PROGRESS=0x{:02x} completed={} | PLM(0x118128)=0x{:08x} rd_l0={}",
                gfw_boot, gfw_progress, gfw_progress == 0xff, gfw_plm, gfw_plm & 1));

            let _fb = bar0 + GSP_BASE;
            #[cfg(any())] { // old M1.4-M1.12a test blocks disabled: booter run needs a pristine falcon
            let fb = bar0 + GSP_BASE;   // falcon regs
            let f2 = bar0 + GSP2_BASE;  // RISC-V/peregrine regs

            // --- BEFORE ---
            let hw0 = unsafe { rb.r32(fb + F_HWCFG2) };
            log.push(format!("  BEFORE HWCFG2=0x{:08x} reset_ready={} mem_scrub={} riscv_bit={}",
                hw0, (hw0>>31)&1, (hw0>>12)&1, (hw0>>10)&1));
            let bcr0 = unsafe { rb.r32(f2 + R_BCR_CTRL) };
            log.push(format!("  BEFORE BCR_CTRL(0x111668)=0x{:08x} core_select={} valid={}",
                bcr0, (bcr0>>4)&1, bcr0&1));

            // --- reset_eng (nova ga102): ENGINE.reset pulse (set bit0, 10us, clear) ---
            let eng = unsafe { rb.r32(fb + F_ENGINE) };
            unsafe { rb.w32(fb + F_ENGINE, eng | 1); }
            boot::stall(20);
            unsafe { rb.w32(fb + F_ENGINE, eng & !1); }
            // wait mem scrubbing done (HWCFG2 bit12 -> 0)
            let mut scrub_done = false; let mut hw1 = 0;
            for _ in 0..200 { hw1 = unsafe { rb.r32(fb + F_HWCFG2) }; if (hw1>>12)&1==0 { scrub_done=true; break; } boot::stall(100); }
            log.push(format!("  after ENGINE reset: HWCFG2=0x{:08x} reset_ready={} scrub_done={}",
                hw1, (hw1>>31)&1, scrub_done));

            // --- select_core = Falcon (BCR_CTRL core_select=0, valid=1), then poll ---
            unsafe { rb.w32(f2 + R_BCR_CTRL, 0x1); } // core_select=Falcon(0) | valid(1)
            let mut sel_ok = false; let mut bcr1 = 0;
            for _ in 0..100 { bcr1 = unsafe { rb.r32(f2 + R_BCR_CTRL) }; if bcr1 & 1 == 1 { sel_ok=true; break; } boot::stall(50); }
            log.push(format!("  select_core Falcon: BCR_CTRL=0x{:08x} valid={} => {}",
                bcr1, bcr1&1, if sel_ok {"ok"} else {"timeout"}));

            // --- read the CORRECT RISC-V CPUCTL (not plain-falcon 0x110100) ---
            let rcpu = unsafe { rb.r32(f2 + R_CPUCTL) };
            let rstat = unsafe { rb.r32(f2 + R_CORE_STATUS) };
            let sane = rcpu != 0xDEAD_DEAD && (rcpu & 0xffff_0000) != 0xbadf_0000;
            log.push(format!("  RISCV_CPUCTL(0x111388)=0x{:08x} halted={} active={}  CORE_STATUS={} {}",
                rcpu, (rcpu>>4)&1, (rcpu>>7)&1, rstat&1, if sane {"[SANE reg ✓]"} else {"[still sentinel]"}));
            log.push(format!("  MAILBOX0=0x{:08x}", unsafe { rb.r32(fb + F_MAILBOX0) }));
            let hwok = ((hw1>>31)&1)==1;
            log.push(format!("  => M1.4b RESULT: scrub_done={} reset_ready={} core_select_ok={} riscv_cpuctl_sane={} {}",
                scrub_done, hwok, sel_ok, sane,
                if scrub_done && sel_ok && sane { "GSP FALCON RESET+CORE-SELECT OK" } else { "partial" }));

            // --- M1.5a: coherent DMA buffer alloc + Map (foundation for GSP fw load) ---
            // AllocateBuffer(AllocateAnyPages=0, EfiBootServicesData=4, pages, &host, attrs)
            let mut host: *mut c_void = core::ptr::null_mut();
            let pages = 4usize; // 16KB test buffer
            let ab = unsafe { (rb.allocate_buffer)(&mut *rb as *mut _, 0, 4, pages, &mut host, 0) };
            if ab == RawStatus::SUCCESS && !host.is_null() {
                log.push(format!("  DMA AllocateBuffer: host=0x{:016x} ({} pages)", host as usize, pages));
                // Map for common-buffer DMA (EfiPciOperationBusMasterCommonBuffer = 2; 6 was Maximum=invalid)
                let mut nbytes: usize = pages * 4096;
                let mut devaddr: u64 = 0;
                let mut mapping: *mut c_void = core::ptr::null_mut();
                let mp = unsafe { (rb.map)(&mut *rb as *mut _, 2, host, &mut nbytes, &mut devaddr, &mut mapping) };
                if mp == RawStatus::SUCCESS {
                    log.push(format!("  DMA Map OK: device(bus)_addr=0x{:016x} mapped_bytes={} {}",
                        devaddr, nbytes, if devaddr != 0 { "[BUS-ADDRESSABLE ✓]" } else { "[devaddr 0?]" }));
                    // Write a marker to the host buffer to prove CPU access.
                    unsafe { core::ptr::write_volatile(host as *mut u32, 0xCAFEBABE); }
                    let rb_marker = unsafe { core::ptr::read_volatile(host as *const u32) };
                    log.push(format!("  host buffer CPU r/w: wrote 0xCAFEBABE read 0x{:08x} {}",
                        rb_marker, if rb_marker == 0xCAFEBABE { "OK" } else { "FAIL" }));
                } else {
                    log.push(format!("  DMA Map FAILED status=0x{:x}", mp.0));
                }
            } else {
                log.push(format!("  DMA AllocateBuffer FAILED status=0x{:x}", ab.0));
            }

            // --- M1.9: radix3 build self-test with a synthetic ~1MB "ELF" ---
            // Verifies the L0->L1->L2->data chain resolves with real bus addresses.
            {
                let mut dma = RbioDma { rb: &mut *rb as *mut _ };
                // 256 pages = 1MB test image; fill with a recognizable pattern.
                let test_elf = alloc::vec![0xABu8; 256 * 4096];
                match gsp::build_radix3(&mut dma, &test_elf) {
                    Some(r) => {
                        log.push(format!("  RADIX3 built: level0.bus=0x{:x} level1.bus=0x{:x} level2.bus=0x{:x} data.bus=0x{:x} elf={}KB",
                            r.level0.bus, r.level1.bus, r.level2.bus, r.data.bus, r.elf_size/1024));
                        // Verify the chain: L0[0] must equal level1.bus; L2[0] must equal data.bus.
                        let l0e = unsafe { core::ptr::read_volatile(r.level0.host as *const u64) };
                        let l2e = unsafe { core::ptr::read_volatile(r.level2.host as *const u64) };
                        let chain_ok = l0e == r.level1.bus && l2e == r.data.bus && r.level0.bus != 0;
                        log.push(format!("  RADIX3 chain: L0[0]=0x{:x}(want L1 0x{:x}) L2[0]=0x{:x}(want data 0x{:x}) {}",
                            l0e, r.level1.bus, l2e, r.data.bus, if chain_ok {"[CHAIN OK ✓]"} else {"[CHAIN BAD]"}));
                    }
                    None => log.push(String::from("  RADIX3 build FAILED (alloc)")),
                }
            }

            // --- M1.10: read real FB size, compute FB layout, build WprMeta ---
            {
                // NV_USABLE_FB_SIZE_IN_MB aliases AON_SCRATCH_42 at 0x1183a4; value*1MB.
                let fb_mb = unsafe { rb.r32(fb + 0x1183a4 - GSP_BASE) } & 0xffff_ffff;
                let fb_size = (fb_mb as u64) * (1 << 20);
                // Real sizes: gsp ELF 63MB, bootloader ~24KB (staged separately).
                let elf_size: u64 = 63_571_696;
                let bl_size: u64 = 24_684;
                let layout = gsp::compute_fb_layout(fb_size, elf_size, bl_size);
                log.push(format!("  FB_SIZE: {} MB ({} GB)", fb_mb, fb_size >> 30));
                log.push(format!("  layout: heap={}MiB frts_off=0x{:x} elf_off=0x{:x} wpr2=[0x{:x}..0x{:x}]",
                    layout.wpr2_heap_size >> 20, layout.frts_off, layout.elf_off, layout.wpr2_start, layout.wpr2_end));
                // Placeholder DMA addrs (real ones from radix3/bootloader staging next tick).
                let meta = gsp::make_wpr_meta(&layout, 0x1000, elf_size, 0x2000, bl_size, 0x100, 0x200, 0x300, 0x4000, 0x180);
                log.push(format!("  WprMeta: magic=0x{:x} rev={} fbSize=0x{:x} heapOff=0x{:x} nonWpr=0x{:x} {}",
                    meta.magic, meta.revision, meta.fb_size, meta.gsp_fw_heap_offset, meta.non_wpr_heap_offset,
                    if meta.magic == 0xDC3AAE21371A60B3 && layout.wpr2_start < layout.wpr2_end && layout.wpr2_end <= fb_size
                        { "[LAYOUT SANE ✓]" } else { "[LAYOUT BAD]" }));
            }

            // --- M1.12: parse the SEC2 Booter blob (nvfw) + verify load params ---
            {
                static BOOTER: &[u8] = include_bytes!("booter_load.bin");
                match falcon::parse_hs_fw(BOOTER) {
                    Some(h) => {
                        log.push(format!("  BOOTER parse: data@0x{:x} len0x{:x} IMEM(off0x{:x} sz0x{:x}) DMEM(off0x{:x} sz0x{:x})",
                            h.data_off, h.data_size, h.imem_base_img, h.imem_size, h.dmem_base_img, h.dmem_size));
                        log.push(format!("  BOOTER meta: fuse_ver={} engine_id={} ucode_id={} boot_addr=0x{:x} dmem_sign=0x{:x} nsig={}",
                            h.fuse_ver, h.engine_id, h.ucode_id, h.boot_addr, h.dmem_sign, h.num_sig));
                        // Verify against the known-good Python parse.
                        let ok = h.imem_size == 0x8900 && h.dmem_size == 0x6200
                            && h.imem_base_img == 0x100 && h.dmem_base_img == 0x8a00
                            && h.data_off == 0x378 && h.ucode_id == 3 && h.engine_id == 1;
                        log.push(format!("  BOOTER parse {}", if ok { "[MATCHES known-good ✓]" } else { "[MISMATCH]" }));
                    }
                    None => log.push(String::from("  BOOTER parse FAILED (magic/struct)")),
                }
            }
            } // end disabled old-test cfg block

            // --- M1.13: FWSEC-FRTS on the GSP falcon (MUST precede the SEC2 booter) ---
            //
            // On GA104 the FRTS region is unconditionally required: OpenRM binds the
            // "skip FRTS" stub only to GA100, so kgspBootstrap_TU102 always runs
            // FWSEC before Booter Load.  FWSEC is what establishes WPR2 - and our
            // WPR2_ADDR_HI reads 0, i.e. it has never run here.  This is the
            // strongest remaining explanation for the booter's 0x91.
            {
                let rbp3: *mut PciRootBridgeIoProtocol = &mut *rb as *mut _;
                let rd = |addr: u64| -> u32 { unsafe { (*rbp3).mem_read_raw(addr) } };
                let wr = |addr: u64, v: u32| { unsafe { (*rbp3).mem_write_raw(addr, v); } };
                let dly = |us: u64| { boot::stall(us as usize); };

                // Slurp the VBIOS out of the PROM aperture.  On hardware this exposes
                // the PCI expansion ROM directly at offset 0 (verified: PROM[0]=0x55AA
                // + "VIDEO", BIT @0x1b0) - unlike our offline .rom dump which carries an
                // NVGI/IFR wrapper.
                //
                // SIZE MATTERS: the falcon ucode table lives around image offset
                // 0x88000 (~544KB) and the FWSEC payload past it, so a 256KB read
                // truncates the image and the table walk falls off the end (observed:
                // parse failed with a bogus expansionRomOffset).  Read 1MB.
                const VBIOS_LEN: usize = 1024 * 1024;
                let prom = bar0 + 0x300000;
                let mut vbios: Vec<u8> = Vec::with_capacity(VBIOS_LEN);
                {
                    let mut o = 0usize;
                    while o < VBIOS_LEN {
                        let w = rd(prom + o as u64);
                        vbios.extend_from_slice(&w.to_le_bytes());
                        o += 4;
                    }
                }
                log.push(format!("  FWSEC: read {}KB of VBIOS from PROM; head={:02x}{:02x}{:02x}{:02x}",
                    VBIOS_LEN / 1024, vbios[0], vbios[1], vbios[2], vbios[3]));

                match fwsec::parse_fwsec_from_vbios(&vbios, &mut log) {
                    Some(mut uc) => {
                        log.push(format!(
                            "  FWSEC parsed: IMEM 0x{:x}@0x{:x} DMEM 0x{:x}@0x{:x} ucodeId={} engMask=0x{:x} sigs={}",
                            uc.imem_load_size, uc.imem_phys_base, uc.dmem_load_size, uc.dmem_phys_base,
                            uc.ucode_id, uc.engine_id_mask, uc.signature_count));

                        // FRTS goes at the FB offset our layout reserved for it.
                        // Derive the VGA workspace base from HARDWARE exactly as the
                        // booter block does - the default compute_fb_layout() assumes a
                        // 1 MiB workspace at the top of FB, which on this board placed
                        // FRTS 0xe0000 bytes INTO the real VGA workspace and made FWSEC
                        // silently decline to create the region (mbox0=0 but WPR2 HI=0).
                        let frts_off = {
                            let fb_mb = rd(bar0 + 0x1183a4) & 0xffff_ffff;
                            let fb_size = (fb_mb as u64) << 20;
                            let display_fuse = rd(bar0 + 0x820c04);
                            let vga_reg = rd(bar0 + 0x625f04);
                            let pramin_base = fb_size - (1 << 20);
                            let vga_off = if display_fuse & 1 == 0 && vga_reg & 8 != 0 {
                                let reported = ((vga_reg & 0xffff_ff00) as u64) << 8;
                                if reported < pramin_base { fb_size - 0x20000 } else { reported }
                            } else {
                                pramin_base
                            };
                            let layout = gsp::compute_fb_layout_with_vga(
                                fb_size, 63_571_696, 24_684, vga_off);
                            log.push(format!(
                                "  FWSEC: fb=0x{:x} vga_off=0x{:x} frts_off=0x{:x} (frts_end=0x{:x})",
                                fb_size, vga_off, layout.frts_off, layout.frts_off + 0x100000));
                            layout.frts_off
                        };
                        // Select + patch the PKC signature for THIS board before the
                        // image is loaded.  Without it the BROM has nothing to verify
                        // and the ucode halts without doing its work.
                        let sigs = core::mem::take(&mut uc.signatures);
                        let sig_ok = fwsec::patch_signature(&mut uc, &sigs, bar0, &rd, &mut log).is_some();
                        if !sig_ok {
                            log.push(String::from("  FWSEC: signature patch FAILED - continuing to observe"));
                        }
                        if fwsec::patch_frts_command(&mut uc, frts_off, &mut log) {
                            let gpf = bar0 + GSP_BASE;   // FWSEC runs on the GSP falcon
                            let gpf2 = bar0 + GSP2_BASE;

                            let (wpr_up_before, lo_b, hi_b) = fwsec::wpr2_is_up(bar0, &rd);
                            log.push(format!("  FWSEC: WPR2 before LO=0x{:08x} HI=0x{:08x} up={}",
                                lo_b, hi_b, wpr_up_before));

                            let (scrub, sel, _) = falcon::falcon_reset_and_select(gpf, gpf2, &rd, &wr, &dly);
                            log.push(format!("  FWSEC: GSP falcon reset scrub={} sel={}", scrub, sel));

                            falcon::fw_load_prep(gpf, &rd, &wr);
                            // Load by PIO: the image is only ~60KB and this avoids
                            // needing a DMA-visible staging buffer for it.
                            let imem_end = uc.imem_load_size as usize;
                            let dmem_end = imem_end + uc.dmem_load_size as usize;
                            if dmem_end <= uc.image.len() {
                                falcon::pio_imem_wr(gpf, &uc.image[..imem_end],
                                    uc.imem_phys_base, true, (uc.imem_virt_base >> 8) as u16, &wr);
                                falcon::pio_dmem_wr(gpf, &uc.image[imem_end..dmem_end],
                                    uc.dmem_phys_base, &wr);
                                log.push(String::from("  FWSEC: IMEM+DMEM loaded via PIO"));

                                falcon::program_brom(gpf2, uc.pkc_data_offset,
                                    uc.engine_id_mask as u32, uc.ucode_id as u32, &wr);
                                let (mbox0, ok) = falcon::gm200_fw_boot(
                                    gpf, 0, 0, uc.imem_virt_base, 0, &rd, &wr);
                                let cpuctl = rd(gpf + 0x100);
                                let (wpr_up, lo_a, hi_a) = fwsec::wpr2_is_up(bar0, &rd);
                                log.push(format!(
                                    "  *** FWSEC RUN: mbox0=0x{:08x} halt_ok={} CPUCTL=0x{:08x} ***",
                                    mbox0, ok, cpuctl));
                                log.push(format!(
                                    "  *** WPR2 after FWSEC: LO=0x{:08x} HI=0x{:08x} UP={} {} ***",
                                    lo_a, hi_a, wpr_up,
                                    if wpr_up { "<<< FRTS ESTABLISHED >>>" } else { "(not established)" }));
                            } else {
                                log.push(String::from("  FWSEC: image shorter than IMEM+DMEM sizes"));
                            }
                        }
                    }
                    None => log.push(String::from("  FWSEC: parse FAILED (see trace above)")),
                }
            }

            // --- M1.12g: FIRST REAL BOOTER RUN on the GSP falcon ---
            {
                static BOOTER: &[u8] = include_bytes!("booter_load.bin");
                static BOOTLOADER: &[u8] = include_bytes!("../fw/ga102-570/bootloader-570.144.bin");
                let rbp: *mut PciRootBridgeIoProtocol = &mut *rb as *mut _;
                let rd = |addr: u64| -> u32 { unsafe { (*rbp).mem_read_raw(addr) } };
                let wr = |addr: u64, v: u32| { unsafe { (*rbp).mem_write_raw(addr, v); } };

                if let Some(h) = falcon::parse_hs_fw(BOOTER) {
                    // (1) signature select by fuse version. engine_id&1 -> fuse 0x824140+(uid-1)*4.
                    let fuse_reg = 0x824140u64 + (h.ucode_id as u64 - 1) * 4;
                    let fuse_raw = rd(bar0 + fuse_reg);
                    let reg_fuse_ver = if fuse_raw == 0 { 0 } else { 32 - fuse_raw.leading_zeros() }; // fls
                    let idx = (h.fuse_ver as i64 - reg_fuse_ver as i64).max(0) as usize;
                    let per_sig = (h.sig_prod_size / h.num_sig.max(1)) as usize; // 768/2=384
                    log.push(format!("  BOOTER sig: fuse_reg@0x{:x}=0x{:08x} fls={} idx={} per_sig={} cnt={} sig_base=0x{:x}",
                        fuse_reg, fuse_raw, reg_fuse_ver, idx, per_sig, h.num_sig, h.sig_base));

                    // (2) copy data blob to a mutable buffer.
                    let mut buf: Vec<u8> = BOOTER[h.data_off .. h.data_off + h.data_size].to_vec();
                    // (3) patch selected sig: sigs[sig_base + idx*per_sig] -> img@(loc); loc = dmem_base_img+dmem_sign.
                    let loc_file = h.dmem_base_img + h.dmem_sign; // = loc, blob-relative
                    let sig_src = h.sig_base as usize + idx * per_sig;
                    let dst = loc_file as usize;
                    if sig_src + per_sig <= BOOTER.len() && dst + per_sig <= buf.len() {
                        buf[dst..dst + per_sig].copy_from_slice(&BOOTER[sig_src..sig_src + per_sig]);
                        log.push(format!("  BOOTER patched sig: src=0x{:x} dst=0x{:x} {}B", sig_src, dst, per_sig));
                    } else {
                        log.push(format!("  BOOTER sig patch OOB: src=0x{:x} dst=0x{:x} buf={} file={}", sig_src, dst, buf.len(), BOOTER.len()));
                    }

                    // (4) Read the full GSP-RM ELF from the raw second disk, then
                    // stage nouveau's `.fwimage` and `.fwsignature_ga10x` inputs.
                    let mut dma = RbioDma { rb: rbp };
                    let raw_gsp = match read_gsp_disk(&mut dma, &mut log) {
                        Some(region) => region,
                        None => {
                            log.push(String::from("  GSP raw firmware disk not found/readable"));
                            continue;
                        }
                    };
                    let gsp_elf = unsafe { core::slice::from_raw_parts(raw_gsp.host, raw_gsp.size) };
                    let (fwimage_off, fwimage_size) = match gsp::elf64_section_range(gsp_elf, b".fwimage") {
                        Some(v) => v,
                        None => {
                            log.push(String::from("  GSP ELF missing .fwimage"));
                            continue;
                        }
                    };
                    let (signature_off, signature_size) = match gsp::elf64_section_range(gsp_elf, b".fwsignature_ga10x") {
                        Some(v) => v,
                        None => {
                            log.push(String::from("  GSP ELF missing .fwsignature_ga10x"));
                            continue;
                        }
                    };
                    let fwimage = &gsp_elf[fwimage_off..fwimage_off + fwimage_size];
                    let radix3 = match gsp::build_radix3(&mut dma, fwimage) {
                        Some(v) => v,
                        None => {
                            log.push(String::from("  GSP .fwimage/radix3 DMA allocation failed"));
                            continue;
                        }
                    };
                    let sig_dma = match dma.alloc_pages((signature_size + 4095) / 4096) {
                        Some(region) => {
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    gsp_elf.as_ptr().add(signature_off), region.host, signature_size,
                                );
                            }
                            region
                        }
                        None => {
                            log.push(String::from("  GSP signature DMA allocation failed"));
                            continue;
                        }
                    };

                    // Parse RM_RISCV_UCODE_DESC and stage only nvfw_bin_hdr's
                    // data payload, matching r535_gsp_rm_boot_ctor.
                    let bl = match gsp::parse_riscv_bootloader(BOOTLOADER) {
                        Some(v) => v,
                        None => {
                            log.push(String::from("  GSP bootloader nvfw parse failed"));
                            continue;
                        }
                    };
                    let boot_dma = match dma.alloc_pages((bl.payload_size + 4095) / 4096) {
                        Some(region) => {
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    BOOTLOADER.as_ptr().add(bl.payload_offset), region.host, bl.payload_size,
                                );
                            }
                            region
                        }
                        None => {
                            log.push(String::from("  GSP bootloader DMA allocation failed"));
                            continue;
                        }
                    };

                    // Populate every initial-boot WprMeta input from the exact
                    // section/payload sizes and bus addresses above.
                    let fb_mb = rd(bar0 + 0x1183a4) & 0xffff_ffff;
                    let fb_size = (fb_mb as u64) << 20;
                    let display_fuse = rd(bar0 + 0x820c04);
                    let vga_reg = rd(bar0 + 0x625f04);
                    let pramin_base = fb_size - (1 << 20);
                    let vga_off = if display_fuse & 1 == 0 && vga_reg & 8 != 0 {
                        let reported = ((vga_reg & 0xffff_ff00) as u64) << 8;
                        if reported < pramin_base { fb_size - 0x20000 } else { reported }
                    } else {
                        pramin_base
                    };
                    let layout = gsp::compute_fb_layout_with_vga(
                        fb_size, fwimage_size as u64, bl.payload_size as u64, vga_off,
                    );
                    let meta = gsp::make_wpr_meta(
                        &layout,
                        radix3.level0.bus, fwimage_size as u64,
                        boot_dma.bus, bl.payload_size as u64,
                        bl.code_offset as u64, bl.data_offset as u64, bl.manifest_offset as u64,
                        sig_dma.bus, signature_size as u64,
                    );
                    let wpr_region = match dma.alloc_pages(1) {
                        Some(reg) => {
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    &meta as *const _ as *const u8,
                                    reg.host,
                                    core::mem::size_of::<gsp::GspFwWprMeta>(),
                                );
                            }
                            reg
                        }
                        None => {
                            log.push(String::from("  WprMeta DMA allocation failed"));
                            continue;
                        }
                    };
                    let wpr_dma = wpr_region.bus;
                    let handshake = match gsp::stage_boot_handshake(&mut dma) {
                        Some(v) => v,
                        None => {
                            log.push(String::from("  GSP boot handshake DMA allocation failed"));
                            continue;
                        }
                    };
                    let bar1 = unsafe { pci_resource_addr(&mut rb, bus as u8, dev, func, 1) };
                    let bar3 = unsafe { pci_resource_addr(&mut rb, bus as u8, dev, func, 3) };
                    let pci_device_id = ((bus as u16) << 8) | ((dev as u16) << 3) | func as u16;
                    let pci_subsystem_id = unsafe {
                        rb.pci_read32(bus as u8, dev, func, 0x2c)
                    }.unwrap_or(0);
                    let pci_revision_id = unsafe {
                        rb.pci_read32(bus as u8, dev, func, 0x08)
                    }.unwrap_or(0) & 0xff;
                    let init_rpcs = match gsp::preload_init_rpcs(
                        &handshake, bar0, bar1, bar3, pci_device_id,
                        id, pci_subsystem_id, pci_revision_id,
                    ) {
                        Some(v) => v,
                        None => {
                            log.push(String::from("  GSP preboot RPC staging failed"));
                            continue;
                        }
                    };
                    let l0e = unsafe { core::ptr::read_volatile(radix3.level0.host as *const u64) };
                    let l2e = unsafe { core::ptr::read_volatile(radix3.level2.host as *const u64) };
                    log.push(format!("  GSP ELF: .fwimage off=0x{:x} size=0x{:x}; ga10x sig off=0x{:x} size=0x{:x}",
                        fwimage_off, fwimage_size, signature_off, signature_size));
                    log.push(format!("  RADIX3: root=0x{:x} L0[0]=0x{:x} L2[0]=0x{:x} data=0x{:x} {}",
                        radix3.level0.bus, l0e, l2e, radix3.data.bus,
                        if l0e == radix3.level1.bus && l2e == radix3.data.bus { "CHAIN OK" } else { "CHAIN BAD" }));
                    log.push(format!("  GSP bootloader: payload=0x{:x} bus=0x{:x} code=0x{:x} data=0x{:x} manifest=0x{:x} app=0x{:x}",
                        bl.payload_size, boot_dma.bus, bl.code_offset, bl.data_offset, bl.manifest_offset, bl.app_version));
                    log.push(format!("  WprMeta DMA=0x{:x}: radix=0x{:x}/0x{:x} boot=0x{:x}/0x{:x} sig=0x{:x}/0x{:x}",
                        wpr_dma, meta.sysmem_addr_of_radix3_elf, meta.size_of_radix3_elf,
                        meta.sysmem_addr_of_bootloader, meta.size_of_bootloader,
                        meta.sysmem_addr_of_signature, meta.size_of_signature));
                    log.push(format!("  WPR FB: nonWpr=0x{:x}/0x{:x} wpr=[0x{:x}..0x{:x}] heap=0x{:x}/0x{:x} elf=0x{:x} boot=0x{:x}",
                        meta.non_wpr_heap_offset, meta.non_wpr_heap_size,
                        meta.gsp_fw_wpr_start, meta.gsp_fw_wpr_end,
                        meta.gsp_fw_heap_offset, meta.gsp_fw_heap_size,
                        meta.gsp_fw_offset, meta.boot_bin_offset));
                    log.push(format!("  VGA layout: fuse=0x{:08x} reg=0x{:08x} workspace=0x{:x}/0x{:x}; meta_size=0x{:x}",
                        display_fuse, vga_reg, layout.vga_off, layout.vga_size,
                        core::mem::size_of::<gsp::GspFwWprMeta>()));
                    log.push(format!("  GSP queues: shared=0x{:x}/0x{:x} ptes={} cmdq=+0x{:x} msgq=+0x{:x} entries={}",
                        handshake.shared.bus, handshake.shared.size, handshake.pte_count,
                        handshake.cmdq_offset, handshake.msgq_offset, handshake.msg_count));
                    log.push(format!("  GSP preboot RPCs: bars=[0x{:x},0x{:x},0x{:x}] dbdf=0x{:04x} pci=[0x{:08x},0x{:08x},rev=0x{:02x}] platform=KVM/passthru host_page=4096 registry=disable-display wptr={} checksum=[0x{:08x},0x{:08x}]",
                        bar0, bar1, bar3, pci_device_id, id, pci_subsystem_id,
                        pci_revision_id, init_rpcs.write_ptr,
                        init_rpcs.system_checksum, init_rpcs.registry_checksum));
                    log.push(format!("  LIBOS args=0x{:x} RMARGS=0x{:x} logs=[0x{:x},0x{:x},0x{:x}]",
                        handshake.libos.bus, handshake.rmargs.bus, handshake.loginit.bus,
                        handshake.logintr.bus, handshake.logrm.bus));

                    // Stage the complete, signature-patched nvfw data image in a
                    // 256-byte-aligned coherent DMA buffer.  ga102_flcn_fw_load
                    // addresses IMEM from the image base and folds the DMEM image
                    // offset into the DMA base, exactly as falcon::dma_wr does.
                    let fw_region = dma.alloc_pages((buf.len() + 4095) / 4096);
                    let fw_dma = match fw_region {
                        Some(reg) => {
                            unsafe { core::ptr::copy_nonoverlapping(buf.as_ptr(), reg.host, buf.len()); }
                            reg.bus
                        }
                        None => 0,
                    };
                    log.push(format!("  BOOTER DMA staging: bus=0x{:x} size=0x{:x}", fw_dma, buf.len()));

                    // Nouveau's GA100 MC init enables the device masks before
                    // subdevice bring-up.  This is required before SEC2 PRI is
                    // reachable on a cold VFIO-reset GPU.
                    let pmc200_before = rd(bar0 + 0x000200);
                    let pmc600_before = rd(bar0 + 0x000600);
                    let wpr2_lo_before = rd(bar0 + 0x1fa824);
                    let wpr2_hi_before = rd(bar0 + 0x1fa828);
                    wr(bar0 + 0x000200, 0xffff_ffff);
                    wr(bar0 + 0x000600, 0xffff_ffff);
                    let pmc200_after = rd(bar0 + 0x000200);
                    let pmc600_after = rd(bar0 + 0x000600);
                    log.push(format!("  MC enable: PMC200 {:08x}->{:08x} PMC600 {:08x}->{:08x}",
                        pmc200_before, pmc200_after, pmc600_before, pmc600_after));
                    log.push(format!("  WPR2 before: LO=0x{:08x} HI=0x{:08x} decoded=[0x{:x}..0x{:x}]",
                        wpr2_lo_before, wpr2_hi_before,
                        (wpr2_lo_before as u64) << 8, (wpr2_hi_before as u64) << 8));

                    // Nouveau prepares GSP itself in RISC-V mode before it
                    // executes the SEC2 booter.  The booter installs the RM
                    // image and bootloader into WPR2 and then starts this core.
                    let dly = |us: u64| { boot::stall(us as usize); };
                    let (gsp_scrub, gsp_bcr) = falcon::falcon_reset_into_riscv(
                        bar0 + falcon::GSP_PFALCON,
                        bar0 + falcon::GSP_PFALCON2,
                        &rd, &wr, &dly,
                    );
                    // The GSP bootloader consumes the LIBOS init-argument list
                    // from its own mailboxes. SEC2's mailboxes independently
                    // receive WprMeta below when the signed booter is started.
                    wr(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX0,
                        handshake.libos.bus as u32);
                    wr(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX1,
                        (handshake.libos.bus >> 32) as u32);
                    log.push(format!("  GSP prepared: scrub={} BCR_CTRL=0x{:08x} core_select={} br_fetch={} valid={}",
                        gsp_scrub, gsp_bcr, (gsp_bcr >> 4) & 1, (gsp_bcr >> 8) & 1, gsp_bcr & 1));
                    log.push(format!("  GSP boot args mailbox: mbox0=0x{:08x} mbox1=0x{:08x}",
                        rd(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX0),
                        rd(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX1)));

                    // (5) Nouveau constructs gsp->booter.load against
                    // device->sec2->falcon (GA102 base 0x840000), not the GSP
                    // falcon. Reset and select SEC2's Falcon core.
                    let gpf = bar0 + falcon::SEC2_PFALCON;
                    let gpf2 = bar0 + falcon::SEC2_PFALCON2;
                    let (scrub, sel, _) = falcon::falcon_reset_and_select(gpf, gpf2, &rd, &wr, &dly);
                    let bcr_now = rd(gpf2 + 0x668);
                    log.push(format!("  SEC2 falcon reset: scrub={} sel={} BCR_CTRL=0x{:08x} (core_select={} want Falcon 0)",
                        scrub, sel, bcr_now, (bcr_now >> 4) & 1));

                    // (6) Match ga102_flcn_fw_load in full: FBIF_CTL, DMACTL,
                    // TRANSCFG preamble, then secure IMEM DMA followed by DMEM DMA.
                    let ctl_before = rd(gpf + falcon::fb_off::FBIF_CTL);
                    let dmactl_before = rd(gpf + falcon::fb_off::DMACTL);
                    let trans_before = rd(gpf + falcon::fb_off::FBIF_TRANSCFG);
                    falcon::fw_load_prep(gpf, &rd, &wr);
                    let imem_ok = fw_dma != 0 && falcon::dma_wr(
                        gpf, fw_dma, h.imem_base_img, 0, h.imem_size, true, true, &rd, &wr);
                    let dmem_ok = fw_dma != 0 && falcon::dma_wr(
                        gpf, fw_dma, h.dmem_base_img, 0, h.dmem_size, false, false, &rd, &wr);
                    log.push(format!("  LOAD preamble: 624 {:08x}->{:08x} 10c {:08x}->{:08x} 600 {:08x}->{:08x}",
                        ctl_before, rd(gpf + falcon::fb_off::FBIF_CTL),
                        dmactl_before, rd(gpf + falcon::fb_off::DMACTL),
                        trans_before, rd(gpf + falcon::fb_off::FBIF_TRANSCFG)));
                    log.push(format!("  DMA loaded: IMEM 0x{:x} {} DMEM 0x{:x} {} DMATRFCMD=0x{:08x}",
                        h.imem_size, if imem_ok { "OK" } else { "FAIL" },
                        h.dmem_size, if dmem_ok { "OK" } else { "FAIL" },
                        rd(gpf + falcon::fb_off::DMATRFCMD)));

                    // (7) ga102_flcn_fw_boot programs BROM, then delegates to
                    // gm200_flcn_fw_boot for mailbox/bootvec/start/wait ordering.
                    let (mbox0, ok) = if imem_ok && dmem_ok {
                        falcon::program_brom(gpf2, h.dmem_sign, h.engine_id, h.ucode_id, &wr);
                        falcon::gm200_fw_boot(
                            gpf, (wpr_dma & 0xffff_ffff) as u32, (wpr_dma >> 32) as u32,
                            h.boot_addr, 0, &rd, &wr)
                    } else {
                        (rd(gpf + falcon::fb_off::MAILBOX0), false)
                    };
                    let riscv_cpuctl = rd(gpf2 + 0x388);
                    let flcn_cpuctl = rd(gpf + 0x100);
                    let flcn_os = rd(gpf + 0x080);
                    let mbox1_after = rd(gpf + 0x044);
                    let wpr2_lo_after = rd(bar0 + 0x1fa824);
                    let wpr2_hi_after = rd(bar0 + 0x1fa828);
                    let meta_verified_f8 = unsafe { core::ptr::read_volatile(wpr_region.host.add(0xf8) as *const u64) };
                    log.push(format!("  *** BOOTER RUN: mbox0=0x{:08x} mbox1=0x{:08x} halt_ok={} ***", mbox0, mbox1_after, ok));
                    log.push(format!("  flcn CPUCTL(0x100)=0x{:08x} OS(0x80)=0x{:08x} RISCV_CPUCTL=0x{:08x} wpr_dma=0x{:x}",
                        flcn_cpuctl, flcn_os, riscv_cpuctl, wpr_dma));
                    log.push(format!("  WPR2 after: LO=0x{:08x} HI=0x{:08x} decoded=[0x{:x}..0x{:x}] sysmem.verified=0x{:016x}",
                        wpr2_lo_after, wpr2_hi_after,
                        (wpr2_lo_after as u64) << 8, (wpr2_hi_after as u64) << 8,
                        meta_verified_f8));
                    log.push(format!("  *** {} ***", if ok && mbox0 == 0 { "BOOTER SUCCESS — WPR OPENED FOR GSP!" } else { "booter did not report success (debug next)" }));
                    // Match r535_gsp_booter_load by publishing appVersion,
                    // then poll the coherent status queue for GspInitDone.
                    wr(bar0 + falcon::GSP_PFALCON + 0x080, bl.app_version);
                    let gsp_mbox0 = rd(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX0);
                    let gsp_mbox1 = rd(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX1);
                    let gsp_cpuctl = rd(bar0 + falcon::GSP_PFALCON2 + 0x388);
                    let gsp_core = rd(bar0 + falcon::GSP_PFALCON2 + 0x240);
                    let gsp_bcr_after = rd(bar0 + falcon::GSP_PFALCON2 + 0x668);
                    log.push(format!("  GSP after booter: mbox=[0x{:08x},0x{:08x}] CPUCTL=0x{:08x} CORE=0x{:08x} BCR=0x{:08x} OS=0x{:08x}",
                        gsp_mbox0, gsp_mbox1, gsp_cpuctl, gsp_core, gsp_bcr_after,
                        rd(bar0 + falcon::GSP_PFALCON + 0x080)));

                    let cmdq = unsafe { handshake.shared.host.add(handshake.cmdq_offset) };
                    let msgq = unsafe { handshake.shared.host.add(handshake.msgq_offset) };
                    let mut init_done = false;
                    let mut consumed_messages = 0u32;
                    let mut consumed_elements = 0u32;
                    let mut nocat_messages = 0u32;
                    let mut first_fn = 0u32;
                    let mut last_fn = 0u32;
                    let mut last_seq = 0u32;
                    let mut last_result = 0u32;
                    let mut bad_elem_count = 0u32;
                    let mut seq_events = 0u32;
                    let mut seq_status = 0u32;
                    let mut seq_buffer_dwords = 0u32;
                    let mut seq_cmd_index = 0u32;
                    let mut seq_commands = 0u32;
                    let mut seq_last_op = 0u32;
                    let mut seq_last_addr = 0u32;
                    let mut seq_poll_error = 0u32;
                    let mut seq_poll_mask = 0u32;
                    let mut seq_poll_value = 0u32;
                    let mut seq_poll_observed = 0u32;
                    let mut seq_core_ops = 0u32;
                    let mut resume_flags = 0u32;
                    let mut resume_scratch_before = 0u32;
                    let mut resume_scratch_after = 0u32;
                    let mut resume_sec_cpuctl_before = 0u32;
                    let mut resume_sec_cpuctl_after = 0u32;
                    let mut resume_sec_mbox0 = 0u32;
                    let mut resume_sec_mbox1 = 0u32;
                    let mut resume_gsp_cpuctl = 0u32;
                    let mut resume_gsp_bcr = 0u32;
                    // Keep a compact copy of every early message header and
                    // diagnostic payload before acknowledging the ring slot.
                    let mut message_trace = [[0u32; 17]; 16];
                    let mut message_text = [[0u8; 65]; 16];
                    let mut message_aux_text = [[0u8; 65]; 16];
                    let mut message_diag = [[0u8; 64]; 16];
                    let mut message_trace_count = 0usize;
                    for _ in 0..5000 {
                        let msg_count = unsafe { core::ptr::read_volatile(msgq.add(0x0c) as *const u32) };
                        let entry_off = unsafe { core::ptr::read_volatile(msgq.add(0x1c) as *const u32) };
                        if msg_count != 0 && msg_count <= handshake.msg_count && entry_off == 0x1000 {
                            // Drain at most one ring's worth per pass.  Advancing
                            // cmdq.rx.readPtr is the CPU acknowledgement used by
                            // r535_gsp_msgq_recv, and frees slots for later events.
                            for _ in 0..msg_count {
                                let write_ptr = unsafe { core::ptr::read_volatile(msgq.add(0x10) as *const u32) };
                                let read_ptr = unsafe { core::ptr::read_volatile(cmdq.add(0x20) as *const u32) };
                                if write_ptr >= msg_count || read_ptr >= msg_count || write_ptr == read_ptr {
                                    break;
                                }
                                let used = if write_ptr >= read_ptr {
                                    write_ptr - read_ptr
                                } else {
                                    write_ptr + msg_count - read_ptr
                                };
                                let elem = unsafe { msgq.add(entry_off as usize + read_ptr as usize * 0x1000) };
                                let elem_count = unsafe { core::ptr::read_volatile(elem.add(0x28) as *const u32) };
                                if elem_count == 0 || elem_count >= msg_count || elem_count > used {
                                    bad_elem_count = elem_count;
                                    break;
                                }
                                let function = unsafe { core::ptr::read_volatile(elem.add(0x3c) as *const u32) };
                                let sequence = unsafe { core::ptr::read_volatile(elem.add(0x24) as *const u32) };
                                let result = unsafe { core::ptr::read_volatile(elem.add(0x40) as *const u32) };
                                if message_trace_count < message_trace.len() {
                                    let trace = &mut message_trace[message_trace_count];
                                    let read = |off: usize| unsafe {
                                        core::ptr::read_volatile(elem.add(off) as *const u32)
                                    };
                                    let read_bytes = |off: usize, dst: &mut [u8]| {
                                        for (j, byte) in dst.iter_mut().enumerate() {
                                            *byte = unsafe {
                                                core::ptr::read_volatile(elem.add(off + j) as *const u8)
                                            };
                                        }
                                    };
                                    trace[0] = elem_count;
                                    trace[1] = sequence;
                                    trace[2] = read(0x38); // RPC length
                                    trace[3] = function;
                                    trace[4] = result;
                                    trace[5] = read(0x44); // private result
                                    trace[6] = read(0x48); // RPC sequence
                                    trace[7] = read(0x50); // first two payload dwords
                                    trace[8] = read(0x54);
                                    if function == 0x1020 {
                                        // NV2080CtrlNocatJournalInsertRecord fields.
                                        trace[9] = read(0x60) & 0xff; // record type
                                        trace[10] = read(0x64);       // bugcheck
                                        trace[11] = read(0xac);       // subsystem
                                        trace[12] = read(0xb0);       // error code low
                                        trace[13] = read(0xb4);       // error code high
                                        trace[14] = read(0xfc);       // TDR reason
                                        trace[15] = read(0x100);      // diagnostic length
                                        trace[16] = read(0x104);      // first diagnostic dword
                                        read_bytes(0x68, &mut message_text[message_trace_count]);
                                        read_bytes(0xb8, &mut message_aux_text[message_trace_count]);
                                        read_bytes(0x104, &mut message_diag[message_trace_count]);
                                    } else if function == 0x100c {
                                        // rpc_ucode_libos_print_v1E_08: the
                                        // byte buffer follows engDesc and size.
                                        read_bytes(0x58, &mut message_text[message_trace_count]);
                                    }
                                    message_trace_count += 1;
                                }
                                if consumed_messages == 0 { first_fn = function; }
                                last_fn = function;
                                last_seq = sequence;
                                last_result = result;
                                consumed_messages += 1;
                                consumed_elements += elem_count;
                                if function == 0x1020 { nocat_messages += 1; }

                                if function == 0x1002 {
                                    // rpc_run_cpu_sequencer_v17_00 starts at
                                    // queue-envelope 0x30 + RPC header 0x20.
                                    // Its commandBuffer follows two u32 fields
                                    // and the eight-word regSaveArea at +0x28.
                                    seq_events += 1;
                                    let ring_bytes = msg_count as usize * 0x1000;
                                    let read_msg_u32 = |off: usize| -> u32 {
                                        let linear = (read_ptr as usize * 0x1000 + off) % ring_bytes;
                                        unsafe {
                                            core::ptr::read_volatile(
                                                msgq.add(entry_off as usize + linear) as *const u32,
                                            )
                                        }
                                    };
                                    let write_msg_u32 = |off: usize, value: u32| {
                                        let linear = (read_ptr as usize * 0x1000 + off) % ring_bytes;
                                        unsafe {
                                            core::ptr::write_volatile(
                                                msgq.add(entry_off as usize + linear) as *mut u32,
                                                value,
                                            );
                                        }
                                    };
                                    let rpc_len = read_msg_u32(0x38) as usize;
                                    seq_buffer_dwords = read_msg_u32(0x50);
                                    seq_cmd_index = read_msg_u32(0x54);
                                    let event_bytes = elem_count as usize * 0x1000;
                                    let command_bytes = (seq_cmd_index as usize).saturating_mul(4);
                                    if rpc_len < 0x48
                                        || 0x30usize.saturating_add(rpc_len) > event_bytes
                                        || seq_buffer_dwords == 0
                                        || seq_cmd_index >= seq_buffer_dwords
                                        || 0x78usize.saturating_add(command_bytes) > event_bytes
                                    {
                                        seq_status = 1; // malformed sequence header
                                    } else {
                                        let mut ptr = 0u32;
                                        while ptr < seq_cmd_index {
                                            let opcode = read_msg_u32(0x78 + ptr as usize * 4);
                                            ptr += 1;
                                            seq_last_op = opcode;
                                            seq_commands += 1;
                                            let payload_words = match opcode {
                                                0 => 2, // REG_WRITE
                                                1 => 3, // REG_MODIFY
                                                2 => 5, // REG_POLL
                                                3 => 1, // DELAY_US
                                                4 => 2, // REG_STORE
                                                5..=8 => 0, // core operations
                                                _ => {
                                                    seq_status = 8;
                                                    0
                                                }
                                            };
                                            if seq_status != 0 { break; }
                                            if ptr + payload_words > seq_cmd_index {
                                                seq_status = 2; // truncated command
                                                break;
                                            }
                                            let arg = |n: u32| -> u32 {
                                                read_msg_u32(0x78 + (ptr + n) as usize * 4)
                                            };
                                            match opcode {
                                                0 => {
                                                    let addr = arg(0);
                                                    seq_last_addr = addr;
                                                    wr(bar0 + addr as u64, arg(1));
                                                }
                                                1 => {
                                                    let addr = arg(0);
                                                    let mask = arg(1);
                                                    let value = arg(2);
                                                    seq_last_addr = addr;
                                                    let old = rd(bar0 + addr as u64);
                                                    wr(bar0 + addr as u64, (old & !mask) | value);
                                                }
                                                2 => {
                                                    let addr = arg(0);
                                                    let mask = arg(1);
                                                    let value = arg(2);
                                                    let timeout = if arg(3) == 0 { 4_000_000 } else { arg(3) };
                                                    seq_poll_error = arg(4);
                                                    seq_poll_mask = mask;
                                                    seq_poll_value = value;
                                                    seq_last_addr = addr;
                                                    let mut settled = false;
                                                    for _ in 0..timeout {
                                                        seq_poll_observed = rd(bar0 + addr as u64);
                                                        if seq_poll_observed & mask == value {
                                                            settled = true;
                                                            break;
                                                        }
                                                        boot::stall(1);
                                                    }
                                                    if !settled {
                                                        seq_status = 3;
                                                    }
                                                }
                                                3 => boot::stall(arg(0) as usize),
                                                4 => {
                                                    let addr = arg(0);
                                                    let index = arg(1);
                                                    seq_last_addr = addr;
                                                    if index < 8 {
                                                        write_msg_u32(0x58 + index as usize * 4,
                                                            rd(bar0 + addr as u64));
                                                    } else {
                                                        seq_status = 4;
                                                    }
                                                }
                                                5 => {
                                                    // Full nvkm_falcon_reset followed by
                                                    // disable_ctx_req (624 bit7, DMACTL=0).
                                                    seq_core_ops |= 1 << 0;
                                                    let pf = bar0 + falcon::GSP_PFALCON;
                                                    let (scrubbed, selected) =
                                                        falcon::falcon_runtime_reset_to_falcon(
                                                            pf,
                                                            bar0 + falcon::GSP_PFALCON2,
                                                            rd(bar0),
                                                            &rd, &wr, &dly,
                                                        );
                                                    let ctl = rd(pf + falcon::fb_off::FBIF_CTL);
                                                    wr(pf + falcon::fb_off::FBIF_CTL, ctl | 0x80);
                                                    wr(pf + falcon::fb_off::DMACTL, 0);
                                                    if !scrubbed || !selected { seq_status = 5; }
                                                }
                                                6 => {
                                                    seq_core_ops |= 1 << 1;
                                                    let pf = bar0 + falcon::GSP_PFALCON;
                                                    if rd(pf + 0x100) & 0x40 != 0 {
                                                        wr(pf + 0x130, 2);
                                                    } else {
                                                        wr(pf + 0x100, 2);
                                                    }
                                                }
                                                7 => {
                                                    seq_core_ops |= 1 << 2;
                                                    let pf = bar0 + falcon::GSP_PFALCON;
                                                    let mut halted = false;
                                                    for _ in 0..20_000 {
                                                        if rd(pf + 0x100) & 0x10 != 0 {
                                                            halted = true;
                                                            break;
                                                        }
                                                        boot::stall(100);
                                                    }
                                                    if !halted { seq_status = 6; }
                                                }
                                                8 => {
                                                    // GA102 arch-specific CORE_RESUME:
                                                    // reset GSP into RISC-V, republish LIBOS
                                                    // args, then start halted SEC2 to reload it.
                                                    seq_core_ops |= 1 << 3;
                                                    let (scrubbed, bcr) = falcon::falcon_reset_into_riscv(
                                                        bar0 + falcon::GSP_PFALCON,
                                                        bar0 + falcon::GSP_PFALCON2,
                                                        &rd, &wr, &dly,
                                                    );
                                                    wr(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX0,
                                                        handshake.libos.bus as u32);
                                                    wr(bar0 + falcon::GSP_PFALCON + falcon::fb_off::MAILBOX1,
                                                        (handshake.libos.bus >> 32) as u32);
                                                    let sec_pf = bar0 + falcon::SEC2_PFALCON;
                                                    resume_sec_cpuctl_before = rd(sec_pf + 0x100);
                                                    resume_scratch_before = rd(bar0 + 0x1180f8);
                                                    if resume_sec_cpuctl_before & 0x40 != 0 {
                                                        wr(sec_pf + 0x130, 2);
                                                    } else {
                                                        wr(sec_pf + 0x100, 2);
                                                    }
                                                    let mut reloaded = false;
                                                    for _ in 0..20_000 {
                                                        resume_scratch_after = rd(bar0 + 0x1180f8);
                                                        if resume_scratch_after & 0x0400_0000 != 0 {
                                                            reloaded = true;
                                                            break;
                                                        }
                                                        boot::stall(100);
                                                    }
                                                    resume_sec_cpuctl_after = rd(sec_pf + 0x100);
                                                    resume_sec_mbox0 = rd(sec_pf + falcon::fb_off::MAILBOX0);
                                                    resume_sec_mbox1 = rd(sec_pf + falcon::fb_off::MAILBOX1);
                                                    wr(bar0 + falcon::GSP_PFALCON + 0x080, bl.app_version);
                                                    resume_gsp_cpuctl = rd(bar0 + falcon::GSP_PFALCON2 + 0x388);
                                                    resume_gsp_bcr = rd(bar0 + falcon::GSP_PFALCON2 + 0x668);
                                                    if scrubbed { resume_flags |= 1 << 0; }
                                                    if bcr & 0x111 == 0x111 { resume_flags |= 1 << 1; }
                                                    if reloaded { resume_flags |= 1 << 2; }
                                                    if resume_sec_mbox0 == 0 { resume_flags |= 1 << 3; }
                                                    if resume_gsp_cpuctl & 0x80 != 0 { resume_flags |= 1 << 4; }
                                                    if resume_flags != 0x1f {
                                                        seq_status = 7;
                                                    }
                                                }
                                                _ => {}
                                            }
                                            if seq_status != 0 { break; }
                                            ptr += payload_words;
                                        }
                                    }
                                }

                                let mut next = read_ptr + elem_count;
                                if next >= msg_count { next -= msg_count; }
                                core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
                                unsafe { core::ptr::write_volatile(cmdq.add(0x20) as *mut u32, next); }
                                core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
                                if function == gsp::GSP_INIT_DONE {
                                    init_done = true;
                                    break;
                                }
                            }
                        }
                        if init_done { break; }
                        boot::stall(1000);
                    }
                    let tx_ver = unsafe { core::ptr::read_volatile(msgq as *const u32) };
                    let tx_size = unsafe { core::ptr::read_volatile(msgq.add(0x04) as *const u32) };
                    let tx_msg_size = unsafe { core::ptr::read_volatile(msgq.add(0x08) as *const u32) };
                    let tx_count = unsafe { core::ptr::read_volatile(msgq.add(0x0c) as *const u32) };
                    let tx_wptr = unsafe { core::ptr::read_volatile(msgq.add(0x10) as *const u32) };
                    let tx_flags = unsafe { core::ptr::read_volatile(msgq.add(0x14) as *const u32) };
                    let tx_rxoff = unsafe { core::ptr::read_volatile(msgq.add(0x18) as *const u32) };
                    let tx_entryoff = unsafe { core::ptr::read_volatile(msgq.add(0x1c) as *const u32) };
                    let rx_rptr = unsafe { core::ptr::read_volatile(cmdq.add(0x20) as *const u32) };
                    log.push(format!("  GSP MSGQ: v={} size=0x{:x} msg=0x{:x} cnt={} wptr={} rptr={} flags=0x{:x} rxoff=0x{:x} entry=0x{:x}",
                        tx_ver, tx_size, tx_msg_size, tx_count, tx_wptr, rx_rptr,
                        tx_flags, tx_rxoff, tx_entryoff));
                    log.push(format!("  GSP MSGQ drain: messages={} elems={} nocat={} first_fn=0x{:08x} last_fn=0x{:08x} last_seq={} last_result=0x{:08x} bad_elems={}",
                        consumed_messages, consumed_elements, nocat_messages, first_fn,
                        last_fn, last_seq, last_result, bad_elem_count));

                    // --- M2: drive the LIVE RPC protocol now that GSP-RM is up ---
                    //
                    // The preboot RPCs only proved we can stage messages GSP eats
                    // during init.  This issues a real request AFTER GspInitDone and
                    // waits for GSP to answer, which is the gate to VRAM alloc / BAR1.
                        // GET_GSP_STATIC_INFO is the natural first call: nouveau uses its
                        // reply for hInternalClient/Device/Subdevice and bar1/bar2 PDE
                        // bases (upstream-gsp-r535.c:940-958).
                        if init_done {
                            let gsp_falcon = bar0 + GSP_BASE;
                            // Focused SWGEN0 experiment: expose the Falcon's bit-6
                            // interrupt before notifying queue 0.  The GA102 RISC-V
                            // mask/destination are read-only diagnostics here; do not
                            // disturb firmware routing.
                            let irqstat_before = rd(gsp_falcon + 0x008);
                            let irqmask_before = rd(gsp_falcon + 0x018);
                            wr(gsp_falcon + 0x010, 0x0000_0040);
                            let irqmask_after = rd(gsp_falcon + 0x018);
                            let riscv_irqmask = rd(gsp_falcon + 0x1000 + 0x528);
                            let riscv_irqdest = rd(gsp_falcon + 0x1000 + 0x52c);
                            log.push(format!(
                                "  M2 SWGEN0 enable: irqstat=0x{:08x} irqmask=0x{:08x}->0x{:08x} riscv_mask=0x{:08x} riscv_dest=0x{:08x}",
                                irqstat_before, irqmask_before, irqmask_after,
                                riscv_irqmask, riscv_irqdest));
                            // The COMMAND sequence is a monotonic cmdq counter
                        // (nouveau: cmd->sequence = gsp->cmdq.seq++), NOT tied to the
                        // status-ring sequence.  Preboot sent SET_SYSTEM_INFO(seq=0)
                        // and SET_REGISTRY(seq=1), so the next command must be seq=2.
                        // Using last_seq(from the msg ring)+1 = 1 collided with the
                        // registry RPC and GSP dropped it.
                        let seq = 2u32;
                        // GET_GSP_STATIC_INFO is a read-style RPC: nouveau declares
                        // rpc->length = 0x20 + sizeof(GspStaticConfigInfo).  That
                        // struct is 1656 bytes (verified against the r570 bindgen
                        // layout), so argc=1656 (one page total).  A zero-argc request
                        // is a runt GSP silently drops.
                        const GSP_STATIC_INFO_SIZE: usize = 1656;
                        let pushed = unsafe {
                            gsp::rpc_send_argc(
                                &handshake, gsp::RPC_GET_GSP_STATIC_INFO,
                                &[], GSP_STATIC_INFO_SIZE, seq)
                        };
                        match pushed {
                            Some(csum) => {
                                // Snapshot the ring pointers around the doorbell so we
                                // can tell "GSP never saw the command" (cmd_rptr never
                                // advances) apart from "GSP consumed it but did not
                                // reply" (cmd_rptr advances, msg wptr stays put).
                                // GSP's read pointer for OUR commands lives in the
                                // MSGQ's rx header (nouveau: cmdq.rptr = &msgq->rx.readPtr),
                                // NOT the cmdq's own rx header.  Read it from there.
                                let msg_rxoff = unsafe {
                                    core::ptr::read_volatile(msgq.add(0x18) as *const u32) };
                                let cmd_w_before = unsafe {
                                    core::ptr::read_volatile(cmdq.add(0x10) as *const u32) };
                                let cmd_r_before = unsafe {
                                    core::ptr::read_volatile(msgq.add(msg_rxoff as usize) as *const u32) };
                                let msg_w_before = unsafe {
                                    core::ptr::read_volatile(msgq.add(0x10) as *const u32) };
                                gsp::rpc_doorbell(gsp_falcon, &wr);
                                // NV_PGSP_QUEUE_HEAD(0) is an MMIO doorbell.  Force
                                // the posted PCIe write to reach the GPU before we
                                // start polling coherent sysmem for completion.
                                let doorbell_readback = rd(gsp_falcon + 0xc00);
                                log.push(format!(
                                    "  M2 RPC sent: fn={} (GET_GSP_STATIC_INFO) seq={} checksum=0x{:08x} doorbell_readback=0x{:08x}",
                                    gsp::RPC_GET_GSP_STATIC_INFO, seq, csum,
                                    doorbell_readback));
                                let _ = (cmd_r_before, msg_w_before);

                                // Poll the status ring for the reply.
                                let mut got_fn: u32 = 0;
                                let mut got_result: u32 = 0;
                                let mut got_len: u32 = 0;
                                let mut got_data = [0u32; 4];
                                let mut static_valid = false;
                                let mut static_handles = [0u32; 3];
                                let mut static_bar_pde = [0u64; 2];
                                let mut static_fb_length = 0u64;
                                let mut static_fb_count = 0u32;
                                let mut static_fb_regions = [[0u64; 4]; 16];
                                let mut static_fb_flags = [[0u32; 4]; 16];
                                let mut replied = false;
                                for _ in 0..20000 {
                                    let wptr = unsafe {
                                        core::ptr::read_volatile(msgq.add(0x10) as *const u32) };
                                    let rptr = unsafe {
                                        core::ptr::read_volatile(cmdq.add(0x20) as *const u32) };
                                    if wptr != rptr {
                                        let elem = unsafe {
                                            msgq.add(tx_entryoff as usize + rptr as usize * 0x1000) };
                                        got_fn = unsafe {
                                            core::ptr::read_volatile(elem.add(0x3c) as *const u32) };
                                        got_result = unsafe {
                                            core::ptr::read_volatile(elem.add(0x40) as *const u32) };
                                        got_len = unsafe {
                                            core::ptr::read_volatile(elem.add(0x38) as *const u32) };
                                        for (i, d) in got_data.iter_mut().enumerate() {
                                            *d = unsafe { core::ptr::read_volatile(
                                                elem.add(0x44 + i * 4) as *const u32) };
                                        }
                                        if got_fn == gsp::RPC_GET_GSP_STATIC_INFO
                                            && got_result == 0
                                            && got_len as usize >= 0x20 + GSP_STATIC_INFO_SIZE
                                        {
                                            // RPC data starts at message offset 0x50.
                                            // Decode the r570 GspStaticConfigInfo fields
                                            // needed by Nouveau's FB/BAR setup.
                                            let s = unsafe { elem.add(0x50) };
                                            let s32 = |off: usize| unsafe {
                                                core::ptr::read_volatile(s.add(off) as *const u32)
                                            };
                                            let s64 = |off: usize| -> u64 {
                                                s32(off) as u64 | ((s32(off + 4) as u64) << 32)
                                            };
                                            static_fb_length = s64(0x4c8);
                                            static_bar_pde[0] = s64(0x600);
                                            static_bar_pde[1] = s64(0x608);
                                            static_handles[0] = s32(0x640);
                                            static_handles[1] = s32(0x644);
                                            static_handles[2] = s32(0x648);
                                            static_fb_count = core::cmp::min(s32(0x158), 16);
                                            for i in 0..static_fb_count as usize {
                                                let ro = 0x160 + i * 0x30;
                                                static_fb_regions[i][0] = s64(ro);
                                                static_fb_regions[i][1] = s64(ro + 0x08);
                                                static_fb_regions[i][2] = s64(ro + 0x10);
                                                static_fb_regions[i][3] =
                                                    static_fb_regions[i][1]
                                                        .saturating_add(1)
                                                        .saturating_sub(static_fb_regions[i][0]);
                                                static_fb_flags[i][0] = s32(ro + 0x18);
                                                static_fb_flags[i][1] = unsafe {
                                                    core::ptr::read_volatile(s.add(ro + 0x1c)) as u32
                                                };
                                                static_fb_flags[i][2] = unsafe {
                                                    core::ptr::read_volatile(s.add(ro + 0x1d)) as u32
                                                };
                                                static_fb_flags[i][3] = unsafe {
                                                    core::ptr::read_volatile(s.add(ro + 0x1e)) as u32
                                                };
                                            }
                                            static_valid = true;
                                        }
                                        let ec = unsafe {
                                            core::ptr::read_volatile(elem.add(0x28) as *const u32) };
                                        let step = if ec == 0 { 1 } else { ec };
                                        let mut next = rptr + step;
                                        if next >= tx_count { next -= tx_count; }
                                        core::sync::atomic::fence(
                                            core::sync::atomic::Ordering::SeqCst);
                                        unsafe { core::ptr::write_volatile(
                                            cmdq.add(0x20) as *mut u32, next); }
                                        if got_fn == gsp::RPC_GET_GSP_STATIC_INFO {
                                            replied = true;
                                            break;
                                        }
                                    }
                                    boot::stall(100);
                                }
                                let cmd_r_final = unsafe {
                                    core::ptr::read_volatile(msgq.add(msg_rxoff as usize) as *const u32) };
                                let msg_w_final = unsafe {
                                    core::ptr::read_volatile(msgq.add(0x10) as *const u32) };
                                log.push(format!(
                                    "  M2 rings after poll: cmd_w={} cmd_r(msgq.rx)={} msg_w={} (cmd_r reaching cmd_w = GSP consumed our RPC)",
                                    cmd_w_before, cmd_r_final, msg_w_final));
                                log.push(format!(
                                    "  *** M2 RPC reply: matched={} fn={} result=0x{:08x} len=0x{:x} data=[0x{:08x},0x{:08x},0x{:08x},0x{:08x}] ***",
                                    replied, got_fn, got_result, got_len,
                                    got_data[0], got_data[1], got_data[2], got_data[3]));
                                if replied && got_result == 0 {
                                    log.push(String::from(
                                        "  *** M2 SUCCESS: live GSP-RM RPC round-trip working ***"));
                                }
                                if static_valid {
                                    log.push(format!(
                                        "  M2 STATIC: handles=[0x{:08x},0x{:08x},0x{:08x}] bar_pde=[0x{:x},0x{:x}] fb_length=0x{:x} regions={}",
                                        static_handles[0], static_handles[1], static_handles[2],
                                        static_bar_pde[0], static_bar_pde[1],
                                        static_fb_length, static_fb_count));
                                    for i in 0..static_fb_count as usize {
                                        log.push(format!(
                                            "  M2 FB[{}]: [0x{:x}..0x{:x}] size=0x{:x} reserved=0x{:x} perf=0x{:x} comp={} iso={} protected={}",
                                            i, static_fb_regions[i][0], static_fb_regions[i][1],
                                            static_fb_regions[i][3], static_fb_regions[i][2],
                                            static_fb_flags[i][0], static_fb_flags[i][1],
                                            static_fb_flags[i][2], static_fb_flags[i][3]));
                                    }

                                    // Take the first page from an unreserved,
                                    // unprotected FB region as our initial VRAM
                                    // allocation.  Validate it through the BAR0
                                    // PRAMIN window, then restore every byte so
                                    // this bring-up probe leaves no persistent
                                    // marker in firmware-managed VRAM.
                                    let mut scratch = None;
                                    for i in 0..static_fb_count as usize {
                                        let start = (static_fb_regions[i][0] + 0xfff) & !0xfff;
                                        let end = static_fb_regions[i][1];
                                        if static_fb_regions[i][2] == 0
                                            && static_fb_flags[i][3] == 0
                                            && start <= end
                                            && end.saturating_add(1).saturating_sub(start) >= 0x1000
                                        {
                                            scratch = Some((i, start));
                                            break;
                                        }
                                    }
                                    if let Some((region, phys)) = scratch {
                                        // nv50_instmem_set_bar0_window_addr(),
                                        // retained by r535_instmem_new(): the
                                        // 1 MiB FB window selector is addr>>16,
                                        // with aperture at BAR0 + 0x700000.
                                        let window_reg = bar0 + 0x001700;
                                        let aperture = bar0 + 0x700000 + (phys & 0x000f_ffff);
                                        let window_saved = rd(window_reg);
                                        wr(window_reg, ((phys & !0x000f_ffff) >> 16) as u32);
                                        let window_live = rd(window_reg);
                                        let mut saved = [0u32; 4];
                                        let pattern = [
                                            0x4750_5521,
                                            0xa5a5_5a5a,
                                            phys as u32,
                                            ((phys >> 32) as u32) ^ 0x1357_9bdf,
                                        ];
                                        for i in 0..4 {
                                            saved[i] = rd(aperture + (i * 4) as u64);
                                            wr(aperture + (i * 4) as u64, pattern[i]);
                                        }
                                        core::sync::atomic::fence(
                                            core::sync::atomic::Ordering::SeqCst);
                                        let mut observed = [0u32; 4];
                                        for i in 0..4 {
                                            observed[i] = rd(aperture + (i * 4) as u64);
                                            wr(aperture + (i * 4) as u64, saved[i]);
                                        }
                                        core::sync::atomic::fence(
                                            core::sync::atomic::Ordering::SeqCst);
                                        let mut restored = [0u32; 4];
                                        for i in 0..4 {
                                            restored[i] = rd(aperture + (i * 4) as u64);
                                        }
                                        wr(window_reg, window_saved);
                                        let roundtrip_ok = observed == pattern;
                                        let restore_ok = restored == saved;
                                        log.push(format!(
                                            "  M2 VRAM scratch: FB[{}] phys=0x{:x} size=0x1000 pramin=+0x{:x} window=0x{:08x}->0x{:08x}",
                                            region, phys, aperture - bar0,
                                            window_saved, window_live));
                                        log.push(format!(
                                            "  M2 VRAM roundtrip: wrote=[{:08x},{:08x},{:08x},{:08x}] read=[{:08x},{:08x},{:08x},{:08x}] match={} restored={}",
                                            pattern[0], pattern[1], pattern[2], pattern[3],
                                            observed[0], observed[1], observed[2], observed[3],
                                            roundtrip_ok, restore_ok));
                                        if roundtrip_ok && restore_ok {
                                            log.push(String::from(
                                                "  *** M2 VRAM SCRATCH SUCCESS: allocated page is CPU-readable/writable ***"));
                                        }

                                        // Build the minimal GP100/TU102 4 KiB
                                        // page-table chain for BAR1 VA 0.  GSP
                                        // owns the PD3 root returned in
                                        // bar1PdeBase; only attach our chain if
                                        // its entry zero is unused, and detach
                                        // it before leaving the probe.
                                        let region_end = static_fb_regions[region][1];
                                        let pd3 = static_bar_pde[0];
                                        let spt = phys + 0x1000;
                                        let pd0 = phys + 0x2000;
                                        let pd1 = phys + 0x3000;
                                        let pd2 = phys + 0x4000;
                                        let tables_fit = pd2.saturating_add(0x1000)
                                            <= region_end.saturating_add(1);
                                        let fb_window = |addr: u64| {
                                            wr(window_reg,
                                                ((addr & !0x000f_ffff) >> 16) as u32);
                                            bar0 + 0x700000 + (addr & 0x000f_ffff)
                                        };
                                        let fb_read64 = |addr: u64| -> u64 {
                                            let ap = fb_window(addr);
                                            rd(ap) as u64 | ((rd(ap + 4) as u64) << 32)
                                        };
                                        let fb_write64 = |addr: u64, value: u64| {
                                            let ap = fb_window(addr);
                                            wr(ap, value as u32);
                                            wr(ap + 4, (value >> 32) as u32);
                                        };
                                        let root_before = fb_read64(pd3);
                                        let bar1_ctl = rd(bar0 + 0xb80f40);
                                        let bar1_inst = ((bar1_ctl & 0x7fff_ffff) as u64) << 12;
                                        let inst_pdb_raw = fb_read64(bar1_inst + 0x200);
                                        let inst_limit = fb_read64(bar1_inst + 0x208);
                                        let bar1_limit_target = bar3
                                            .saturating_sub(bar1)
                                            .saturating_sub(1);
                                        let inst_21c_ap = fb_window(bar1_inst + 0x21c);
                                        let inst_21c = rd(inst_21c_ap);
                                        let inst_fault_mask = fb_read64(bar1_inst + 0x298);
                                        let inst_pdb_copy0 = fb_read64(bar1_inst + 0x2a0);
                                        let inst_pdb_copy0_hi = {
                                            let ap = fb_window(bar1_inst + 0x2a8);
                                            rd(ap)
                                        };
                                        log.push(format!(
                                            "  M2 BAR1 root: ctl=0x{:08x} pdb=0x{:x} entry0=0x{:016x} tables_fit={}",
                                            bar1_ctl, pd3,
                                            root_before, tables_fit));
                                        log.push(format!(
                                            "  M2 BAR1 inst: phys=0x{:x} pdb_raw=0x{:016x} pdb=0x{:x} flags=0x{:03x} limit=0x{:x} va_size=0x{:x} target_limit=0x{:x}",
                                            bar1_inst, inst_pdb_raw,
                                            inst_pdb_raw & !0xfff,
                                            inst_pdb_raw & 0xfff,
                                            inst_limit, inst_limit.saturating_add(1),
                                            bar1_limit_target));
                                        log.push(format!(
                                            "  M2 BAR1 inst GV100: word21c=0x{:08x} fault_mask=0x{:016x} pdb_copy0=0x{:016x} copy_aux=0x{:08x}",
                                            inst_21c, inst_fault_mask,
                                            inst_pdb_copy0, inst_pdb_copy0_hi));
                                        if tables_fit && root_before == 0 {
                                            // Allocate/clear four page-table pages
                                            // bottom-up.  Unreferenced table pages
                                            // are harmless while being prepared.
                                            for table in [spt, pd0, pd1, pd2] {
                                                let ap = fb_window(table);
                                                for word in 0..0x400usize {
                                                    wr(ap + (word * 4) as u64, 0);
                                                }
                                            }

                                            // VRAM PTE: VALID plus physical addr>>4.
                                            fb_write64(spt, (phys >> 4) | 1);
                                            // VRAM PDE: target=VRAM (1<<1) plus
                                            // child address>>4.  PD0 has paired
                                            // small/large entries; use small only.
                                            // PD0 slot 0 is the 64 KiB LPT;
                                            // slot 1 is the 4 KiB SPT.
                                            fb_write64(pd0, 0);
                                            fb_write64(pd0 + 8, (spt >> 4) | 2);
                                            fb_write64(pd1, (pd0 >> 4) | 2);
                                            fb_write64(pd2, (pd1 >> 4) | 2);

                                            // Put a known value in the mapped page,
                                            // then publish PD3 last.
                                            let data_ap = fb_window(phys);
                                            for i in 0..4 {
                                                wr(data_ap + (i * 4) as u64, pattern[i]);
                                            }
                                            core::sync::atomic::fence(
                                                core::sync::atomic::Ordering::SeqCst);
                                            fb_write64(pd3, (pd2 >> 4) | 2);
                                            core::sync::atomic::fence(
                                                core::sync::atomic::Ordering::SeqCst);

                                            // TU102 BAR VMM invalidate: PAGE_ALL |
                                            // ALL_PDB | HUB_ONLY.
                                            wr(bar0 + 0xb830a0, (pd3 >> 8) as u32);
                                            wr(bar0 + 0xb830a4, (pd3 >> 40) as u32);
                                            wr(bar0 + 0xb830b0, 0x8000_0007);
                                            let mut flush_live = 0u32;
                                            let mut flush_done = false;
                                            for _ in 0..20000 {
                                                flush_live = rd(bar0 + 0xb830b0);
                                                if flush_live & 0x8000_0000 == 0 {
                                                    flush_done = true;
                                                    break;
                                                }
                                                boot::stall(10);
                                            }
                                            for _ in 0..20000 {
                                                if rd(bar0 + 0xb80f50) & 3 == 0 {
                                                    break;
                                                }
                                                boot::stall(10);
                                            }

                                            let mut bar1_seen = [0u32; 4];
                                            for i in 0..4 {
                                                bar1_seen[i] = rd(bar1 + (i * 4) as u64);
                                            }
                                            let bar1_write = [
                                                0x4241_5231,
                                                0x5aa5_a55a,
                                                !(phys as u32),
                                                0x2468_ace0,
                                            ];
                                            for i in 0..4 {
                                                wr(bar1 + (i * 4) as u64, bar1_write[i]);
                                            }
                                            core::sync::atomic::fence(
                                                core::sync::atomic::Ordering::SeqCst);
                                            let data_ap = fb_window(phys);
                                            let mut pramin_seen = [0u32; 4];
                                            for i in 0..4 {
                                                pramin_seen[i] = rd(data_ap + (i * 4) as u64);
                                                wr(data_ap + (i * 4) as u64, saved[i]);
                                            }

                                            // Detach VA0 and flush once more before
                                            // restoring the caller's PRAMIN window.
                                            fb_write64(pd3, root_before);
                                            core::sync::atomic::fence(
                                                core::sync::atomic::Ordering::SeqCst);
                                            wr(bar0 + 0xb830a0, (pd3 >> 8) as u32);
                                            wr(bar0 + 0xb830a4, (pd3 >> 40) as u32);
                                            wr(bar0 + 0xb830b0, 0x8000_0007);
                                            for _ in 0..20000 {
                                                if rd(bar0 + 0xb830b0) & 0x8000_0000 == 0 {
                                                    break;
                                                }
                                                boot::stall(10);
                                            }
                                            wr(window_reg, window_saved);

                                            let read_ok = bar1_seen == pattern;
                                            let write_ok = pramin_seen == bar1_write;
                                            log.push(format!(
                                                "  M2 BAR1 map: va=0x0 phys=0x{:x} tables=[0x{:x},0x{:x},0x{:x},0x{:x}] flush_done={} flush=0x{:08x}",
                                                phys, spt, pd0, pd1, pd2,
                                                flush_done, flush_live));
                                            log.push(format!(
                                                "  M2 BAR1 read: expected=[{:08x},{:08x},{:08x},{:08x}] observed=[{:08x},{:08x},{:08x},{:08x}] match={}",
                                                pattern[0], pattern[1], pattern[2], pattern[3],
                                                bar1_seen[0], bar1_seen[1], bar1_seen[2], bar1_seen[3],
                                                read_ok));
                                            log.push(format!(
                                                "  M2 BAR1 write: expected=[{:08x},{:08x},{:08x},{:08x}] pramin=[{:08x},{:08x},{:08x},{:08x}] match={} root_restored={}",
                                                bar1_write[0], bar1_write[1], bar1_write[2], bar1_write[3],
                                                pramin_seen[0], pramin_seen[1], pramin_seen[2], pramin_seen[3],
                                                write_ok, fb_read64(pd3) == root_before));
                                            wr(window_reg, window_saved);
                                            if flush_done && read_ok && write_ok {
                                                log.push(String::from(
                                                    "  *** M2 BAR1 SUCCESS: bidirectional CPU mapping of allocated VRAM ***"));
                                            }
                                        } else if tables_fit {
                                            // GSP normally prebuilds the upper
                                            // BAR1 directory chain.  Reuse it and
                                            // add only a missing small-page table
                                            // (or one free PTE in the existing
                                            // table), preserving every entry we
                                            // touch for teardown.
                                            let pde_child = |entry: u64| -> u64 {
                                                // RM may populate PDE flags in
                                                // bits 4..7 (observed 0xf2 on
                                                // the small-page PDE).  A child
                                                // table is 4 KiB aligned, so all
                                                // low eight encoded bits must be
                                                // removed before undoing addr>>4.
                                                (entry & 0x0000_ffff_ffff_ff00) << 4
                                            };
                                            let pd2_live = pde_child(root_before);
                                            let pd2e = fb_read64(pd2_live);
                                            let pd1_live = if pd2e != 0 {
                                                pde_child(pd2e)
                                            } else {
                                                0
                                            };
                                            let pd1e = if pd1_live != 0 {
                                                fb_read64(pd1_live)
                                            } else {
                                                0
                                            };
                                            let pd0_live = if pd1e != 0 {
                                                pde_child(pd1e)
                                            } else {
                                                0
                                            };
                                            let pd0_large = if pd0_live != 0 {
                                                fb_read64(pd0_live)
                                            } else {
                                                0
                                            };
                                            let pd0_small = if pd0_live != 0 {
                                                fb_read64(pd0_live + 8)
                                            } else {
                                                0
                                            };
                                            log.push(format!(
                                                "  M2 BAR1 chain: pd2=0x{:x}/0x{:016x} pd1=0x{:x}/0x{:016x} pd0=0x{:x} small=0x{:016x} large=0x{:016x}",
                                                pd2_live, pd2e, pd1_live, pd1e,
                                                pd0_live, pd0_small, pd0_large));

                                            if pd2e != 0 && pd1e != 0 {
                                                let attach_spt = pd0_small == 0;
                                                let spt_live = if attach_spt {
                                                    spt
                                                } else {
                                                    pde_child(pd0_small)
                                                };
                                                if attach_spt {
                                                    let ap = fb_window(spt_live);
                                                    for word in 0..0x400usize {
                                                        wr(ap + (word * 4) as u64, 0);
                                                    }
                                                }
                                                let mut free_pte = None;
                                                for i in 0..512usize {
                                                    if fb_read64(spt_live + (i * 8) as u64) == 0 {
                                                        free_pte = Some(i);
                                                        break;
                                                    }
                                                }
                                                if let Some(ptei) = free_pte {
                                                    let va = (ptei as u64) << 12;
                                                    let pte_addr = spt_live + (ptei * 8) as u64;
                                                    let pte_before = fb_read64(pte_addr);
                                                    // A nonzero invalid LPTE (the
                                                    // firmware uses VALID_FALSE |
                                                    // PRIV) explicitly blocks the
                                                    // corresponding SPTEs.  Clear
                                                    // the covering 64 KiB entry
                                                    // while our 4 KiB mapping owns
                                                    // this range, then restore it.
                                                    let lpt_live = if pd0_large != 0 {
                                                        (pd0_large & 0x0000_ffff_ffff_fff0) << 4
                                                    } else {
                                                        0
                                                    };
                                                    let lpti = ptei >> 4;
                                                    let lpte_addr = lpt_live + (lpti * 8) as u64;
                                                    let lpte_before = if lpt_live != 0 {
                                                        fb_read64(lpte_addr)
                                                    } else {
                                                        0
                                                    };
                                                    let mut subctx_saved = [[0u32; 3]; 64];
                                                    for i in 0..64usize {
                                                        let ap = fb_window(
                                                            bar1_inst + 0x2a0 + (i * 0x10) as u64);
                                                        subctx_saved[i][0] = rd(ap);
                                                        subctx_saved[i][1] = rd(ap + 4);
                                                        subctx_saved[i][2] = rd(ap + 8);
                                                    }
                                                    // BAR1's GSP-created instance
                                                    // block has a zero VA limit.
                                                    // Complete its GV100 subcontext
                                                    // table exactly as Nouveau's
                                                    // gv100_vmm_join, then rebind the
                                                    // unchanged instance pointer.
                                                    fb_write64(bar1_inst + 0x208,
                                                        bar1_limit_target);
                                                    {
                                                        let ap = fb_window(bar1_inst + 0x21c);
                                                        wr(ap, 0);
                                                    }
                                                    for i in 0..64usize {
                                                        let ap = fb_window(
                                                            bar1_inst + 0x2a0 + (i * 0x10) as u64);
                                                        if i == 0 {
                                                            wr(ap, inst_pdb_raw as u32);
                                                            wr(ap + 4, (inst_pdb_raw >> 32) as u32);
                                                        } else {
                                                            wr(ap, 1);
                                                            wr(ap + 4, 1);
                                                        }
                                                        wr(ap + 8, 0);
                                                    }
                                                    fb_write64(bar1_inst + 0x298, 1);
                                                    core::sync::atomic::fence(
                                                        core::sync::atomic::Ordering::SeqCst);
                                                    wr(bar0 + 0xb80f40, bar1_ctl);
                                                    for _ in 0..20000 {
                                                        if rd(bar0 + 0xb80f50) & 3 == 0 {
                                                            break;
                                                        }
                                                        boot::stall(10);
                                                    }
                                                    let data_ap = fb_window(phys);
                                                    for i in 0..4 {
                                                        wr(data_ap + (i * 4) as u64, pattern[i]);
                                                    }
                                                    core::sync::atomic::fence(
                                                        core::sync::atomic::Ordering::SeqCst);
                                                    if lpt_live != 0 {
                                                        fb_write64(lpte_addr, 0);
                                                    }
                                                    fb_write64(pte_addr, (phys >> 4) | 1);
                                                    if attach_spt {
                                                        core::sync::atomic::fence(
                                                            core::sync::atomic::Ordering::SeqCst);
                                                        fb_write64(pd0_live + 8, (spt_live >> 4) | 2);
                                                    }
                                                    core::sync::atomic::fence(
                                                        core::sync::atomic::Ordering::SeqCst);

                                                    wr(bar0 + 0xb830a0, (pd3 >> 8) as u32);
                                                    wr(bar0 + 0xb830a4, (pd3 >> 40) as u32);
                                                    wr(bar0 + 0xb830b0, 0x8000_0007);
                                                    let mut flush_live = 0u32;
                                                    let mut flush_done = false;
                                                    for _ in 0..20000 {
                                                        flush_live = rd(bar0 + 0xb830b0);
                                                        if flush_live & 0x8000_0000 == 0 {
                                                            flush_done = true;
                                                            break;
                                                        }
                                                        boot::stall(10);
                                                    }
                                                    for _ in 0..20000 {
                                                        if rd(bar0 + 0xb80f50) & 3 == 0 {
                                                            break;
                                                        }
                                                        boot::stall(10);
                                                    }

                                                    let mut bar1_seen = [0u32; 4];
                                                    for i in 0..4 {
                                                        bar1_seen[i] = rd(bar1 + va + (i * 4) as u64);
                                                    }
                                                    let bar1_write = [
                                                        0x4241_5231,
                                                        0x5aa5_a55a,
                                                        !(phys as u32),
                                                        0x2468_ace0,
                                                    ];
                                                    for i in 0..4 {
                                                        wr(bar1 + va + (i * 4) as u64, bar1_write[i]);
                                                    }
                                                    core::sync::atomic::fence(
                                                        core::sync::atomic::Ordering::SeqCst);
                                                    let data_ap = fb_window(phys);
                                                    let mut pramin_seen = [0u32; 4];
                                                    for i in 0..4 {
                                                        pramin_seen[i] = rd(data_ap + (i * 4) as u64);
                                                        wr(data_ap + (i * 4) as u64, saved[i]);
                                                    }

                                                    // Remove the most-parent entry
                                                    // first, then restore the leaf.
                                                    if attach_spt {
                                                        fb_write64(pd0_live + 8, pd0_small);
                                                    }
                                                    fb_write64(pte_addr, pte_before);
                                                    if lpt_live != 0 {
                                                        fb_write64(lpte_addr, lpte_before);
                                                    }
                                                    // Disable our subcontext mask
                                                    // before restoring its table.
                                                    fb_write64(bar1_inst + 0x298,
                                                        inst_fault_mask);
                                                    for i in 0..64usize {
                                                        let ap = fb_window(
                                                            bar1_inst + 0x2a0 + (i * 0x10) as u64);
                                                        wr(ap, subctx_saved[i][0]);
                                                        wr(ap + 4, subctx_saved[i][1]);
                                                        wr(ap + 8, subctx_saved[i][2]);
                                                    }
                                                    {
                                                        let ap = fb_window(bar1_inst + 0x21c);
                                                        wr(ap, inst_21c);
                                                    }
                                                    fb_write64(bar1_inst + 0x208, inst_limit);
                                                    core::sync::atomic::fence(
                                                        core::sync::atomic::Ordering::SeqCst);
                                                    wr(bar0 + 0xb80f40, bar1_ctl);
                                                    for _ in 0..20000 {
                                                        if rd(bar0 + 0xb80f50) & 3 == 0 {
                                                            break;
                                                        }
                                                        boot::stall(10);
                                                    }
                                                    wr(bar0 + 0xb830a0, (pd3 >> 8) as u32);
                                                    wr(bar0 + 0xb830a4, (pd3 >> 40) as u32);
                                                    wr(bar0 + 0xb830b0, 0x8000_0007);
                                                    for _ in 0..20000 {
                                                        if rd(bar0 + 0xb830b0) & 0x8000_0000 == 0 {
                                                            break;
                                                        }
                                                        boot::stall(10);
                                                    }
                                                    let pte_restored = fb_read64(pte_addr) == pte_before;
                                                    let parent_restored = fb_read64(pd0_live + 8) == pd0_small;
                                                    let lpte_restored = lpt_live == 0
                                                        || fb_read64(lpte_addr) == lpte_before;
                                                    let limit_restored =
                                                        fb_read64(bar1_inst + 0x208) == inst_limit;
                                                    let mut gv100_restored =
                                                        fb_read64(bar1_inst + 0x298) == inst_fault_mask;
                                                    for i in 0..64usize {
                                                        let ap = fb_window(
                                                            bar1_inst + 0x2a0 + (i * 0x10) as u64);
                                                        gv100_restored &= rd(ap) == subctx_saved[i][0]
                                                            && rd(ap + 4) == subctx_saved[i][1]
                                                            && rd(ap + 8) == subctx_saved[i][2];
                                                    }
                                                    wr(window_reg, window_saved);

                                                    let read_ok = bar1_seen == pattern;
                                                    let write_ok = pramin_seen == bar1_write;
                                                    log.push(format!(
                                                        "  M2 BAR1 leaf: spt=0x{:x} ptei={} va=0x{:x} attach_spt={} pte=0x{:016x} lpt=0x{:x} lpti={} lpte_before=0x{:016x} flush_done={} flush=0x{:08x}",
                                                        spt_live, ptei, va, attach_spt,
                                                        (phys >> 4) | 1, lpt_live, lpti,
                                                        lpte_before, flush_done, flush_live));
                                                    log.push(format!(
                                                        "  M2 BAR1 read: expected=[{:08x},{:08x},{:08x},{:08x}] observed=[{:08x},{:08x},{:08x},{:08x}] match={}",
                                                        pattern[0], pattern[1], pattern[2], pattern[3],
                                                        bar1_seen[0], bar1_seen[1], bar1_seen[2], bar1_seen[3],
                                                        read_ok));
                                                    log.push(format!(
                                                        "  M2 BAR1 write: expected=[{:08x},{:08x},{:08x},{:08x}] pramin=[{:08x},{:08x},{:08x},{:08x}] match={} pte_restored={} parent_restored={} lpte_restored={} limit_restored={} gv100_restored={}",
                                                        bar1_write[0], bar1_write[1], bar1_write[2], bar1_write[3],
                                                        pramin_seen[0], pramin_seen[1], pramin_seen[2], pramin_seen[3],
                                                        write_ok, pte_restored, parent_restored, lpte_restored,
                                                        limit_restored, gv100_restored));
                                                    if flush_done && read_ok && write_ok
                                                        && pte_restored && parent_restored
                                                        && lpte_restored && limit_restored
                                                        && gv100_restored
                                                    {
                                                        log.push(String::from(
                                                            "  *** M2 BAR1 SUCCESS: bidirectional CPU mapping of allocated VRAM ***"));
                                                    }
                                                } else {
                                                    wr(window_reg, window_saved);
                                                    log.push(String::from(
                                                        "  M2 BAR1 mapping skipped: firmware small-page table has no free PTE"));
                                                }
                                            } else {
                                                wr(window_reg, window_saved);
                                                log.push(String::from(
                                                    "  M2 BAR1 mapping skipped: firmware upper directory chain is incomplete"));
                                            }
                                        } else {
                                            wr(window_reg, window_saved);
                                            log.push(String::from(
                                                "  M2 BAR1 mapping skipped: table allocation does not fit"));
                                        }
                                    } else {
                                        log.push(String::from(
                                            "  M2 VRAM scratch FAILED: no unreserved, unprotected 4K FB page"));
                                    }
                                }

                                // --- M3: establish a user RM client namespace ---
                                //
                                // Nouveau's first object allocation is an NV01_ROOT
                                // client (r535_gsp_client_ctor).  The root object is
                                // its own parent, so hClient/hParent/hObject are the
                                // same generated handle.  NV0000_ALLOC_PARAMETERS is
                                // 120 bytes on this ABI: handle + pid + 100-byte name,
                                // 4 bytes of alignment, then an 8-byte OS-pid pointer.
                                if replied && got_result == 0 {
                                    const RPC_GSP_RM_ALLOC: u32 = 103;
                                    const CLIENT_HANDLE: u32 = 0xc1d0_0000;
                                    const RM_ALLOC_HEADER_SIZE: usize = 32;
                                    const NV0000_ALLOC_SIZE: usize = 120;
                                    const RM_ALLOC_SIZE: usize =
                                        RM_ALLOC_HEADER_SIZE + NV0000_ALLOC_SIZE;
                                    let mut alloc = [0u8; RM_ALLOC_SIZE];
                                    alloc[0x00..0x04].copy_from_slice(&CLIENT_HANDLE.to_le_bytes());
                                    alloc[0x04..0x08].copy_from_slice(&CLIENT_HANDLE.to_le_bytes());
                                    alloc[0x08..0x0c].copy_from_slice(&CLIENT_HANDLE.to_le_bytes());
                                    alloc[0x0c..0x10].copy_from_slice(&0u32.to_le_bytes()); // NV01_ROOT
                                    alloc[0x10..0x14].copy_from_slice(&0u32.to_le_bytes()); // status
                                    alloc[0x14..0x18]
                                        .copy_from_slice(&(NV0000_ALLOC_SIZE as u32).to_le_bytes());
                                    // flags and reserved remain zero.  The client
                                    // parameters begin at +0x20.
                                    alloc[0x20..0x24].copy_from_slice(&CLIENT_HANDLE.to_le_bytes());
                                    alloc[0x24..0x28].copy_from_slice(&u32::MAX.to_le_bytes());

                                    let alloc_seq = 3u32;
                                    let alloc_pushed = unsafe {
                                        gsp::rpc_send_argc(
                                            &handshake, RPC_GSP_RM_ALLOC,
                                            &alloc, RM_ALLOC_SIZE, alloc_seq)
                                    };
                                    match alloc_pushed {
                                        Some(alloc_csum) => {
                                            let alloc_cmd_w = unsafe {
                                                core::ptr::read_volatile(
                                                    cmdq.add(0x10) as *const u32) };
                                            let alloc_cmd_r_before = unsafe {
                                                core::ptr::read_volatile(
                                                    msgq.add(msg_rxoff as usize) as *const u32) };
                                            let alloc_msg_w_before = unsafe {
                                                core::ptr::read_volatile(
                                                    msgq.add(0x10) as *const u32) };
                                            gsp::rpc_doorbell(gsp_falcon, &wr);
                                            let alloc_doorbell = rd(gsp_falcon + 0xc00);
                                            log.push(format!(
                                                "  M3 CLIENT RPC sent: fn={} seq={} handle=0x{:08x} class=0x00000000 argc={} checksum=0x{:08x} doorbell_readback=0x{:08x}",
                                                RPC_GSP_RM_ALLOC, alloc_seq, CLIENT_HANDLE,
                                                RM_ALLOC_SIZE, alloc_csum, alloc_doorbell));

                                            let mut alloc_matched = false;
                                            let mut alloc_reply_seq = u32::MAX;
                                            let mut alloc_reply_fn = 0u32;
                                            let mut alloc_rpc_result = u32::MAX;
                                            let mut alloc_status = u32::MAX;
                                            let mut alloc_reply_len = 0u32;
                                            for _ in 0..20000 {
                                                let wptr = unsafe {
                                                    core::ptr::read_volatile(
                                                        msgq.add(0x10) as *const u32) };
                                                let rptr = unsafe {
                                                    core::ptr::read_volatile(
                                                        cmdq.add(0x20) as *const u32) };
                                                if wptr != rptr {
                                                    let elem = unsafe {
                                                        msgq.add(tx_entryoff as usize
                                                            + rptr as usize * 0x1000) };
                                                    alloc_reply_seq = unsafe {
                                                        core::ptr::read_volatile(
                                                            elem.add(0x24) as *const u32) };
                                                    alloc_reply_len = unsafe {
                                                        core::ptr::read_volatile(
                                                            elem.add(0x38) as *const u32) };
                                                    alloc_reply_fn = unsafe {
                                                        core::ptr::read_volatile(
                                                            elem.add(0x3c) as *const u32) };
                                                    alloc_rpc_result = unsafe {
                                                        core::ptr::read_volatile(
                                                            elem.add(0x40) as *const u32) };
                                                    if alloc_reply_fn == RPC_GSP_RM_ALLOC
                                                        && alloc_reply_len >= 0x20 + RM_ALLOC_HEADER_SIZE as u32
                                                    {
                                                        alloc_status = unsafe {
                                                            core::ptr::read_volatile(
                                                                elem.add(0x60) as *const u32) };
                                                    }
                                                    let ec = unsafe {
                                                        core::ptr::read_volatile(
                                                            elem.add(0x28) as *const u32) };
                                                    let step = if ec == 0 { 1 } else { ec };
                                                    let mut next = rptr + step;
                                                    if next >= tx_count { next -= tx_count; }
                                                    core::sync::atomic::fence(
                                                        core::sync::atomic::Ordering::SeqCst);
                                                    unsafe {
                                                        core::ptr::write_volatile(
                                                            cmdq.add(0x20) as *mut u32, next);
                                                    }
                                                    // The message envelope sequence is
                                                    // the status ring's monotonic TX
                                                    // sequence, not an echo of cmdq.seq.
                                                    // The synchronous waiter therefore
                                                    // identifies this reply by function,
                                                    // as the proven fn65 path above does.
                                                    if alloc_reply_fn == RPC_GSP_RM_ALLOC {
                                                        alloc_matched = true;
                                                        break;
                                                    }
                                                }
                                                boot::stall(100);
                                            }
                                            let alloc_cmd_r_final = unsafe {
                                                core::ptr::read_volatile(
                                                    msgq.add(msg_rxoff as usize) as *const u32) };
                                            let alloc_msg_w_final = unsafe {
                                                core::ptr::read_volatile(
                                                    msgq.add(0x10) as *const u32) };
                                            log.push(format!(
                                                "  M3 CLIENT rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                alloc_cmd_w, alloc_cmd_r_before,
                                                alloc_cmd_r_final, alloc_msg_w_before,
                                                alloc_msg_w_final));
                                            log.push(format!(
                                                "  *** M3 CLIENT reply: matched={} seq={} fn={} rpc_result=0x{:08x} alloc_status=0x{:08x} len=0x{:x} ***",
                                                alloc_matched, alloc_reply_seq, alloc_reply_fn,
                                                alloc_rpc_result, alloc_status, alloc_reply_len));
                                            if alloc_matched && alloc_rpc_result == 0
                                                && alloc_status == 0
                                            {
                                                log.push(format!(
                                                    "  *** M3 CLIENT SUCCESS: NV01_ROOT allocated handle=0x{:08x} ***",
                                                    CLIENT_HANDLE));

                                                // Nouveau's next object is NV01_DEVICE_0
                                                // under the new root client.  r535 only
                                                // supplies hClientShare; the remaining
                                                // NV0080 allocation fields stay zero.
                                                const DEVICE_HANDLE: u32 = 0xde1d_0000;
                                                const NV01_DEVICE_0: u32 = 0x80;
                                                const NV0080_ALLOC_SIZE: usize = 56;
                                                const DEVICE_ALLOC_SIZE: usize =
                                                    RM_ALLOC_HEADER_SIZE + NV0080_ALLOC_SIZE;
                                                let mut device_alloc = [0u8; DEVICE_ALLOC_SIZE];
                                                device_alloc[0x00..0x04]
                                                    .copy_from_slice(&CLIENT_HANDLE.to_le_bytes());
                                                device_alloc[0x04..0x08]
                                                    .copy_from_slice(&CLIENT_HANDLE.to_le_bytes());
                                                device_alloc[0x08..0x0c]
                                                    .copy_from_slice(&DEVICE_HANDLE.to_le_bytes());
                                                device_alloc[0x0c..0x10]
                                                    .copy_from_slice(&NV01_DEVICE_0.to_le_bytes());
                                                device_alloc[0x14..0x18].copy_from_slice(
                                                    &(NV0080_ALLOC_SIZE as u32).to_le_bytes());
                                                // NV0080_ALLOC_PARAMETERS::hClientShare
                                                // is the second dword in params.
                                                device_alloc[0x24..0x28]
                                                    .copy_from_slice(&CLIENT_HANDLE.to_le_bytes());

                                                let device_seq = 4u32;
                                                let device_pushed = unsafe {
                                                    gsp::rpc_send_argc(
                                                        &handshake, RPC_GSP_RM_ALLOC,
                                                        &device_alloc, DEVICE_ALLOC_SIZE,
                                                        device_seq)
                                                };
                                                match device_pushed {
                                                    Some(device_csum) => {
                                                        let device_cmd_w = unsafe {
                                                            core::ptr::read_volatile(
                                                                cmdq.add(0x10) as *const u32) };
                                                        let device_cmd_r_before = unsafe {
                                                            core::ptr::read_volatile(
                                                                msgq.add(msg_rxoff as usize)
                                                                    as *const u32) };
                                                        let device_msg_w_before = unsafe {
                                                            core::ptr::read_volatile(
                                                                msgq.add(0x10) as *const u32) };
                                                        gsp::rpc_doorbell(gsp_falcon, &wr);
                                                        let device_doorbell =
                                                            rd(gsp_falcon + 0xc00);
                                                        log.push(format!(
                                                            "  M3 DEVICE RPC sent: fn={} seq={} client=0x{:08x} parent=0x{:08x} object=0x{:08x} class=0x{:08x} argc={} checksum=0x{:08x} doorbell_readback=0x{:08x}",
                                                            RPC_GSP_RM_ALLOC, device_seq,
                                                            CLIENT_HANDLE, CLIENT_HANDLE,
                                                            DEVICE_HANDLE, NV01_DEVICE_0,
                                                            DEVICE_ALLOC_SIZE, device_csum,
                                                            device_doorbell));

                                                        let mut device_matched = false;
                                                        let mut device_reply_seq = u32::MAX;
                                                        let mut device_reply_fn = 0u32;
                                                        let mut device_rpc_result = u32::MAX;
                                                        let mut device_status = u32::MAX;
                                                        let mut device_reply_len = 0u32;
                                                        for _ in 0..20000 {
                                                            let wptr = unsafe {
                                                                core::ptr::read_volatile(
                                                                    msgq.add(0x10) as *const u32) };
                                                            let rptr = unsafe {
                                                                core::ptr::read_volatile(
                                                                    cmdq.add(0x20) as *const u32) };
                                                            if wptr != rptr {
                                                                let elem = unsafe {
                                                                    msgq.add(tx_entryoff as usize
                                                                        + rptr as usize * 0x1000) };
                                                                device_reply_seq = unsafe {
                                                                    core::ptr::read_volatile(
                                                                        elem.add(0x24) as *const u32) };
                                                                device_reply_len = unsafe {
                                                                    core::ptr::read_volatile(
                                                                        elem.add(0x38) as *const u32) };
                                                                device_reply_fn = unsafe {
                                                                    core::ptr::read_volatile(
                                                                        elem.add(0x3c) as *const u32) };
                                                                device_rpc_result = unsafe {
                                                                    core::ptr::read_volatile(
                                                                        elem.add(0x40) as *const u32) };
                                                                if device_reply_fn == RPC_GSP_RM_ALLOC
                                                                    && device_reply_len >= 0x20
                                                                        + RM_ALLOC_HEADER_SIZE as u32
                                                                {
                                                                    device_status = unsafe {
                                                                        core::ptr::read_volatile(
                                                                            elem.add(0x60)
                                                                                as *const u32) };
                                                                }
                                                                let ec = unsafe {
                                                                    core::ptr::read_volatile(
                                                                        elem.add(0x28) as *const u32) };
                                                                let step = if ec == 0 { 1 } else { ec };
                                                                let mut next = rptr + step;
                                                                if next >= tx_count {
                                                                    next -= tx_count;
                                                                }
                                                                core::sync::atomic::fence(
                                                                    core::sync::atomic::Ordering::SeqCst);
                                                                unsafe {
                                                                    core::ptr::write_volatile(
                                                                        cmdq.add(0x20) as *mut u32,
                                                                        next);
                                                                }
                                                                if device_reply_fn
                                                                    == RPC_GSP_RM_ALLOC
                                                                {
                                                                    device_matched = true;
                                                                    break;
                                                                }
                                                            }
                                                            boot::stall(100);
                                                        }
                                                        let device_cmd_r_final = unsafe {
                                                            core::ptr::read_volatile(
                                                                msgq.add(msg_rxoff as usize)
                                                                    as *const u32) };
                                                        let device_msg_w_final = unsafe {
                                                            core::ptr::read_volatile(
                                                                msgq.add(0x10) as *const u32) };
                                                        log.push(format!(
                                                            "  M3 DEVICE rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                            device_cmd_w, device_cmd_r_before,
                                                            device_cmd_r_final,
                                                            device_msg_w_before,
                                                            device_msg_w_final));
                                                        log.push(format!(
                                                            "  *** M3 DEVICE reply: matched={} seq={} fn={} rpc_result=0x{:08x} alloc_status=0x{:08x} len=0x{:x} ***",
                                                            device_matched, device_reply_seq,
                                                            device_reply_fn, device_rpc_result,
                                                            device_status, device_reply_len));
                                                        if device_matched
                                                            && device_rpc_result == 0
                                                            && device_status == 0
                                                        {
                                                            log.push(format!(
                                                                "  *** M3 DEVICE SUCCESS: NV01_DEVICE_0 allocated handle=0x{:08x} parent=0x{:08x} ***",
                                                                DEVICE_HANDLE, CLIENT_HANDLE));

                                                            // r535_gsp_subdevice_ctor:
                                                            // allocate NV20_SUBDEVICE_0
                                                            // under NV01_DEVICE_0.  Its
                                                            // sole parameter, subDeviceId,
                                                            // is zero for the first GPU.
                                                            const SUBDEVICE_HANDLE: u32 =
                                                                0x5d1d_0000;
                                                            const NV20_SUBDEVICE_0: u32 =
                                                                0x2080;
                                                            const NV2080_ALLOC_SIZE: usize = 4;
                                                            const SUBDEVICE_ALLOC_SIZE: usize =
                                                                RM_ALLOC_HEADER_SIZE
                                                                    + NV2080_ALLOC_SIZE;
                                                            let mut subdevice_alloc =
                                                                [0u8; SUBDEVICE_ALLOC_SIZE];
                                                            subdevice_alloc[0x00..0x04]
                                                                .copy_from_slice(
                                                                    &CLIENT_HANDLE.to_le_bytes());
                                                            subdevice_alloc[0x04..0x08]
                                                                .copy_from_slice(
                                                                    &DEVICE_HANDLE.to_le_bytes());
                                                            subdevice_alloc[0x08..0x0c]
                                                                .copy_from_slice(
                                                                    &SUBDEVICE_HANDLE.to_le_bytes());
                                                            subdevice_alloc[0x0c..0x10]
                                                                .copy_from_slice(
                                                                    &NV20_SUBDEVICE_0.to_le_bytes());
                                                            subdevice_alloc[0x14..0x18]
                                                                .copy_from_slice(
                                                                    &(NV2080_ALLOC_SIZE as u32)
                                                                        .to_le_bytes());

                                                            let subdevice_seq = 5u32;
                                                            let subdevice_pushed = unsafe {
                                                                gsp::rpc_send_argc(
                                                                    &handshake,
                                                                    RPC_GSP_RM_ALLOC,
                                                                    &subdevice_alloc,
                                                                    SUBDEVICE_ALLOC_SIZE,
                                                                    subdevice_seq)
                                                            };
                                                            match subdevice_pushed {
                                                                Some(subdevice_csum) => {
                                                                    let subdevice_cmd_w = unsafe {
                                                                        core::ptr::read_volatile(
                                                                            cmdq.add(0x10)
                                                                                as *const u32) };
                                                                    let subdevice_cmd_r_before = unsafe {
                                                                        core::ptr::read_volatile(
                                                                            msgq.add(
                                                                                msg_rxoff as usize)
                                                                                as *const u32) };
                                                                    let subdevice_msg_w_before = unsafe {
                                                                        core::ptr::read_volatile(
                                                                            msgq.add(0x10)
                                                                                as *const u32) };
                                                                    gsp::rpc_doorbell(
                                                                        gsp_falcon, &wr);
                                                                    let subdevice_doorbell =
                                                                        rd(gsp_falcon + 0xc00);
                                                                    log.push(format!(
                                                                        "  M3 SUBDEVICE RPC sent: fn={} seq={} client=0x{:08x} parent=0x{:08x} object=0x{:08x} class=0x{:08x} argc={} checksum=0x{:08x} doorbell_readback=0x{:08x}",
                                                                        RPC_GSP_RM_ALLOC,
                                                                        subdevice_seq,
                                                                        CLIENT_HANDLE,
                                                                        DEVICE_HANDLE,
                                                                        SUBDEVICE_HANDLE,
                                                                        NV20_SUBDEVICE_0,
                                                                        SUBDEVICE_ALLOC_SIZE,
                                                                        subdevice_csum,
                                                                        subdevice_doorbell));

                                                                    let mut subdevice_matched = false;
                                                                    let mut subdevice_reply_seq =
                                                                        u32::MAX;
                                                                    let mut subdevice_reply_fn = 0u32;
                                                                    let mut subdevice_rpc_result =
                                                                        u32::MAX;
                                                                    let mut subdevice_status =
                                                                        u32::MAX;
                                                                    let mut subdevice_reply_len = 0u32;
                                                                    for _ in 0..20000 {
                                                                        let wptr = unsafe {
                                                                            core::ptr::read_volatile(
                                                                                msgq.add(0x10)
                                                                                    as *const u32) };
                                                                        let rptr = unsafe {
                                                                            core::ptr::read_volatile(
                                                                                cmdq.add(0x20)
                                                                                    as *const u32) };
                                                                        if wptr != rptr {
                                                                            let elem = unsafe {
                                                                                msgq.add(
                                                                                    tx_entryoff as usize
                                                                                        + rptr as usize
                                                                                            * 0x1000) };
                                                                            subdevice_reply_seq = unsafe {
                                                                                core::ptr::read_volatile(
                                                                                    elem.add(0x24)
                                                                                        as *const u32) };
                                                                            subdevice_reply_len = unsafe {
                                                                                core::ptr::read_volatile(
                                                                                    elem.add(0x38)
                                                                                        as *const u32) };
                                                                            subdevice_reply_fn = unsafe {
                                                                                core::ptr::read_volatile(
                                                                                    elem.add(0x3c)
                                                                                        as *const u32) };
                                                                            subdevice_rpc_result = unsafe {
                                                                                core::ptr::read_volatile(
                                                                                    elem.add(0x40)
                                                                                        as *const u32) };
                                                                            if subdevice_reply_fn
                                                                                == RPC_GSP_RM_ALLOC
                                                                                && subdevice_reply_len
                                                                                    >= 0x20
                                                                                        + RM_ALLOC_HEADER_SIZE
                                                                                            as u32
                                                                            {
                                                                                subdevice_status = unsafe {
                                                                                    core::ptr::read_volatile(
                                                                                        elem.add(0x60)
                                                                                            as *const u32) };
                                                                            }
                                                                            let ec = unsafe {
                                                                                core::ptr::read_volatile(
                                                                                    elem.add(0x28)
                                                                                        as *const u32) };
                                                                            let step = if ec == 0 {
                                                                                1
                                                                            } else {
                                                                                ec
                                                                            };
                                                                            let mut next = rptr + step;
                                                                            if next >= tx_count {
                                                                                next -= tx_count;
                                                                            }
                                                                            core::sync::atomic::fence(
                                                                                core::sync::atomic::Ordering::SeqCst);
                                                                            unsafe {
                                                                                core::ptr::write_volatile(
                                                                                    cmdq.add(0x20)
                                                                                        as *mut u32,
                                                                                    next);
                                                                            }
                                                                            if subdevice_reply_fn
                                                                                == RPC_GSP_RM_ALLOC
                                                                            {
                                                                                subdevice_matched = true;
                                                                                break;
                                                                            }
                                                                        }
                                                                        boot::stall(100);
                                                                    }
                                                                    let subdevice_cmd_r_final = unsafe {
                                                                        core::ptr::read_volatile(
                                                                            msgq.add(
                                                                                msg_rxoff as usize)
                                                                                as *const u32) };
                                                                    let subdevice_msg_w_final = unsafe {
                                                                        core::ptr::read_volatile(
                                                                            msgq.add(0x10)
                                                                                as *const u32) };
                                                                    log.push(format!(
                                                                        "  M3 SUBDEVICE rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                                        subdevice_cmd_w,
                                                                        subdevice_cmd_r_before,
                                                                        subdevice_cmd_r_final,
                                                                        subdevice_msg_w_before,
                                                                        subdevice_msg_w_final));
                                                                    log.push(format!(
                                                                        "  *** M3 SUBDEVICE reply: matched={} seq={} fn={} rpc_result=0x{:08x} alloc_status=0x{:08x} len=0x{:x} ***",
                                                                        subdevice_matched,
                                                                        subdevice_reply_seq,
                                                                        subdevice_reply_fn,
                                                                        subdevice_rpc_result,
                                                                        subdevice_status,
                                                                        subdevice_reply_len));
                                                                    if subdevice_matched
                                                                        && subdevice_rpc_result == 0
                                                                        && subdevice_status == 0
                                                                    {
                                                                        log.push(format!(
                                                                            "  *** M3 SUBDEVICE SUCCESS: NV20_SUBDEVICE_0 allocated handle=0x{:08x} parent=0x{:08x} ***",
                                                                            SUBDEVICE_HANDLE,
                                                                            DEVICE_HANDLE));

                                                                        // r535_mmu_promote_vmm:
                                                                        // create an externally-owned
                                                                        // FERMI_VASPACE_A under this
                                                                        // device.  A later control RPC
                                                                        // will attach our page directory.
                                                                        const VASPACE_HANDLE: u32 =
                                                                            0x90f1_0000;
                                                                        const FERMI_VASPACE_A: u32 =
                                                                            0x0000_90f1;
                                                                        const NV_VASPACE_ALLOC_SIZE: usize =
                                                                            48;
                                                                        const VASPACE_ALLOC_SIZE: usize =
                                                                            RM_ALLOC_HEADER_SIZE
                                                                                + NV_VASPACE_ALLOC_SIZE;
                                                                        let mut vaspace_alloc =
                                                                            [0u8; VASPACE_ALLOC_SIZE];
                                                                        vaspace_alloc[0x00..0x04]
                                                                            .copy_from_slice(
                                                                                &CLIENT_HANDLE
                                                                                    .to_le_bytes());
                                                                        vaspace_alloc[0x04..0x08]
                                                                            .copy_from_slice(
                                                                                &DEVICE_HANDLE
                                                                                    .to_le_bytes());
                                                                        vaspace_alloc[0x08..0x0c]
                                                                            .copy_from_slice(
                                                                                &VASPACE_HANDLE
                                                                                    .to_le_bytes());
                                                                        vaspace_alloc[0x0c..0x10]
                                                                            .copy_from_slice(
                                                                                &FERMI_VASPACE_A
                                                                                    .to_le_bytes());
                                                                        vaspace_alloc[0x14..0x18]
                                                                            .copy_from_slice(
                                                                                &(NV_VASPACE_ALLOC_SIZE
                                                                                    as u32)
                                                                                    .to_le_bytes());
                                                                        // index=GPU_NEW is zero.  flags
                                                                        // is the second parameter dword.
                                                                        vaspace_alloc[0x24..0x28]
                                                                            .copy_from_slice(
                                                                                &8u32.to_le_bytes());

                                                                        let vaspace_seq = 6u32;
                                                                        let vaspace_pushed = unsafe {
                                                                            gsp::rpc_send_argc(
                                                                                &handshake,
                                                                                RPC_GSP_RM_ALLOC,
                                                                                &vaspace_alloc,
                                                                                VASPACE_ALLOC_SIZE,
                                                                                vaspace_seq)
                                                                        };
                                                                        match vaspace_pushed {
                                                                            Some(vaspace_csum) => {
                                                                                let vaspace_cmd_w = unsafe {
                                                                                    core::ptr::read_volatile(
                                                                                        cmdq.add(0x10)
                                                                                            as *const u32) };
                                                                                let vaspace_cmd_r_before = unsafe {
                                                                                    core::ptr::read_volatile(
                                                                                        msgq.add(
                                                                                            msg_rxoff as usize)
                                                                                            as *const u32) };
                                                                                let vaspace_msg_w_before = unsafe {
                                                                                    core::ptr::read_volatile(
                                                                                        msgq.add(0x10)
                                                                                            as *const u32) };
                                                                                gsp::rpc_doorbell(
                                                                                    gsp_falcon, &wr);
                                                                                let vaspace_doorbell =
                                                                                    rd(gsp_falcon + 0xc00);
                                                                                log.push(format!(
                                                                                    "  M3 VASPACE RPC sent: fn={} seq={} client=0x{:08x} parent=0x{:08x} object=0x{:08x} class=0x{:08x} flags=0x00000008 argc={} checksum=0x{:08x} doorbell_readback=0x{:08x}",
                                                                                    RPC_GSP_RM_ALLOC,
                                                                                    vaspace_seq,
                                                                                    CLIENT_HANDLE,
                                                                                    DEVICE_HANDLE,
                                                                                    VASPACE_HANDLE,
                                                                                    FERMI_VASPACE_A,
                                                                                    VASPACE_ALLOC_SIZE,
                                                                                    vaspace_csum,
                                                                                    vaspace_doorbell));

                                                                                let mut vaspace_matched = false;
                                                                                let mut vaspace_reply_seq =
                                                                                    u32::MAX;
                                                                                let mut vaspace_reply_fn = 0u32;
                                                                                let mut vaspace_rpc_result =
                                                                                    u32::MAX;
                                                                                let mut vaspace_status =
                                                                                    u32::MAX;
                                                                                let mut vaspace_reply_len = 0u32;
                                                                                let mut vaspace_size = 0u64;
                                                                                let mut vaspace_base = 0u64;
                                                                                for _ in 0..20000 {
                                                                                    let wptr = unsafe {
                                                                                        core::ptr::read_volatile(
                                                                                            msgq.add(0x10)
                                                                                                as *const u32) };
                                                                                    let rptr = unsafe {
                                                                                        core::ptr::read_volatile(
                                                                                            cmdq.add(0x20)
                                                                                                as *const u32) };
                                                                                    if wptr != rptr {
                                                                                        let elem = unsafe {
                                                                                            msgq.add(
                                                                                                tx_entryoff
                                                                                                    as usize
                                                                                                    + rptr
                                                                                                        as usize
                                                                                                        * 0x1000) };
                                                                                        vaspace_reply_seq = unsafe {
                                                                                            core::ptr::read_volatile(
                                                                                                elem.add(0x24)
                                                                                                    as *const u32) };
                                                                                        vaspace_reply_len = unsafe {
                                                                                            core::ptr::read_volatile(
                                                                                                elem.add(0x38)
                                                                                                    as *const u32) };
                                                                                        vaspace_reply_fn = unsafe {
                                                                                            core::ptr::read_volatile(
                                                                                                elem.add(0x3c)
                                                                                                    as *const u32) };
                                                                                        vaspace_rpc_result = unsafe {
                                                                                            core::ptr::read_volatile(
                                                                                                elem.add(0x40)
                                                                                                    as *const u32) };
                                                                                        if vaspace_reply_fn
                                                                                            == RPC_GSP_RM_ALLOC
                                                                                            && vaspace_reply_len
                                                                                                >= 0x20
                                                                                                    + RM_ALLOC_HEADER_SIZE
                                                                                                        as u32
                                                                                        {
                                                                                            vaspace_status = unsafe {
                                                                                                core::ptr::read_volatile(
                                                                                                    elem.add(0x60)
                                                                                                        as *const u32) };
                                                                                            if vaspace_reply_len
                                                                                                >= 0x20
                                                                                                    + VASPACE_ALLOC_SIZE
                                                                                                        as u32
                                                                                            {
                                                                                                let va_lo = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        elem.add(0x78)
                                                                                                            as *const u32) };
                                                                                                let va_hi = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        elem.add(0x7c)
                                                                                                            as *const u32) };
                                                                                                let base_lo = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        elem.add(0x98)
                                                                                                            as *const u32) };
                                                                                                let base_hi = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        elem.add(0x9c)
                                                                                                            as *const u32) };
                                                                                                vaspace_size = va_lo
                                                                                                    as u64
                                                                                                    | ((va_hi
                                                                                                        as u64)
                                                                                                        << 32);
                                                                                                vaspace_base = base_lo
                                                                                                    as u64
                                                                                                    | ((base_hi
                                                                                                        as u64)
                                                                                                        << 32);
                                                                                            }
                                                                                        }
                                                                                        let ec = unsafe {
                                                                                            core::ptr::read_volatile(
                                                                                                elem.add(0x28)
                                                                                                    as *const u32) };
                                                                                        let step = if ec == 0 {
                                                                                            1
                                                                                        } else {
                                                                                            ec
                                                                                        };
                                                                                        let mut next = rptr + step;
                                                                                        if next >= tx_count {
                                                                                            next -= tx_count;
                                                                                        }
                                                                                        core::sync::atomic::fence(
                                                                                            core::sync::atomic::Ordering::SeqCst);
                                                                                        unsafe {
                                                                                            core::ptr::write_volatile(
                                                                                                cmdq.add(0x20)
                                                                                                    as *mut u32,
                                                                                                next);
                                                                                        }
                                                                                        if vaspace_reply_fn
                                                                                            == RPC_GSP_RM_ALLOC
                                                                                        {
                                                                                            vaspace_matched = true;
                                                                                            break;
                                                                                        }
                                                                                    }
                                                                                    boot::stall(100);
                                                                                }
                                                                                let vaspace_cmd_r_final = unsafe {
                                                                                    core::ptr::read_volatile(
                                                                                        msgq.add(
                                                                                            msg_rxoff as usize)
                                                                                            as *const u32) };
                                                                                let vaspace_msg_w_final = unsafe {
                                                                                    core::ptr::read_volatile(
                                                                                        msgq.add(0x10)
                                                                                            as *const u32) };
                                                                                log.push(format!(
                                                                                    "  M3 VASPACE rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                                                    vaspace_cmd_w,
                                                                                    vaspace_cmd_r_before,
                                                                                    vaspace_cmd_r_final,
                                                                                    vaspace_msg_w_before,
                                                                                    vaspace_msg_w_final));
                                                                                log.push(format!(
                                                                                    "  *** M3 VASPACE reply: matched={} seq={} fn={} rpc_result=0x{:08x} alloc_status=0x{:08x} len=0x{:x} va_size=0x{:x} va_base=0x{:x} ***",
                                                                                    vaspace_matched,
                                                                                    vaspace_reply_seq,
                                                                                    vaspace_reply_fn,
                                                                                    vaspace_rpc_result,
                                                                                    vaspace_status,
                                                                                    vaspace_reply_len,
                                                                                    vaspace_size,
                                                                                    vaspace_base));
                                                                                if vaspace_matched
                                                                                    && vaspace_rpc_result == 0
                                                                                    && vaspace_status == 0
                                                                                {
                                                                                    log.push(format!(
                                                                                        "  *** M3 VASPACE SUCCESS: FERMI_VASPACE_A allocated handle=0x{:08x} parent=0x{:08x} external=true ***",
                                                                                        VASPACE_HANDLE,
                                                                                        DEVICE_HANDLE));

                                                                                    // Attach a client-owned
                                                                                    // GA10x top-level page
                                                                                    // directory to the VAS.
                                                                                    // gp100_vmm_desc_16[4]
                                                                                    // has two index bits, so
                                                                                    // RM must receive exactly
                                                                                    // four 64-bit entries.
                                                                                    let mut channel_pdb = None;
                                                                                    if static_valid {
                                                                                        for i in 0..static_fb_count
                                                                                            as usize
                                                                                        {
                                                                                            let first =
                                                                                                (static_fb_regions[i][0]
                                                                                                    + 0xfff)
                                                                                                    & !0xfff;
                                                                                            let pdb = first
                                                                                                .saturating_add(0x2000);
                                                                                            if static_fb_regions[i][2]
                                                                                                == 0
                                                                                                && static_fb_flags[i][3]
                                                                                                    == 0
                                                                                                && pdb
                                                                                                    .saturating_add(0x1000)
                                                                                                    <= static_fb_regions[i][1]
                                                                                                        .saturating_add(1)
                                                                                            {
                                                                                                channel_pdb =
                                                                                                    Some((i, pdb));
                                                                                                break;
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    // STEP 1: force the whole channel-memory chain into the
                                                                                    // 1 MiB non-WPR reserved slice at 0x2f3f00000 (ends AT the
                                                                                    // WPR boundary 0x2f4000000, no overlap) so it sits OUTSIDE
                                                                                    // GSP-RM's PMA-managed FB heap. Tests the heap-collision
                                                                                    // hypothesis behind the 0x1f channel reject.
                                                                                    const NONWPR_SLICE_BASE: u64 =
                                                                                        0x2f3f0_0000;
                                                                                    if static_valid {
                                                                                        for i in 0..static_fb_count
                                                                                            as usize
                                                                                        {
                                                                                            if static_fb_regions[i][0]
                                                                                                <= NONWPR_SLICE_BASE
                                                                                                && NONWPR_SLICE_BASE
                                                                                                    .saturating_add(0x9000)
                                                                                                    <= static_fb_regions[i][1]
                                                                                                        .saturating_add(1)
                                                                                            {
                                                                                                channel_pdb = Some((
                                                                                                    i,
                                                                                                    NONWPR_SLICE_BASE
                                                                                                        + 0x2000,
                                                                                                ));
                                                                                                break;
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    if let Some((pdb_region, pdb_phys)) =
                                                                                        channel_pdb
                                                                                    {
                                                                                        let window_reg =
                                                                                            bar0 + 0x001700;
                                                                                        let window_saved =
                                                                                            rd(window_reg);
                                                                                        wr(window_reg,
                                                                                            ((pdb_phys
                                                                                                & !0x000f_ffff)
                                                                                                >> 16)
                                                                                                as u32);
                                                                                        let pdb_aperture = bar0
                                                                                            + 0x700000
                                                                                            + (pdb_phys
                                                                                                & 0x000f_ffff);
                                                                                        for i in 0..8u64 {
                                                                                            wr(pdb_aperture + i * 4,
                                                                                                0);
                                                                                        }
                                                                                        core::sync::atomic::fence(
                                                                                            core::sync::atomic::Ordering::SeqCst);
                                                                                        let mut pdb_zero = true;
                                                                                        for i in 0..8u64 {
                                                                                            pdb_zero &= rd(
                                                                                                pdb_aperture + i * 4)
                                                                                                == 0;
                                                                                        }
                                                                                        wr(window_reg, window_saved);
                                                                                        log.push(format!(
                                                                                            "  M3 PDB init: FB[{}] phys=0x{:x} entries=4 zero={} aperture=VIDMEM",
                                                                                            pdb_region, pdb_phys,
                                                                                            pdb_zero));

                                                                                        const RPC_GSP_RM_CONTROL: u32 =
                                                                                            76;
                                                                                        const NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY: u32 =
                                                                                            0x0080_1813;
                                                                                        const RM_CONTROL_HEADER_SIZE: usize =
                                                                                            24;
                                                                                        const SET_PDB_PARAMS_SIZE: usize =
                                                                                            32;
                                                                                        const SET_PDB_RPC_SIZE: usize =
                                                                                            RM_CONTROL_HEADER_SIZE
                                                                                                + SET_PDB_PARAMS_SIZE;
                                                                                        let mut set_pdb =
                                                                                            [0u8; SET_PDB_RPC_SIZE];
                                                                                        set_pdb[0x00..0x04]
                                                                                            .copy_from_slice(
                                                                                                &CLIENT_HANDLE
                                                                                                    .to_le_bytes());
                                                                                        set_pdb[0x04..0x08]
                                                                                            .copy_from_slice(
                                                                                                &DEVICE_HANDLE
                                                                                                    .to_le_bytes());
                                                                                        set_pdb[0x08..0x0c]
                                                                                            .copy_from_slice(
                                                                                                &NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY
                                                                                                    .to_le_bytes());
                                                                                        set_pdb[0x10..0x14]
                                                                                            .copy_from_slice(
                                                                                                &(SET_PDB_PARAMS_SIZE
                                                                                                    as u32)
                                                                                                    .to_le_bytes());
                                                                                        set_pdb[0x18..0x20]
                                                                                            .copy_from_slice(
                                                                                                &pdb_phys
                                                                                                    .to_le_bytes());
                                                                                        set_pdb[0x20..0x24]
                                                                                            .copy_from_slice(
                                                                                                &4u32.to_le_bytes());
                                                                                        // flags=VIDMEM is zero.
                                                                                        set_pdb[0x28..0x2c]
                                                                                            .copy_from_slice(
                                                                                                &VASPACE_HANDLE
                                                                                                    .to_le_bytes());

                                                                                        let set_pdb_seq = 7u32;
                                                                                        let set_pdb_pushed = unsafe {
                                                                                            gsp::rpc_send_argc(
                                                                                                &handshake,
                                                                                                RPC_GSP_RM_CONTROL,
                                                                                                &set_pdb,
                                                                                                SET_PDB_RPC_SIZE,
                                                                                                set_pdb_seq)
                                                                                        };
                                                                                        match set_pdb_pushed {
                                                                                            Some(set_pdb_csum) => {
                                                                                                let set_pdb_cmd_w = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        cmdq.add(0x10)
                                                                                                            as *const u32) };
                                                                                                let set_pdb_cmd_r_before = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        msgq.add(
                                                                                                            msg_rxoff
                                                                                                                as usize)
                                                                                                            as *const u32) };
                                                                                                let set_pdb_msg_w_before = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        msgq.add(0x10)
                                                                                                            as *const u32) };
                                                                                                gsp::rpc_doorbell(
                                                                                                    gsp_falcon, &wr);
                                                                                                let set_pdb_doorbell =
                                                                                                    rd(gsp_falcon
                                                                                                        + 0xc00);
                                                                                                log.push(format!(
                                                                                                    "  M3 SET_PDB RPC sent: fn={} seq={} client=0x{:08x} object=0x{:08x} cmd=0x{:08x} vas=0x{:08x} pdb=0x{:x} entries=4 argc={} checksum=0x{:08x} doorbell_readback=0x{:08x}",
                                                                                                    RPC_GSP_RM_CONTROL,
                                                                                                    set_pdb_seq,
                                                                                                    CLIENT_HANDLE,
                                                                                                    DEVICE_HANDLE,
                                                                                                    NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY,
                                                                                                    VASPACE_HANDLE,
                                                                                                    pdb_phys,
                                                                                                    SET_PDB_RPC_SIZE,
                                                                                                    set_pdb_csum,
                                                                                                    set_pdb_doorbell));

                                                                                                let mut set_pdb_matched =
                                                                                                    false;
                                                                                                let mut set_pdb_reply_seq =
                                                                                                    u32::MAX;
                                                                                                let mut set_pdb_reply_fn = 0u32;
                                                                                                let mut set_pdb_rpc_result =
                                                                                                    u32::MAX;
                                                                                                let mut set_pdb_status =
                                                                                                    u32::MAX;
                                                                                                let mut set_pdb_reply_len = 0u32;
                                                                                                for _ in 0..20000 {
                                                                                                    let wptr = unsafe {
                                                                                                        core::ptr::read_volatile(
                                                                                                            msgq.add(0x10)
                                                                                                                as *const u32) };
                                                                                                    let rptr = unsafe {
                                                                                                        core::ptr::read_volatile(
                                                                                                            cmdq.add(0x20)
                                                                                                                as *const u32) };
                                                                                                    if wptr != rptr {
                                                                                                        let elem = unsafe {
                                                                                                            msgq.add(
                                                                                                                tx_entryoff
                                                                                                                    as usize
                                                                                                                    + rptr
                                                                                                                        as usize
                                                                                                                        * 0x1000) };
                                                                                                        set_pdb_reply_seq = unsafe {
                                                                                                            core::ptr::read_volatile(
                                                                                                                elem.add(0x24)
                                                                                                                    as *const u32) };
                                                                                                        set_pdb_reply_len = unsafe {
                                                                                                            core::ptr::read_volatile(
                                                                                                                elem.add(0x38)
                                                                                                                    as *const u32) };
                                                                                                        set_pdb_reply_fn = unsafe {
                                                                                                            core::ptr::read_volatile(
                                                                                                                elem.add(0x3c)
                                                                                                                    as *const u32) };
                                                                                                        set_pdb_rpc_result = unsafe {
                                                                                                            core::ptr::read_volatile(
                                                                                                                elem.add(0x40)
                                                                                                                    as *const u32) };
                                                                                                        if set_pdb_reply_fn
                                                                                                            == RPC_GSP_RM_CONTROL
                                                                                                            && set_pdb_reply_len
                                                                                                                >= 0x20
                                                                                                                    + RM_CONTROL_HEADER_SIZE
                                                                                                                        as u32
                                                                                                        {
                                                                                                            set_pdb_status = unsafe {
                                                                                                                core::ptr::read_volatile(
                                                                                                                    elem.add(0x5c)
                                                                                                                        as *const u32) };
                                                                                                        }
                                                                                                        let ec = unsafe {
                                                                                                            core::ptr::read_volatile(
                                                                                                                elem.add(0x28)
                                                                                                                    as *const u32) };
                                                                                                        let step = if ec == 0 {
                                                                                                            1
                                                                                                        } else {
                                                                                                            ec
                                                                                                        };
                                                                                                        let mut next = rptr + step;
                                                                                                        if next >= tx_count {
                                                                                                            next -= tx_count;
                                                                                                        }
                                                                                                        core::sync::atomic::fence(
                                                                                                            core::sync::atomic::Ordering::SeqCst);
                                                                                                        unsafe {
                                                                                                            core::ptr::write_volatile(
                                                                                                                cmdq.add(0x20)
                                                                                                                    as *mut u32,
                                                                                                                next);
                                                                                                        }
                                                                                                        if set_pdb_reply_fn
                                                                                                            == RPC_GSP_RM_CONTROL
                                                                                                        {
                                                                                                            set_pdb_matched = true;
                                                                                                            break;
                                                                                                        }
                                                                                                    }
                                                                                                    boot::stall(100);
                                                                                                }
                                                                                                let set_pdb_cmd_r_final = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        msgq.add(
                                                                                                            msg_rxoff
                                                                                                                as usize)
                                                                                                            as *const u32) };
                                                                                                let set_pdb_msg_w_final = unsafe {
                                                                                                    core::ptr::read_volatile(
                                                                                                        msgq.add(0x10)
                                                                                                            as *const u32) };
                                                                                                wr(window_reg,
                                                                                                    ((pdb_phys
                                                                                                        & !0x000f_ffff)
                                                                                                        >> 16)
                                                                                                        as u32);
                                                                                                let mut pdb_after =
                                                                                                    [0u64; 4];
                                                                                                for i in 0..4usize {
                                                                                                    let lo = rd(
                                                                                                        pdb_aperture
                                                                                                            + (i * 8)
                                                                                                                as u64);
                                                                                                    let hi = rd(
                                                                                                        pdb_aperture
                                                                                                            + (i * 8 + 4)
                                                                                                                as u64);
                                                                                                    pdb_after[i] = lo
                                                                                                        as u64
                                                                                                        | ((hi as u64)
                                                                                                            << 32);
                                                                                                }
                                                                                                wr(window_reg,
                                                                                                    window_saved);
                                                                                                log.push(format!(
                                                                                                    "  M3 SET_PDB rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                                                                    set_pdb_cmd_w,
                                                                                                    set_pdb_cmd_r_before,
                                                                                                    set_pdb_cmd_r_final,
                                                                                                    set_pdb_msg_w_before,
                                                                                                    set_pdb_msg_w_final));
                                                                                                log.push(format!(
                                                                                                    "  M3 SET_PDB entries: [{:016x},{:016x},{:016x},{:016x}]",
                                                                                                    pdb_after[0],
                                                                                                    pdb_after[1],
                                                                                                    pdb_after[2],
                                                                                                    pdb_after[3]));
                                                                                                log.push(format!(
                                                                                                    "  *** M3 SET_PDB reply: matched={} seq={} fn={} rpc_result=0x{:08x} ctrl_status=0x{:08x} len=0x{:x} ***",
                                                                                                    set_pdb_matched,
                                                                                                    set_pdb_reply_seq,
                                                                                                    set_pdb_reply_fn,
                                                                                                    set_pdb_rpc_result,
                                                                                                    set_pdb_status,
                                                                                                    set_pdb_reply_len));
                                                                                                if pdb_zero
                                                                                                    && set_pdb_matched
                                                                                                    && set_pdb_rpc_result
                                                                                                        == 0
                                                                                                    && set_pdb_status == 0
                                                                                                {
                                                                                                    log.push(format!(
                                                                                                        "  *** M3 SET_PDB SUCCESS: external VAS 0x{:08x} attached to VIDMEM PDB 0x{:x} ***",
                                                                                                        VASPACE_HANDLE,
                                                                                                        pdb_phys));

                                                                                                    // Materialize the
                                                                                                    // first owned 4K VA
                                                                                                    // mapping beneath the
                                                                                                    // attached PD3 root.
                                                                                                    let map_va =
                                                                                                        vaspace_base;
                                                                                                    let pd2_phys =
                                                                                                        pdb_phys + 0x1000;
                                                                                                    let pd1_phys =
                                                                                                        pdb_phys + 0x2000;
                                                                                                    let pd0_phys =
                                                                                                        pdb_phys + 0x3000;
                                                                                                    let spt_phys =
                                                                                                        pdb_phys + 0x4000;
                                                                                                    let data_phys =
                                                                                                        pdb_phys + 0x5000;
                                                                                                    let chain_fits =
                                                                                                        data_phys
                                                                                                            .saturating_add(
                                                                                                                0x1000)
                                                                                                            <= static_fb_regions
                                                                                                                [pdb_region][1]
                                                                                                                .saturating_add(1);
                                                                                                    if chain_fits {
                                                                                                        let fb_aperture =
                                                                                                            |addr: u64| {
                                                                                                                wr(window_reg,
                                                                                                                    ((addr
                                                                                                                        & !0x000f_ffff)
                                                                                                                        >> 16)
                                                                                                                        as u32);
                                                                                                                bar0
                                                                                                                    + 0x700000
                                                                                                                    + (addr
                                                                                                                        & 0x000f_ffff)
                                                                                                            };
                                                                                                        let fb_write64 =
                                                                                                            |addr: u64,
                                                                                                             value: u64| {
                                                                                                                let ap =
                                                                                                                    fb_aperture(
                                                                                                                        addr);
                                                                                                                wr(ap,
                                                                                                                    value
                                                                                                                        as u32);
                                                                                                                wr(ap + 4,
                                                                                                                    (value
                                                                                                                        >> 32)
                                                                                                                        as u32);
                                                                                                            };
                                                                                                        let fb_read64 =
                                                                                                            |addr: u64|
                                                                                                             -> u64 {
                                                                                                                let ap =
                                                                                                                    fb_aperture(
                                                                                                                        addr);
                                                                                                                rd(ap)
                                                                                                                    as u64
                                                                                                                    | ((rd(ap
                                                                                                                        + 4)
                                                                                                                        as u64)
                                                                                                                        << 32)
                                                                                                            };

                                                                                                        for page in [
                                                                                                            pd2_phys,
                                                                                                            pd1_phys,
                                                                                                            pd0_phys,
                                                                                                            spt_phys,
                                                                                                        ] {
                                                                                                            let ap =
                                                                                                                fb_aperture(
                                                                                                                    page);
                                                                                                            for dword in
                                                                                                                0..1024u64
                                                                                                            {
                                                                                                                wr(ap
                                                                                                                    + dword
                                                                                                                        * 4,
                                                                                                                    0);
                                                                                                            }
                                                                                                        }

                                                                                                        let pd3i =
                                                                                                            ((map_va >> 47)
                                                                                                                & 0x3)
                                                                                                                as u64;
                                                                                                        let pd2i =
                                                                                                            ((map_va >> 38)
                                                                                                                & 0x1ff)
                                                                                                                as u64;
                                                                                                        let pd1i =
                                                                                                            ((map_va >> 29)
                                                                                                                & 0x1ff)
                                                                                                                as u64;
                                                                                                        let pd0i =
                                                                                                            ((map_va >> 21)
                                                                                                                & 0xff)
                                                                                                                as u64;
                                                                                                        let spti =
                                                                                                            ((map_va >> 12)
                                                                                                                & 0x1ff)
                                                                                                                as u64;
                                                                                                        let pd3e =
                                                                                                            (pd2_phys >> 4)
                                                                                                                | 2;
                                                                                                        let pd2e =
                                                                                                            (pd1_phys >> 4)
                                                                                                                | 2;
                                                                                                        let pd1e =
                                                                                                            (pd0_phys >> 4)
                                                                                                                | 2;
                                                                                                        let pd0e_small =
                                                                                                            (spt_phys >> 4)
                                                                                                                | 2;
                                                                                                        let spte =
                                                                                                            (data_phys >> 4)
                                                                                                                | 1;
                                                                                                        fb_write64(
                                                                                                            pdb_phys
                                                                                                                + pd3i * 8,
                                                                                                            pd3e);
                                                                                                        fb_write64(
                                                                                                            pd2_phys
                                                                                                                + pd2i * 8,
                                                                                                            pd2e);
                                                                                                        fb_write64(
                                                                                                            pd1_phys
                                                                                                                + pd1i * 8,
                                                                                                            pd1e);
                                                                                                        // PD0 entries are
                                                                                                        // 128-bit: large PDE
                                                                                                        // first, small PDE
                                                                                                        // second.
                                                                                                        fb_write64(
                                                                                                            pd0_phys
                                                                                                                + pd0i
                                                                                                                    * 0x10,
                                                                                                            0);
                                                                                                        fb_write64(
                                                                                                            pd0_phys
                                                                                                                + pd0i
                                                                                                                    * 0x10
                                                                                                                + 8,
                                                                                                            pd0e_small);
                                                                                                        fb_write64(
                                                                                                            spt_phys
                                                                                                                + spti * 8,
                                                                                                            spte);
                                                                                                        let map_pattern = [
                                                                                                            0x5641_4d33u32,
                                                                                                            map_va as u32,
                                                                                                            data_phys as u32,
                                                                                                            0x4d41_5050u32,
                                                                                                        ];
                                                                                                        let data_ap =
                                                                                                            fb_aperture(
                                                                                                                data_phys);
                                                                                                        for i in 0..4usize {
                                                                                                            wr(data_ap
                                                                                                                + (i * 4)
                                                                                                                    as u64,
                                                                                                                map_pattern[i]);
                                                                                                        }
                                                                                                        core::sync::atomic::fence(
                                                                                                            core::sync::atomic::Ordering::SeqCst);

                                                                                                        let chain_seen = [
                                                                                                            fb_read64(
                                                                                                                pdb_phys
                                                                                                                    + pd3i
                                                                                                                        * 8),
                                                                                                            fb_read64(
                                                                                                                pd2_phys
                                                                                                                    + pd2i
                                                                                                                        * 8),
                                                                                                            fb_read64(
                                                                                                                pd1_phys
                                                                                                                    + pd1i
                                                                                                                        * 8),
                                                                                                            fb_read64(
                                                                                                                pd0_phys
                                                                                                                    + pd0i
                                                                                                                        * 0x10
                                                                                                                    + 8),
                                                                                                            fb_read64(
                                                                                                                spt_phys
                                                                                                                    + spti
                                                                                                                        * 8),
                                                                                                        ];
                                                                                                        let chain_expected = [
                                                                                                            pd3e,
                                                                                                            pd2e,
                                                                                                            pd1e,
                                                                                                            pd0e_small,
                                                                                                            spte,
                                                                                                        ];
                                                                                                        let mut data_seen =
                                                                                                            [0u32; 4];
                                                                                                        let data_ap =
                                                                                                            fb_aperture(
                                                                                                                data_phys);
                                                                                                        for i in 0..4usize {
                                                                                                            data_seen[i] =
                                                                                                                rd(data_ap
                                                                                                                    + (i * 4)
                                                                                                                        as u64);
                                                                                                        }
                                                                                                        wr(window_reg,
                                                                                                            window_saved);

                                                                                                        wr(bar0 + 0xb830b0,
                                                                                                            0x8000_0007);
                                                                                                        let mut map_flush_done =
                                                                                                            false;
                                                                                                        let mut map_flush = 0u32;
                                                                                                        for _ in 0..20000 {
                                                                                                            map_flush = rd(
                                                                                                                bar0
                                                                                                                    + 0xb830b0);
                                                                                                            if map_flush
                                                                                                                & 0x8000_0000
                                                                                                                == 0
                                                                                                            {
                                                                                                                map_flush_done =
                                                                                                                    true;
                                                                                                                break;
                                                                                                            }
                                                                                                            boot::stall(10);
                                                                                                        }
                                                                                                        let chain_ok =
                                                                                                            chain_seen
                                                                                                                == chain_expected;
                                                                                                        let data_ok =
                                                                                                            data_seen
                                                                                                                == map_pattern;
                                                                                                        log.push(format!(
                                                                                                            "  M3 VAMAP pages: va=0x{:x} pdb=[0x{:x},0x{:x},0x{:x},0x{:x},0x{:x}] data=0x{:x} idx=[{},{},{},{},{}]",
                                                                                                            map_va,
                                                                                                            pdb_phys,
                                                                                                            pd2_phys,
                                                                                                            pd1_phys,
                                                                                                            pd0_phys,
                                                                                                            spt_phys,
                                                                                                            data_phys,
                                                                                                            pd3i,
                                                                                                            pd2i,
                                                                                                            pd1i,
                                                                                                            pd0i,
                                                                                                            spti));
                                                                                                        log.push(format!(
                                                                                                            "  M3 VAMAP entries: expected=[{:016x},{:016x},{:016x},{:016x},{:016x}] observed=[{:016x},{:016x},{:016x},{:016x},{:016x}] match={}",
                                                                                                            chain_expected[0],
                                                                                                            chain_expected[1],
                                                                                                            chain_expected[2],
                                                                                                            chain_expected[3],
                                                                                                            chain_expected[4],
                                                                                                            chain_seen[0],
                                                                                                            chain_seen[1],
                                                                                                            chain_seen[2],
                                                                                                            chain_seen[3],
                                                                                                            chain_seen[4],
                                                                                                            chain_ok));
                                                                                                        log.push(format!(
                                                                                                            "  M3 VAMAP data: expected=[{:08x},{:08x},{:08x},{:08x}] observed=[{:08x},{:08x},{:08x},{:08x}] match={} flush_done={} flush=0x{:08x}",
                                                                                                            map_pattern[0],
                                                                                                            map_pattern[1],
                                                                                                            map_pattern[2],
                                                                                                            map_pattern[3],
                                                                                                            data_seen[0],
                                                                                                            data_seen[1],
                                                                                                            data_seen[2],
                                                                                                            data_seen[3],
                                                                                                            data_ok,
                                                                                                            map_flush_done,
                                                                                                            map_flush));
                                                                                                        if chain_ok
                                                                                                            && data_ok
                                                                                                            && map_flush_done
                                                                                                        {
                                                                                                            log.push(format!(
                                                                                                                "  *** M3 VAMAP SUCCESS: VA 0x{:x} maps writable VRAM page 0x{:x} through owned GA10x tables ***",
                                                                                                                map_va,
                                                                                                                data_phys));

                                                                                                            // r535 has NO TSG and NO fn=4 ALLOC_MEMORY (class 0x82) — both are
                                                                                                            // vGPU-only stubs on GA104 (assert). Flow straight from the owned
                                                                                                            // VAMAP to the MTHDBUF-size query + AMPERE_CHANNEL_GPFIFO_A alloc,
                                                                                                            // feeding the PTE-mapped map_va as the bare gpFifoOffset.
                                                                                                                        // Match nouveau r535_fifo_runl_ctor: force FIFO device/runlist
                                                                                                                        // discovery on GSP's RM-INTERNAL subdevice before channel
                                                                                                                        // construction. The R570 ABI returns the fixed table inline.
                                                                                                                        const NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE: u32 =
                                                                                                                            0x2080_1112;
                                                                                                                        const DEVINFO_MAX_ENTRIES: usize = 32;
                                                                                                                        const DEVINFO_ENTRY_SIZE: usize = 0x64;
                                                                                                                        const DEVINFO_PARAMS_SIZE: usize =
                                                                                                                            0x0c + DEVINFO_MAX_ENTRIES * DEVINFO_ENTRY_SIZE;
                                                                                                                        const DEVINFO_CONTROL_SIZE: usize =
                                                                                                                            RM_CONTROL_HEADER_SIZE + DEVINFO_PARAMS_SIZE;
                                                                                                                        const DEVINFO_REPLY_LEN: u32 =
                                                                                                                            (0x18 + DEVINFO_CONTROL_SIZE) as u32;
                                                                                                                        let mut devinfo_query =
                                                                                                                            [0u8; DEVINFO_CONTROL_SIZE];
                                                                                                                        devinfo_query[0x00..0x04]
                                                                                                                            .copy_from_slice(
                                                                                                                                &static_handles[0].to_le_bytes());
                                                                                                                        devinfo_query[0x04..0x08]
                                                                                                                            .copy_from_slice(
                                                                                                                                &static_handles[2].to_le_bytes());
                                                                                                                        devinfo_query[0x08..0x0c]
                                                                                                                            .copy_from_slice(
                                                                                                                                &NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE
                                                                                                                                    .to_le_bytes());
                                                                                                                        devinfo_query[0x10..0x14]
                                                                                                                            .copy_from_slice(
                                                                                                                                &(DEVINFO_PARAMS_SIZE as u32).to_le_bytes());
                                                                                                                        // baseIndex at params +0x00 remains zero. numEntries,
                                                                                                                        // bMore, and all 32 inline output entries are zeroed.
                                                                                                                        let devinfo_seq = 8u32;
                                                                                                                        let mut devinfo_ok = false;
                                                                                                                        let mut devinfo_base_index = u32::MAX;
                                                                                                                        let mut devinfo_num_entries = 0u32;
                                                                                                                        let mut devinfo_more = 0u8;
                                                                                                                        let mut devinfo_gr_found = false;
                                                                                                                        // r570_fifo_xlat_rm_engine_type maps the table's
                                                                                                                        // RM_ENGINE_TYPE_GR0 entry to NV2080_ENGINE_TYPE_GR0.
                                                                                                                        // Keep the two namespaces explicit even though GR0 is
                                                                                                                        // ordinal 1 in both of them.
                                                                                                                        let mut devinfo_gr_engine_type = 0u32;
                                                                                                                        let mut devinfo_gr = [0u32; 6];
                                                                                                                        match unsafe {
                                                                                                                            gsp::rpc_send_argc(
                                                                                                                                &handshake,
                                                                                                                                RPC_GSP_RM_CONTROL,
                                                                                                                                &devinfo_query,
                                                                                                                                DEVINFO_CONTROL_SIZE,
                                                                                                                                devinfo_seq)
                                                                                                                        } {
                                                                                                                            Some(devinfo_csum) => {
                                                                                                                                let devinfo_cmd_w = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        cmdq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let devinfo_cmd_r_before = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(
                                                                                                                                            msg_rxoff as usize)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let devinfo_msg_w_before = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                gsp::rpc_doorbell(
                                                                                                                                    gsp_falcon, &wr);
                                                                                                                                let mut devinfo_matched = false;
                                                                                                                                let mut devinfo_reply_seq = u32::MAX;
                                                                                                                                let mut devinfo_reply_fn = 0u32;
                                                                                                                                let mut devinfo_reply_len = 0u32;
                                                                                                                                let mut devinfo_rpc_result = u32::MAX;
                                                                                                                                let mut devinfo_status = u32::MAX;
                                                                                                                                for _ in 0..20000 {
                                                                                                                                    let wptr = unsafe {
                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                            msgq.add(0x10)
                                                                                                                                                as *const u32)
                                                                                                                                    };
                                                                                                                                    let rptr = unsafe {
                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                            cmdq.add(0x20)
                                                                                                                                                as *const u32)
                                                                                                                                    };
                                                                                                                                    if wptr != rptr {
                                                                                                                                        let elem = unsafe {
                                                                                                                                            msgq.add(
                                                                                                                                                tx_entryoff as usize
                                                                                                                                                    + rptr as usize
                                                                                                                                                        * 0x1000)
                                                                                                                                        };
                                                                                                                                        devinfo_reply_seq = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x24)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        devinfo_reply_len = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x38)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        devinfo_reply_fn = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x3c)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        devinfo_rpc_result = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x40)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        if devinfo_reply_fn
                                                                                                                                            == RPC_GSP_RM_CONTROL
                                                                                                                                        {
                                                                                                                                            devinfo_status = unsafe {
                                                                                                                                                core::ptr::read_volatile(
                                                                                                                                                    elem.add(0x5c)
                                                                                                                                                        as *const u32)
                                                                                                                                            };
                                                                                                                                            if devinfo_rpc_result == 0
                                                                                                                                                && devinfo_status == 0
                                                                                                                                                && devinfo_reply_len
                                                                                                                                                    >= DEVINFO_REPLY_LEN
                                                                                                                                            {
                                                                                                                                                devinfo_base_index = unsafe {
                                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                                        elem.add(0x68)
                                                                                                                                                            as *const u32)
                                                                                                                                                };
                                                                                                                                                devinfo_num_entries = unsafe {
                                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                                        elem.add(0x6c)
                                                                                                                                                            as *const u32)
                                                                                                                                                };
                                                                                                                                                devinfo_more = unsafe {
                                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                                        elem.add(0x70)
                                                                                                                                                            as *const u8)
                                                                                                                                                };
                                                                                                                                                let count = core::cmp::min(
                                                                                                                                                    devinfo_num_entries as usize,
                                                                                                                                                    DEVINFO_MAX_ENTRIES);
                                                                                                                                                for i in 0..count {
                                                                                                                                                    let entry = unsafe {
                                                                                                                                                        elem.add(
                                                                                                                                                            0x74
                                                                                                                                                                + i * DEVINFO_ENTRY_SIZE)
                                                                                                                                                    };
                                                                                                                                                    let eng_desc = unsafe {
                                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                                            entry.add(0x00)
                                                                                                                                                                as *const u32)
                                                                                                                                                    };
                                                                                                                                                    let rm_engine = unsafe {
                                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                                            entry.add(0x08)
                                                                                                                                                                as *const u32)
                                                                                                                                                    };
                                                                                                                                                    let runlist = unsafe {
                                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                                            entry.add(0x0c)
                                                                                                                                                                as *const u32)
                                                                                                                                                    };
                                                                                                                                                    let runlist_pri = unsafe {
                                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                                            entry.add(0x2c)
                                                                                                                                                                as *const u32)
                                                                                                                                                    };
                                                                                                                                                    let pbdma0 = unsafe {
                                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                                            entry.add(0x40)
                                                                                                                                                                as *const u32)
                                                                                                                                                    };
                                                                                                                                                    let pbdma1 = unsafe {
                                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                                            entry.add(0x44)
                                                                                                                                                                as *const u32)
                                                                                                                                                    };
                                                                                                                                                    let num_pbdmas = unsafe {
                                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                                            entry.add(0x50)
                                                                                                                                                                as *const u32)
                                                                                                                                                    };
                                                                                                                                                    log.push(format!(
                                                                                                                                                        "  M3 DEVINFO[{}]: eng_desc=0x{:08x} rm_engine=0x{:08x} runlist={} runlist_pri=0x{:08x} pbdma=[0x{:08x},0x{:08x}] num_pbdmas={}",
                                                                                                                                                        i,
                                                                                                                                                        eng_desc,
                                                                                                                                                        rm_engine,
                                                                                                                                                        runlist,
                                                                                                                                                        runlist_pri,
                                                                                                                                                        pbdma0,
                                                                                                                                                        pbdma1,
                                                                                                                                                        num_pbdmas));
                                                                                                                                                    if rm_engine == 1 {
                                                                                                                                                        devinfo_gr_found = true;
                                                                                                                                                        devinfo_gr_engine_type = 1;
                                                                                                                                                        devinfo_gr = [
                                                                                                                                                            eng_desc,
                                                                                                                                                            runlist,
                                                                                                                                                            runlist_pri,
                                                                                                                                                            pbdma0,
                                                                                                                                                            pbdma1,
                                                                                                                                                            num_pbdmas,
                                                                                                                                                        ];
                                                                                                                                                    }
                                                                                                                                                }
                                                                                                                                                devinfo_ok =
                                                                                                                                                    devinfo_num_entries != 0;
                                                                                                                                            }
                                                                                                                                        }
                                                                                                                                        let ec = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x28)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        let step =
                                                                                                                                            if ec == 0 { 1 } else { ec };
                                                                                                                                        let mut next = rptr + step;
                                                                                                                                        if next >= tx_count {
                                                                                                                                            next -= tx_count;
                                                                                                                                        }
                                                                                                                                        core::sync::atomic::fence(
                                                                                                                                            core::sync::atomic::Ordering::SeqCst);
                                                                                                                                        unsafe {
                                                                                                                                            core::ptr::write_volatile(
                                                                                                                                                cmdq.add(0x20)
                                                                                                                                                    as *mut u32,
                                                                                                                                                next);
                                                                                                                                        }
                                                                                                                                        if devinfo_reply_fn
                                                                                                                                            == RPC_GSP_RM_CONTROL
                                                                                                                                        {
                                                                                                                                            devinfo_matched = true;
                                                                                                                                            break;
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                    boot::stall(100);
                                                                                                                                }
                                                                                                                                let devinfo_cmd_r_final = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(
                                                                                                                                            msg_rxoff as usize)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let devinfo_msg_w_final = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                log.push(format!(
                                                                                                                                    "  M3 DEVINFO RPC sent: fn={} seq={} client=0x{:08x} object=0x{:08x} cmd=0x{:08x} params=0x{:x} argc=0x{:x} checksum=0x{:08x}",
                                                                                                                                    RPC_GSP_RM_CONTROL,
                                                                                                                                    devinfo_seq,
                                                                                                                                    static_handles[0],
                                                                                                                                    static_handles[2],
                                                                                                                                    NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE,
                                                                                                                                    DEVINFO_PARAMS_SIZE,
                                                                                                                                    DEVINFO_CONTROL_SIZE,
                                                                                                                                    devinfo_csum));
                                                                                                                                log.push(format!(
                                                                                                                                    "  M3 DEVINFO rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                                                                                                    devinfo_cmd_w,
                                                                                                                                    devinfo_cmd_r_before,
                                                                                                                                    devinfo_cmd_r_final,
                                                                                                                                    devinfo_msg_w_before,
                                                                                                                                    devinfo_msg_w_final));
                                                                                                                                log.push(format!(
                                                                                                                                    "  *** M3 DEVINFO reply: matched={} seq={} fn={} rpc_result=0x{:08x} ctrl_status=0x{:08x} len=0x{:x} base={} entries={} more={} ok={} gr_found={} ***",
                                                                                                                                    devinfo_matched,
                                                                                                                                    devinfo_reply_seq,
                                                                                                                                    devinfo_reply_fn,
                                                                                                                                    devinfo_rpc_result,
                                                                                                                                    devinfo_status,
                                                                                                                                    devinfo_reply_len,
                                                                                                                                    devinfo_base_index,
                                                                                                                                    devinfo_num_entries,
                                                                                                                                    devinfo_more,
                                                                                                                                    devinfo_ok,
                                                                                                                                    devinfo_gr_found));
                                                                                                                                if devinfo_gr_found {
                                                                                                                                    log.push(format!(
                                                                                                                                        "  *** M3 DEVINFO GR0: eng_desc=0x{:08x} runlist={} runlist_pri=0x{:08x} pbdma=[0x{:08x},0x{:08x}] num_pbdmas={} ***",
                                                                                                                                        devinfo_gr[0],
                                                                                                                                        devinfo_gr[1],
                                                                                                                                        devinfo_gr[2],
                                                                                                                                        devinfo_gr[3],
                                                                                                                                        devinfo_gr[4],
                                                                                                                                        devinfo_gr[5]));
                                                                                                                                }
                                                                                                                            }
                                                                                                                            None => log.push(String::from(
                                                                                                                                "  M3 DEVINFO RPC push FAILED (cmdq full?)")),
                                                                                                                        }
                                                                                                                        // r535_fifo_oneinit queries this
                                                                                                                        // exact size before allocating a
                                                                                                                        // channel's coherent CE fault
                                                                                                                        // method buffer.
                                                                                                                        const NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE: u32 =
                                                                                                                            0x2080_2a08;
                                                                                                                        const MTHDBUF_QUERY_SIZE: usize =
                                                                                                                            RM_CONTROL_HEADER_SIZE + 4;
                                                                                                                        let mut mthdbuf_query =
                                                                                                                            [0u8; MTHDBUF_QUERY_SIZE];
                                                                                                                        // CE_GET_FAULT_METHOD_BUFFER_SIZE must be issued on GSP's
                                                                                                                        // RM-INTERNAL client+subdevice (from GET_GSP_STATIC_INFO
                                                                                                                        // reply), NOT our client handles — GSP can only resolve the
                                                                                                                        // internal object tree. static_handles[0]=hInternalClient,
                                                                                                                        // [2]=hInternalSubdevice. (nouveau r535/fifo.c:567 uses
                                                                                                                        // gsp->internal.device.subdevice; OpenRM gpu.c:902-904.)
                                                                                                                        mthdbuf_query[0x00..0x04]
                                                                                                                            .copy_from_slice(
                                                                                                                                &static_handles[0].to_le_bytes());
                                                                                                                        mthdbuf_query[0x04..0x08]
                                                                                                                            .copy_from_slice(
                                                                                                                                &static_handles[2].to_le_bytes());
                                                                                                                        mthdbuf_query[0x08..0x0c]
                                                                                                                            .copy_from_slice(
                                                                                                                                &NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE
                                                                                                                                    .to_le_bytes());
                                                                                                                        mthdbuf_query[0x10..0x14]
                                                                                                                            .copy_from_slice(
                                                                                                                                &4u32.to_le_bytes());
                                                                                                                        let mthdbuf_seq = 9u32;
                                                                                                                        let mut mthdbuf_size = 0u32;
                                                                                                                        let mut mthdbuf_query_ok = false;
                                                                                                                        match unsafe {
                                                                                                                            gsp::rpc_send_argc(
                                                                                                                                &handshake,
                                                                                                                                RPC_GSP_RM_CONTROL,
                                                                                                                                &mthdbuf_query,
                                                                                                                                MTHDBUF_QUERY_SIZE,
                                                                                                                                mthdbuf_seq)
                                                                                                                        } {
                                                                                                                            Some(mthdbuf_csum) => {
                                                                                                                                let mthdbuf_cmd_w = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        cmdq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let mthdbuf_cmd_r_before = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(
                                                                                                                                            msg_rxoff as usize)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let mthdbuf_msg_w_before = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                gsp::rpc_doorbell(
                                                                                                                                    gsp_falcon, &wr);
                                                                                                                                let mut mthdbuf_matched = false;
                                                                                                                                let mut mthdbuf_reply_seq =
                                                                                                                                    u32::MAX;
                                                                                                                                let mut mthdbuf_reply_fn = 0u32;
                                                                                                                                let mut mthdbuf_rpc_result =
                                                                                                                                    u32::MAX;
                                                                                                                                let mut mthdbuf_status =
                                                                                                                                    u32::MAX;
                                                                                                                                for _ in 0..20000 {
                                                                                                                                    let wptr = unsafe {
                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                            msgq.add(0x10)
                                                                                                                                                as *const u32)
                                                                                                                                    };
                                                                                                                                    let rptr = unsafe {
                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                            cmdq.add(0x20)
                                                                                                                                                as *const u32)
                                                                                                                                    };
                                                                                                                                    if wptr != rptr {
                                                                                                                                        let elem = unsafe {
                                                                                                                                            msgq.add(
                                                                                                                                                tx_entryoff as usize
                                                                                                                                                    + rptr as usize
                                                                                                                                                        * 0x1000)
                                                                                                                                        };
                                                                                                                                        mthdbuf_reply_seq = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x24)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        mthdbuf_reply_fn = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x3c)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        mthdbuf_rpc_result = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x40)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        if mthdbuf_reply_fn
                                                                                                                                            == RPC_GSP_RM_CONTROL
                                                                                                                                        {
                                                                                                                                            mthdbuf_status = unsafe {
                                                                                                                                                core::ptr::read_volatile(
                                                                                                                                                    elem.add(0x5c)
                                                                                                                                                        as *const u32)
                                                                                                                                            };
                                                                                                                                            mthdbuf_size = unsafe {
                                                                                                                                                core::ptr::read_volatile(
                                                                                                                                                    elem.add(0x68)
                                                                                                                                                        as *const u32)
                                                                                                                                            };
                                                                                                                                        }
                                                                                                                                        let ec = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x28)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        let step =
                                                                                                                                            if ec == 0 { 1 } else { ec };
                                                                                                                                        let mut next = rptr + step;
                                                                                                                                        if next >= tx_count {
                                                                                                                                            next -= tx_count;
                                                                                                                                        }
                                                                                                                                        core::sync::atomic::fence(
                                                                                                                                            core::sync::atomic::Ordering::SeqCst);
                                                                                                                                        unsafe {
                                                                                                                                            core::ptr::write_volatile(
                                                                                                                                                cmdq.add(0x20)
                                                                                                                                                    as *mut u32,
                                                                                                                                                next);
                                                                                                                                        }
                                                                                                                                        if mthdbuf_reply_fn
                                                                                                                                            == RPC_GSP_RM_CONTROL
                                                                                                                                        {
                                                                                                                                            mthdbuf_matched = true;
                                                                                                                                            break;
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                    boot::stall(100);
                                                                                                                                }
                                                                                                                                let mthdbuf_cmd_r_final = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(
                                                                                                                                            msg_rxoff as usize)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let mthdbuf_msg_w_final = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                mthdbuf_query_ok =
                                                                                                                                    mthdbuf_matched
                                                                                                                                        && mthdbuf_rpc_result == 0
                                                                                                                                        && mthdbuf_status == 0
                                                                                                                                        && mthdbuf_size != 0;
                                                                                                                                log.push(format!(
                                                                                                                                    "  M3 MTHDBUF RPC sent: fn={} seq={} client=0x{:08x} object=0x{:08x} cmd=0x{:08x} argc={} checksum=0x{:08x}",
                                                                                                                                    RPC_GSP_RM_CONTROL,
                                                                                                                                    mthdbuf_seq,
                                                                                                                                    static_handles[0],
                                                                                                                                    static_handles[2],
                                                                                                                                    NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE,
                                                                                                                                    MTHDBUF_QUERY_SIZE,
                                                                                                                                    mthdbuf_csum));
                                                                                                                                log.push(format!(
                                                                                                                                    "  M3 MTHDBUF rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                                                                                                    mthdbuf_cmd_w,
                                                                                                                                    mthdbuf_cmd_r_before,
                                                                                                                                    mthdbuf_cmd_r_final,
                                                                                                                                    mthdbuf_msg_w_before,
                                                                                                                                    mthdbuf_msg_w_final));
                                                                                                                                log.push(format!(
                                                                                                                                    "  *** M3 MTHDBUF reply: matched={} seq={} fn={} rpc_result=0x{:08x} ctrl_status=0x{:08x} size=0x{:x} ***",
                                                                                                                                    mthdbuf_matched,
                                                                                                                                    mthdbuf_reply_seq,
                                                                                                                                    mthdbuf_reply_fn,
                                                                                                                                    mthdbuf_rpc_result,
                                                                                                                                    mthdbuf_status,
                                                                                                                                    mthdbuf_size));
                                                                                                                            }
                                                                                                                            None => log.push(String::from(
                                                                                                                                "  M3 MTHDBUF RPC push FAILED (cmdq full?)")),
                                                                                                                        }

                                                                                                                        let mthdbuf_dma = if mthdbuf_query_ok
                                                                                                                            && devinfo_ok
                                                                                                                            && devinfo_gr_found
                                                                                                                            && devinfo_gr_engine_type != 0
                                                                                                                        {
                                                                                                                            dma.alloc_pages(
                                                                                                                                (mthdbuf_size as usize + 0xfff)
                                                                                                                                    / 0x1000)
                                                                                                                        } else {
                                                                                                                            None
                                                                                                                        };
                                                                                                                        if let Some(mthdbuf_dma) = mthdbuf_dma {
                                                                                                                        // Allocate the Ampere GPFIFO
                                                                                                                        // channel under the device using
                                                                                                                        // r535's four physical backing
                                                                                                                        // descriptors.
                                                                                                                        const AMPERE_CHANNEL_GPFIFO_A: u32 =
                                                                                                                            0x0000_c56f;
                                                                                                                        // R570 reserves CHID 0.  Match
                                                                                                                        // nouveau's NVKM_RM_CHAN(chid)
                                                                                                                        // handle and USERD index encoding.
                                                                                                                        const GPFIFO_CHID: u32 = 1;
                                                                                                                        const GPFIFO_HANDLE: u32 =
                                                                                                                            0xf1f0_0000 | GPFIFO_CHID;
                                                                                                                        const GPFIFO_FLAGS: u32 =
                                                                                                                            0x0020_0000
                                                                                                                                | 0x0000_0020
                                                                                                                                | ((GPFIFO_CHID & 7) << 8)
                                                                                                                                | ((GPFIFO_CHID / 8) << 12);
                                                                                                                        // RPC header is 0x20; NV_CHANNEL_ALLOC_PARAMS is 0x170.
                                                                                                                        const GPFIFO_RPC_SIZE: usize = 0x20 + 368;
                                                                                                                        let mut gpfifo_rpc =
                                                                                                                            [0u8; GPFIFO_RPC_SIZE];
                                                                                                                        gpfifo_rpc[0x00..0x04]
                                                                                                                            .copy_from_slice(
                                                                                                                                &CLIENT_HANDLE.to_le_bytes());
                                                                                                                        gpfifo_rpc[0x04..0x08]
                                                                                                                            .copy_from_slice(
                                                                                                                                &DEVICE_HANDLE.to_le_bytes());
                                                                                                                        gpfifo_rpc[0x08..0x0c]
                                                                                                                            .copy_from_slice(
                                                                                                                                &GPFIFO_HANDLE.to_le_bytes());
                                                                                                                        gpfifo_rpc[0x0c..0x10]
                                                                                                                            .copy_from_slice(
                                                                                                                                &AMPERE_CHANNEL_GPFIFO_A
                                                                                                                                    .to_le_bytes());
                                                                                                                        gpfifo_rpc[0x14..0x18]
                                                                                                                            .copy_from_slice(
                                                                                                                                &368u32.to_le_bytes());
                                                                                                                        // params.gpFifoOffset
                                                                                                                        gpfifo_rpc[0x28..0x30]
                                                                                                                            .copy_from_slice(
                                                                                                                                &map_va.to_le_bytes());
                                                                                                                        // params.gpFifoEntries
                                                                                                                        gpfifo_rpc[0x30..0x34]
                                                                                                                            .copy_from_slice(
                                                                                                                                &32u32.to_le_bytes());
                                                                                                                        // params.flags: PRIVILEGED physical channel,
                                                                                                                        // (bit5 set), fixed USERD page
                                                                                                                        // and CHID-derived USERD slot.
                                                                                                                        gpfifo_rpc[0x34..0x38]
                                                                                                                            .copy_from_slice(
                                                                                                                                &GPFIFO_FLAGS
                                                                                                                                    .to_le_bytes());
                                                                                                                        // params.hContextShare remains zero;
                                                                                                                        // r535 supplies only hVASpace here.
                                                                                                                        gpfifo_rpc[0x3c..0x40]
                                                                                                                            .copy_from_slice(
                                                                                                                                &VASPACE_HANDLE.to_le_bytes());
                                                                                                                        // params.engineType is the NV2080 engine ordinal
                                                                                                                        // translated from the discovered RM_ENGINE_TYPE_GR0
                                                                                                                        // table entry.  ENG_DESC and runlist ID are not part
                                                                                                                        // of NV_CHANNEL_ALLOC_PARAMS.
                                                                                                                        gpfifo_rpc[0xa0..0xa4]
                                                                                                                            .copy_from_slice(
                                                                                                                                &devinfo_gr_engine_type.to_le_bytes());

                                                                                                                        let instance_phys =
                                                                                                                            pdb_phys + 0x6000;
                                                                                                                        let userd_phys =
                                                                                                                            pdb_phys + 0x7000;
                                                                                                                        for (base, bytes) in [
                                                                                                                            (instance_phys, 0x1000u64),
                                                                                                                            (userd_phys, 0x200u64),
                                                                                                                        ] {
                                                                                                                            let ap = fb_aperture(base);
                                                                                                                            for dword in 0..bytes / 4 {
                                                                                                                                wr(ap + dword * 4, 0);
                                                                                                                            }
                                                                                                                        }
                                                                                                                        // R535/nouveau: pass a ZEROED instance block and let
                                                                                                                        // GSP-RM initialise instance/RAMFC/PDB itself during
                                                                                                                        // the channel alloc. (Removed the gv100_vmm_join CPU
                                                                                                                        // pre-write, which GSP rejected with INVALID_ARGUMENT.)
                                                                                                                        wr(window_reg, window_saved);
                                                                                                                        core::sync::atomic::fence(
                                                                                                                            core::sync::atomic::Ordering::SeqCst);

                                                                                                                        // instanceMem: 4 KiB VIDMEM, cached.
                                                                                                                        gpfifo_rpc[0xb0..0xb8]
                                                                                                                            .copy_from_slice(
                                                                                                                                &instance_phys.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xb8..0xc0]
                                                                                                                            .copy_from_slice(
                                                                                                                                &0x1000u64.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xc0..0xc4]
                                                                                                                            .copy_from_slice(
                                                                                                                                &2u32.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xc4..0xc8]
                                                                                                                            .copy_from_slice(
                                                                                                                                &1u32.to_le_bytes());
                                                                                                                        // userdMem: one 0x200-byte VIDMEM slot.
                                                                                                                        gpfifo_rpc[0xc8..0xd0]
                                                                                                                            .copy_from_slice(
                                                                                                                                &userd_phys.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xd0..0xd8]
                                                                                                                            .copy_from_slice(
                                                                                                                                &0x200u64.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xd8..0xdc]
                                                                                                                            .copy_from_slice(
                                                                                                                                &2u32.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xdc..0xe0]
                                                                                                                            .copy_from_slice(
                                                                                                                                &1u32.to_le_bytes());
                                                                                                                        // ramfcMem aliases the first 0x200
                                                                                                                        // bytes of instanceMem.
                                                                                                                        gpfifo_rpc[0xe0..0xe8]
                                                                                                                            .copy_from_slice(
                                                                                                                                &instance_phys.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xe8..0xf0]
                                                                                                                            .copy_from_slice(
                                                                                                                                &0x200u64.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xf0..0xf4]
                                                                                                                            .copy_from_slice(
                                                                                                                                &2u32.to_le_bytes());
                                                                                                                        gpfifo_rpc[0xf4..0xf8]
                                                                                                                            .copy_from_slice(
                                                                                                                                &1u32.to_le_bytes());
                                                                                                                        // mthdbufMem: coherent SYSMEM with
                                                                                                                        // the exact size reported by RM.
                                                                                                                        gpfifo_rpc[0xf8..0x100]
                                                                                                                            .copy_from_slice(
                                                                                                                                &mthdbuf_dma.bus.to_le_bytes());
                                                                                                                        gpfifo_rpc[0x100..0x108]
                                                                                                                            .copy_from_slice(
                                                                                                                                &(mthdbuf_size as u64)
                                                                                                                                    .to_le_bytes());
                                                                                                                        gpfifo_rpc[0x108..0x10c]
                                                                                                                            .copy_from_slice(
                                                                                                                                &1u32.to_le_bytes());
                                                                                                                        // hPhysChannelGroup @params0xf0 (RPC payload +0x110).
                                                                                                                        // an internal KEPLER_CHANNEL_GROUP_A (TSG) when parent=DEVICE
                                                                                                                        // (krnl_chan.c:388) and needs a client-unique handle for it;
                                                                                                                        // Retain the previous explicit handle for the isolated offset probe.
                                                                                                                        gpfifo_rpc[0x110..0x114]
                                                                                                                            .copy_from_slice(
                                                                                                                                &0xa06c_0001u32.to_le_bytes());
                                                                                                                        // internalFlags: USER privilege (0x14),
                                                                                                                        // no error or ECC notifier.
                                                                                                                        gpfifo_rpc[0x114..0x118]
                                                                                                                            .copy_from_slice(
                                                                                                                                &0x14u32.to_le_bytes());

                                                                                                                        let gpfifo_seq = 10u32;
                                                                                                                        match unsafe {
                                                                                                                            gsp::rpc_send_argc(
                                                                                                                                &handshake,
                                                                                                                                RPC_GSP_RM_ALLOC,
                                                                                                                                &gpfifo_rpc,
                                                                                                                                GPFIFO_RPC_SIZE,
                                                                                                                                gpfifo_seq)
                                                                                                                        } {
                                                                                                                            Some(gpfifo_csum) => {
                                                                                                                                let gpfifo_cmd_w = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        cmdq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let gpfifo_cmd_r_before = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(
                                                                                                                                            msg_rxoff as usize)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let gpfifo_msg_w_before = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                gsp::rpc_doorbell(
                                                                                                                                    gsp_falcon, &wr);
                                                                                                                                let gpfifo_doorbell =
                                                                                                                                    rd(gsp_falcon + 0xc00);
                                                                                                                                log.push(format!(
                                                                                                                                    "  M3 GPFIFO RPC sent: fn={} seq={} client=0x{:08x} parent_device=0x{:08x} channel=0x{:08x} chid={} class=0x{:08x} vas=0x{:08x} flags=0x{:08x} engine=0x{:08x}(table GR0) runlist={} eng_desc=0x{:08x} gpfifo_va=0x{:x} entries=32 instance=0x{:x}/0x1000 userd=0x{:x}/0x200 ramfc=0x{:x}/0x200 mthdbuf=0x{:x}/0x{:x} params=368 argc={} checksum=0x{:08x} doorbell_readback=0x{:08x}",
                                                                                                                                    RPC_GSP_RM_ALLOC,
                                                                                                                                    gpfifo_seq,
                                                                                                                                    CLIENT_HANDLE,
                                                                                                                                    DEVICE_HANDLE,
                                                                                                                                    GPFIFO_HANDLE,
                                                                                                                                    GPFIFO_CHID,
                                                                                                                                    AMPERE_CHANNEL_GPFIFO_A,
                                                                                                                                    VASPACE_HANDLE,
                                                                                                                                    GPFIFO_FLAGS,
                                                                                                                                    devinfo_gr_engine_type,
                                                                                                                                    devinfo_gr[1],
                                                                                                                                    devinfo_gr[0],
                                                                                                                                    map_va,
                                                                                                                                    instance_phys,
                                                                                                                                    userd_phys,
                                                                                                                                    instance_phys,
                                                                                                                                    mthdbuf_dma.bus,
                                                                                                                                    mthdbuf_size,
                                                                                                                                    GPFIFO_RPC_SIZE,
                                                                                                                                    gpfifo_csum,
                                                                                                                                    gpfifo_doorbell));

                                                                                                                                let mut gpfifo_matched = false;
                                                                                                                                let mut gpfifo_reply_seq =
                                                                                                                                    u32::MAX;
                                                                                                                                let mut gpfifo_reply_fn = 0u32;
                                                                                                                                let mut gpfifo_result =
                                                                                                                                    u32::MAX;
                                                                                                                                let mut gpfifo_status =
                                                                                                                                    u32::MAX;
                                                                                                                                let mut gpfifo_cid =
                                                                                                                                    u32::MAX;
                                                                                                                                let mut gpfifo_reply_len = 0u32;
                                                                                                                                let mut gpfifo_desc =
                                                                                                                                    [(0u64, 0u64, 0u32); 4];
                                                                                                                                for _ in 0..20000 {
                                                                                                                                    let wptr = unsafe {
                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                            msgq.add(0x10)
                                                                                                                                                as *const u32)
                                                                                                                                    };
                                                                                                                                    let rptr = unsafe {
                                                                                                                                        core::ptr::read_volatile(
                                                                                                                                            cmdq.add(0x20)
                                                                                                                                                as *const u32)
                                                                                                                                    };
                                                                                                                                    if wptr != rptr {
                                                                                                                                        let elem = unsafe {
                                                                                                                                            msgq.add(
                                                                                                                                                tx_entryoff as usize
                                                                                                                                                    + rptr as usize
                                                                                                                                                        * 0x1000)
                                                                                                                                        };
                                                                                                                                        gpfifo_reply_seq = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x24)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        gpfifo_reply_len = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x38)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        gpfifo_reply_fn = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x3c)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        gpfifo_result = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x40)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        if gpfifo_reply_fn
                                                                                                                                            == RPC_GSP_RM_ALLOC
                                                                                                                                        {
                                                                                                                                            gpfifo_status = unsafe {
                                                                                                                                                core::ptr::read_volatile(
                                                                                                                                                    elem.add(0x60)
                                                                                                                                                        as *const u32)
                                                                                                                                            };
                                                                                                                                            gpfifo_cid = unsafe {
                                                                                                                                                core::ptr::read_volatile(
                                                                                                                                                    elem.add(0xf4)
                                                                                                                                                        as *const u32)
                                                                                                                                            };
                                                                                                                                            let desc_off =
                                                                                                                                                [0x100usize, 0x118, 0x130, 0x148];
                                                                                                                                            for i in 0..4usize {
                                                                                                                                                gpfifo_desc[i].0 = unsafe {
                                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                                        elem.add(desc_off[i])
                                                                                                                                                            as *const u64)
                                                                                                                                                };
                                                                                                                                                gpfifo_desc[i].1 = unsafe {
                                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                                        elem.add(desc_off[i] + 8)
                                                                                                                                                            as *const u64)
                                                                                                                                                };
                                                                                                                                                gpfifo_desc[i].2 = unsafe {
                                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                                        elem.add(desc_off[i] + 16)
                                                                                                                                                            as *const u32)
                                                                                                                                                };
                                                                                                                                            }
                                                                                                                                        }
                                                                                                                                        let ec = unsafe {
                                                                                                                                            core::ptr::read_volatile(
                                                                                                                                                elem.add(0x28)
                                                                                                                                                    as *const u32)
                                                                                                                                        };
                                                                                                                                        let step =
                                                                                                                                            if ec == 0 { 1 } else { ec };
                                                                                                                                        let mut next = rptr + step;
                                                                                                                                        if next >= tx_count {
                                                                                                                                            next -= tx_count;
                                                                                                                                        }
                                                                                                                                        core::sync::atomic::fence(
                                                                                                                                            core::sync::atomic::Ordering::SeqCst);
                                                                                                                                        unsafe {
                                                                                                                                            core::ptr::write_volatile(
                                                                                                                                                cmdq.add(0x20)
                                                                                                                                                    as *mut u32,
                                                                                                                                                next);
                                                                                                                                        }
                                                                                                                                        if gpfifo_reply_fn
                                                                                                                                            == RPC_GSP_RM_ALLOC
                                                                                                                                        {
                                                                                                                                            gpfifo_matched = true;
                                                                                                                                            break;
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                    boot::stall(100);
                                                                                                                                }
                                                                                                                                let gpfifo_cmd_r_final = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(
                                                                                                                                            msg_rxoff as usize)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                let gpfifo_msg_w_final = unsafe {
                                                                                                                                    core::ptr::read_volatile(
                                                                                                                                        msgq.add(0x10)
                                                                                                                                            as *const u32)
                                                                                                                                };
                                                                                                                                log.push(format!(
                                                                                                                                    "  M3 GPFIFO rings: cmd_w={} cmd_r={}->{} msg_w={}->{}",
                                                                                                                                    gpfifo_cmd_w,
                                                                                                                                    gpfifo_cmd_r_before,
                                                                                                                                    gpfifo_cmd_r_final,
                                                                                                                                    gpfifo_msg_w_before,
                                                                                                                                    gpfifo_msg_w_final));
                                                                                                                                log.push(format!(
                                                                                                                                    "  *** M3 GPFIFO reply: matched={} seq={} fn={} rpc_result=0x{:08x} alloc_status=0x{:08x} len=0x{:x} cid={} ***",
                                                                                                                                    gpfifo_matched,
                                                                                                                                    gpfifo_reply_seq,
                                                                                                                                    gpfifo_reply_fn,
                                                                                                                                    gpfifo_result,
                                                                                                                                    gpfifo_status,
                                                                                                                                    gpfifo_reply_len,
                                                                                                                                    gpfifo_cid));
                                                                                                                                log.push(format!(
                                                                                                                                    "  M3 GPFIFO memory: instance=[0x{:x},0x{:x},as={}] userd=[0x{:x},0x{:x},as={}] ramfc=[0x{:x},0x{:x},as={}] mthdbuf=[0x{:x},0x{:x},as={}]",
                                                                                                                                    gpfifo_desc[0].0,
                                                                                                                                    gpfifo_desc[0].1,
                                                                                                                                    gpfifo_desc[0].2,
                                                                                                                                    gpfifo_desc[1].0,
                                                                                                                                    gpfifo_desc[1].1,
                                                                                                                                    gpfifo_desc[1].2,
                                                                                                                                    gpfifo_desc[2].0,
                                                                                                                                    gpfifo_desc[2].1,
                                                                                                                                    gpfifo_desc[2].2,
                                                                                                                                    gpfifo_desc[3].0,
                                                                                                                                    gpfifo_desc[3].1,
                                                                                                                                    gpfifo_desc[3].2));
                                                                                                                                // ---- M3 LOGRM raw dump (offline decode source) ----
                                                                                                                                {
                                                                                                                                    let lr_host = handshake.logrm.host;
                                                                                                                                    let lr_size = handshake.logrm.size;
                                                                                                                                    if !lr_host.is_null() && lr_size >= 16 {
                                                                                                                                        let words = lr_size / 8;
                                                                                                                                        let put = unsafe { core::ptr::read_volatile(lr_host as *const u64) };
                                                                                                                                        let log_entries = words as u64 - 1;
                                                                                                                                        log.push(format!("  LOGRM: bus=0x{:x} size=0x{:x} put=0x{:x} log_entries=0x{:x}", handshake.logrm.bus, lr_size, put, log_entries));
                                                                                                                                        // dump a window of entries ending at PUT (backward), 160 entries
                                                                                                                                        let win: u64 = 160;
                                                                                                                                        let start = if put > win { put - win } else { 0 };
                                                                                                                                        let mut idx = start;
                                                                                                                                        while idx < put {
                                                                                                                                            // entry data lives at physicLogBuffer[1 + (idx % log_entries)]
                                                                                                                                            let slot = 1 + (idx % log_entries) as usize;
                                                                                                                                            let v = unsafe { core::ptr::read_volatile((lr_host as *const u64).add(slot)) };
                                                                                                                                            log.push(format!("  LOGRM_RAW idx={} v=0x{:016x}", idx, v));
                                                                                                                                            idx += 1;
                                                                                                                                        }
                                                                                                                                // ---- M3 LOGRM segmented record decode ----
                                                                                                                                {
                                                                                                                                    let lr_host = handshake.logrm.host;
                                                                                                                                    let lr_size = handshake.logrm.size;
                                                                                                                                    if !lr_host.is_null() && lr_size >= 16 {
                                                                                                                                        let words = (lr_size / 8) as u64;
                                                                                                                                        let log_entries = words - 1;
                                                                                                                                        let put = unsafe { core::ptr::read_volatile(lr_host as *const u64) };
                                                                                                                                        let win: u64 = 160;
                                                                                                                                        let start = if put > win { put - win } else { 0 };
                                                                                                                                        let get_ent = |idx: u64| -> u64 {
                                                                                                                                            let slot = 1 + (idx % log_entries) as usize;
                                                                                                                                            unsafe { core::ptr::read_volatile((lr_host as *const u64).add(slot)) }
                                                                                                                                        };
                                                                                                                                        let is_meta = |v: u64| v >= 0x20200000 && v <= 0x20400000;
                                                                                                                                        let mut prev_end = start;
                                                                                                                                        let mut pos = start;
                                                                                                                                        while pos < put {
                                                                                                                                            let v = get_ent(pos);
                                                                                                                                            if is_meta(v) && pos + 1 < put {
                                                                                                                                                let ts = get_ent(pos + 1);
                                                                                                                                                // args = entries in [prev_end, pos)
                                                                                                                                                let mut argc = 0u64;
                                                                                                                                                let mut a: [u64; 8] = [0; 8];
                                                                                                                                                let mut k = prev_end;
                                                                                                                                                while k < pos && (argc as usize) < 8 {
                                                                                                                                                    a[argc as usize] = get_ent(k);
                                                                                                                                                    argc += 1; k += 1;
                                                                                                                                                }
                                                                                                                                               let mut note = "";
                                                                                                                                                if v == 0x2032c2d0 {
                                                                                                                                                    log.push(String::from("  LOGRM_XREF metaVA=0x2032c2d0 emitterPC=0x01ab96c0 branchPC=0x01ab8742: hContextShare!=0 && hVASpace!=0 (source-correlated; separate format ELF unavailable)"));
                                                                                                                                                }
                                                                                                                                                let mut has_c56f = false; let mut has_1f = false; let mut has_bad = false;
                                                                                                                                                for i in 0..(argc as usize) {
                                                                                                                                                    if a[i] == 0xc56f { has_c56f = true; }
                                                                                                                                                    if a[i] == 0x1f { has_1f = true; }
                                                                                                                                                    if (a[i] & 0xffff0000) == 0xbadf0000 || a[i] == 0xbadfffff { has_bad = true; }
                                                                                                                                                }
                                                                                                                                                if has_c56f && has_1f { note = " [CHANNEL_CLASS+INVALID_ARG]"; }
                                                                                                                                                else if has_bad { note = " [BADF-POISON]"; }
                                                                                                                                                else if has_1f { note = " [status=0x1f]"; }
                                                                                                                                                log.push(format!(
                                                                                                                                                    "  LOGRM_REC metaVA=0x{:08x} ts=0x{:08x} argc={} a=[{:x},{:x},{:x},{:x},{:x},{:x}]{}",
                                                                                                                                                    v, ts, argc, a[0],a[1],a[2],a[3],a[4],a[5], note));
                                                                                                                                                prev_end = pos + 2;
                                                                                                                                                pos += 2;
                                                                                                                                            } else {
                                                                                                                                                pos += 1;
                                                                                                                                            }
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                }
                                                                                                                                    } else {
                                                                                                                                        log.push(String::from("  LOGRM: buffer unavailable"));
                                                                                                                                    }
                                                                                                                                }
                                                                                                                                if gpfifo_matched
                                                                                                                                    && gpfifo_result == 0
                                                                                                                                    && gpfifo_status == 0
                                                                                                                                {
                                                                                                                                    log.push(format!(
                                                                                                                                        "  *** M3 GPFIFO SUCCESS: AMPERE_CHANNEL_GPFIFO_A allocated handle=0x{:08x} cid={} parent=DEVICE (no TSG) ***",
                                                                                                                                        GPFIFO_HANDLE,
                                                                                                                                        gpfifo_cid));
                                                                                                                                    m3::run(&handshake, &rd, &wr, bar0, gsp_falcon,
                                                                                                                                        CLIENT_HANDLE, SUBDEVICE_HANDLE, GPFIFO_HANDLE, devinfo_gr_engine_type,
                                                                                                                                        data_phys, userd_phys, map_va, &mut dma,
                                                                                                                                        (static_fb_regions[1], static_fb_flags[1][3]), &mut log);
                                                                                                                                }
                                                                                                                            }
                                                                                                                            None => log.push(String::from(
                                                                                                                                "  M3 GPFIFO RPC push FAILED (cmdq full?)")),
                                                                                                                        }
                                                                                                                        }
                                                                                                        }
                                                                                                    } else {
                                                                                                        log.push(String::from(
                                                                                                            "  M3 VAMAP skipped: table chain does not fit usable FB region"));
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                            None => log.push(String::from(
                                                                                                "  M3 SET_PDB RPC push FAILED (cmdq full?)")),
                                                                                        }
                                                                                    } else {
                                                                                        log.push(String::from(
                                                                                            "  M3 SET_PDB skipped: no usable aligned VRAM page"));
                                                                                    }
                                                                                }
                                                                            }
                                                                            None => log.push(String::from(
                                                                                "  M3 VASPACE RPC push FAILED (cmdq full?)")),
                                                                        }
                                                                    }
                                                                }
                                                                None => log.push(String::from(
                                                                    "  M3 SUBDEVICE RPC push FAILED (cmdq full?)")),
                                                            }
                                                        }
                                                    }
                                                    None => log.push(String::from(
                                                        "  M3 DEVICE RPC push FAILED (cmdq full?)")),
                                                }
                                            }
                                        }
                                        None => log.push(String::from(
                                            "  M3 CLIENT RPC push FAILED (cmdq full?)")),
                                    }
                                }
                            }
                            None => log.push(String::from("  M2 RPC push FAILED (cmdq full?)")),
                        }
                    }
                    for (i, trace) in message_trace[..message_trace_count].iter().enumerate() {
                        log.push(format!("  GSP MSG[{}]: elem={} outer_seq={} len=0x{:x} fn=0x{:08x} result=0x{:08x} private=0x{:08x} rpc_seq={} data=[0x{:08x},0x{:08x}]",
                            i, trace[0], trace[1], trace[2], trace[3], trace[4],
                            trace[5], trace[6], trace[7], trace[8]));
                        if trace[3] == 0x1020 {
                            log.push(format!("  GSP NOCAT[{}]: rec={} bug=0x{:08x} subsystem=0x{:08x} error=0x{:08x}{:08x} tdr=0x{:08x} diag_len={} diag0=0x{:08x}",
                                i, trace[9], trace[10], trace[11], trace[13],
                                trace[12], trace[14], trace[15], trace[16]));
                            log.push(format!("  GSP NOCAT TEXT[{}]: source=\"{}\" engine=\"{}\" diag={}",
                                i,
                                diagnostic_ascii(&message_text[i], 65),
                                diagnostic_ascii(&message_aux_text[i], 65),
                                diagnostic_hex(&message_diag[i], trace[15] as usize)));
                        } else if trace[3] == 0x100c {
                            log.push(format!("  GSP LIBOS[{}]: eng=0x{:08x} size={} text=\"{}\" hex={}",
                                i, trace[7], trace[8],
                                diagnostic_ascii(&message_text[i], trace[8] as usize),
                                diagnostic_hex(&message_text[i], trace[8] as usize)));
                        }
                    }
                    log.push(format!("  GSP SEQ: events={} status={} buffer_dw={} cmd_index={} commands={} last_op={} last_addr=0x{:08x} poll_error={} core_ops=0x{:x}",
                        seq_events, seq_status, seq_buffer_dwords, seq_cmd_index,
                        seq_commands, seq_last_op, seq_last_addr, seq_poll_error, seq_core_ops));
                    log.push(format!("  GSP SEQ poll: mask=0x{:08x} want=0x{:08x} observed=0x{:08x}",
                        seq_poll_mask, seq_poll_value, seq_poll_observed));
                    log.push(format!("  GSP RESUME: flags=0x{:02x} scratch=0x{:08x}->0x{:08x} SEC2_CPUCTL=0x{:08x}->0x{:08x} SEC2_MBOX=[0x{:08x},0x{:08x}]",
                        resume_flags, resume_scratch_before, resume_scratch_after,
                        resume_sec_cpuctl_before, resume_sec_cpuctl_after,
                        resume_sec_mbox0, resume_sec_mbox1));
                    log.push(format!("  GSP RESUME core: CPUCTL=0x{:08x} BCR=0x{:08x} (flags: scrub,bcr,reload,sec0,active)",
                        resume_gsp_cpuctl, resume_gsp_bcr));
                    log.push(format!("  *** GSP INIT DONE: {} ***", init_done));
                } else {
                    log.push(String::from("  BOOTER parse failed"));
                }
            }
        }}}
    }
    dump(&log);
    // Headless auto-return: this stick is booted one-shot (efibootmgr BootNext)
    // on a machine with no monitor and no operator, so the app must hand the
    // box back by itself.  The result is already flushed to \GPURESULT.TXT by
    // dump(); wait ~20s (so a watching serial console can be read, and to leave
    // margin for the FAT write to settle) then warm-reset back into the normal
    // boot order, i.e. straight into Proxmox.
    system::with_stdout(|out| {
        let msg = "\r\nDONE. Auto-rebooting in 20s (back to normal boot order)...\r\n";
        let mut b: Vec<u16> = msg.encode_utf16().collect(); b.push(0);
        if let Ok(c)=uefi::CStr16::from_u16_with_nul(&b) { let _=out.output_string(c); }
    });
    for _ in 0..20 { boot::stall(1_000_000); }
    uefi::runtime::reset(uefi::runtime::ResetType::WARM, Status::SUCCESS, None);
}
fn dump(log:&[String]) {
    system::with_stdout(|out| { for line in log {
        let mut b: Vec<u16> = line.encode_utf16().collect(); b.push(13); b.push(10); b.push(0);
        if let Ok(c)=uefi::CStr16::from_u16_with_nul(&b) { let _=out.output_string(c); }
    }});
    // The target machine is HEADLESS (no monitor), so the screen dump above is
    // not readable by the operator.  Persist the same log to the boot volume as
    // \GPURESULT.TXT: pull the USB stick afterwards and read it anywhere.  This
    // needs no network and no EDK2 NetworkPkg, so it works on any UEFI board.
    let written = write_result_file(log);
    system::with_stdout(|out| {
        let msg = match &written {
            Ok(n)  => format!("RESULT FILE: wrote {} bytes to \\GPURESULT.TXT", n),
            Err(e) => format!("RESULT FILE: FAILED ({}) - screen output only", e),
        };
        let mut b: Vec<u16> = msg.encode_utf16().collect(); b.push(13); b.push(10); b.push(0);
        if let Ok(c)=uefi::CStr16::from_u16_with_nul(&b) { let _=out.output_string(c); }
    });
}

/// Write the whole log to \GPURESULT.TXT on the volume this image booted from.
/// Returns the byte count written.  Uses the loaded-image device handle so we
/// always land on the USB stick we came from, not some other attached disk.
fn write_result_file(log: &[String]) -> Result<usize, &'static str> {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
    use uefi::proto::media::fs::SimpleFileSystem;
    use uefi::proto::loaded_image::LoadedImage;

    let image = boot::image_handle();
    let li = boot::open_protocol_exclusive::<LoadedImage>(image)
        .map_err(|_| "no LoadedImage")?;
    let device = li.device().ok_or("no boot device")?;
    let mut fs = boot::open_protocol_exclusive::<SimpleFileSystem>(device)
        .map_err(|_| "no SimpleFileSystem on boot device")?;
    let mut root = fs.open_volume().map_err(|_| "open_volume failed")?;

    let name = uefi::cstr16!("GPURESULT.TXT");
    let handle = root
        .open(name, FileMode::CreateReadWrite, FileAttribute::empty())
        .map_err(|_| "create GPURESULT.TXT failed")?;
    let mut file = match handle.into_type().map_err(|_| "into_type failed")? {
        FileType::Regular(f) => f,
        FileType::Dir(_) => return Err("GPURESULT.TXT is a directory"),
    };

    // Plain ASCII/CRLF so it opens cleanly in Notepad on Windows.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"uefi-gpu bare-metal result log\r\n");
    buf.extend_from_slice(b"===============================\r\n");
    for line in log {
        for ch in line.chars() {
            buf.push(if (ch as u32) < 0x80 { ch as u8 } else { b'?' });
        }
        buf.extend_from_slice(b"\r\n");
    }
    buf.extend_from_slice(b"=== END OF LOG ===\r\n");
    file.write(&buf).map_err(|_| "write failed")?;
    file.flush().map_err(|_| "flush failed")?;
    file.close();
    Ok(buf.len())
}
