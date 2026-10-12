// gsp.rs — GSP-RM firmware staging for bare-metal UEFI GSP boot (M1.7+).
//
// Builds the structures GSP-RM boot needs in DMA-addressable sysmem:
//   * radix3 page table over the GSP-RM ELF (3-level, 4K pages)
//   * WprMeta (populated in a later step, offsets from r570_144 bindings)
//
// This module is algorithm-correct scaffolding; the exact WprMeta field byte
// offsets and FB-layout constants are wired in from the research report before
// the SEC2 Booter step. Everything here uses coherent DMA buffers so the GPU
// can read the tables (see DmaAlloc in main.rs).

extern crate alloc;
use alloc::vec::Vec;

/// GSP understands only 4K pages regardless of host page size.
pub const GSP_PAGE_SIZE: usize = 0x1000;
pub const GSP_PAGE_SHIFT: u32 = 12;

/// A coherent DMA region: CPU host pointer + GPU-visible bus address.
#[derive(Clone, Copy)]
pub struct DmaRegion {
    pub host: *mut u8,
    pub bus: u64,
    pub size: usize,
}

/// Trait the caller (main.rs) implements using EFI_PCI_ROOT_BRIDGE_IO
/// AllocateBuffer+Map, so this module stays independent of the protocol glue.
pub trait DmaAllocator {
    /// Allocate `pages` 4K pages of coherent, bus-addressable memory, zeroed.
    fn alloc_pages(&mut self, pages: usize) -> Option<DmaRegion>;
}

// Initial GSP-RM host/GSP handshake, matching r570's
// MESSAGE_QUEUE_INIT_ARGUMENTS and LibosMemoryRegionInitArgument layouts.
pub const GSP_QUEUE_SIZE: usize = 0x40000;
pub const GSP_QUEUE_ENTRY_OFF: usize = 0x1000;
pub const GSP_INIT_DONE: u32 = 0x1001;
const GSP_RPC_HEADER_SIZE: usize = 0x20;
const GSP_MSG_HEADER_SIZE: usize = 0x30;
const GSP_RPC_SIGNATURE: u32 = 0x4350_5256; // "CPRV"
const GSP_RPC_SET_SYSTEM_INFO: u32 = 72;
const GSP_RPC_SET_REGISTRY: u32 = 73;
const GSP_SYSTEM_INFO_SIZE: usize = 928;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsgqTxHeader {
    version: u32,
    size: u32,
    msg_size: u32,
    msg_count: u32,
    write_ptr: u32,
    flags: u32,
    rx_hdr_off: u32,
    entry_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MessageQueueInitArguments {
    shared_mem_phys_addr: u64,
    page_table_entry_count: u32,
    _padding0: [u8; 4],
    cmd_queue_offset: u64,
    stat_queue_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GspSrInitArguments {
    old_level: u32,
    flags: u32,
    in_pm_transition: u8,
    _padding0: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GspArgumentsCached {
    message_queue: MessageQueueInitArguments,
    sr: GspSrInitArguments,
    gpu_instance: u32,
    dmem_stack: u8,
    _padding0: [u8; 7],
    profiler_pa: u64,
    profiler_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LibosMemoryRegionInitArgument {
    id8: u64,
    pa: u64,
    size: u64,
    kind: u8,
    loc: u8,
    _padding0: [u8; 6],
}

/// Buffers that must remain alive until GSP-RM emits GspInitDone.
pub struct GspBootHandshake {
    pub shared: DmaRegion,
    pub libos: DmaRegion,
    pub rmargs: DmaRegion,
    pub loginit: DmaRegion,
    pub logintr: DmaRegion,
    pub logrm: DmaRegion,
    pub pte_count: u32,
    pub cmdq_offset: usize,
    pub msgq_offset: usize,
    pub msg_count: u32,
}

pub struct GspInitRpcState {
    pub write_ptr: u32,
    pub system_checksum: u32,
    pub registry_checksum: u32,
}

fn put_u32(buf: &mut [u8], off: usize, value: u32) {
    buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut [u8], off: usize, value: u64) {
    buf[off..off + 8].copy_from_slice(&value.to_le_bytes());
}

unsafe fn cmdq_push_rpc(
    handshake: &GspBootHandshake,
    function: u32,
    payload: &[u8],
    sequence: u32,
) -> Option<u32> {
    // argc defaults to the payload we physically carry.
    unsafe { cmdq_push_rpc_argc(handshake, function, payload, payload.len(), sequence) }
}

/// Push an RPC, declaring an `argc` (the request's argument size, which for a
/// read-style RPC is the REPLY struct size) that may exceed the bytes we copy.
///
/// This ports nouveau r535_gsp_cmdq_push exactly, including multi-page messages:
///   argc = ALIGN(GSP_MSG_HDR_SIZE + argc, PAGE);  elem_count = argc/PAGE;
///   checksum = xor over the WHOLE aligned span (checksum field zeroed first);
///   memcpy the message contiguously across consecutive ring slots, wrapping at
///   cnt; publish wptr; ring the 0xc00 doorbell (done by the caller).
///
/// # Safety
/// `handshake` queues must be mapped and owned by us.
unsafe fn cmdq_push_rpc_argc(
    handshake: &GspBootHandshake,
    function: u32,
    payload: &[u8],
    argc: usize,
    sequence: u32,
) -> Option<u32> {
    let cmdq = unsafe { handshake.shared.host.add(handshake.cmdq_offset) };

    // Nouveau r535_gsp_cmdq_push APPENDS at the current write pointer:
    //   wptr = *gsp->cmdq.wptr;  ... memcpy at wptr ... *gsp->cmdq.wptr = wptr+n;
    // It reads GSP's readPtr ONLY to compute free space (backpressure), and NEVER
    // rewinds wptr to it.  An earlier version here resync'd wptr to readPtr each
    // call, which during preboot (readPtr stuck at 0) made the 2nd preboot RPC
    // OVERWRITE the 1st (observed: wptr=1 after two pushes).  Append, don't resync.
    //
    // Queue wiring (upstream-gsp-r535.c:1699-1704): our writePtr = cmdq->tx.writePtr
    // (cmdq+0x10); GSP's readPtr for our commands = msgq->rx.readPtr (msgq + its
    // rxHdrOff) - used here for the free-space check only.
    let msgq = unsafe { handshake.shared.host.add(handshake.msgq_offset) };
    let write_ptr = unsafe { core::ptr::read_volatile(cmdq.add(0x10) as *const u32) };
    if write_ptr >= handshake.msg_count {
        return None;
    }

    // rpc->length = rpc header + argc  (nouveau r535_gsp_rpc_get).
    let rpc_length = GSP_RPC_HEADER_SIZE.checked_add(argc)?;
    if payload.len() > argc {
        return None;
    }
    // Whole message = msg envelope + rpc header + argc, rounded up to whole pages.
    let total = GSP_MSG_HEADER_SIZE + rpc_length;
    let aligned = (total + GSP_PAGE_SIZE - 1) & !(GSP_PAGE_SIZE - 1);
    let pages = aligned / GSP_PAGE_SIZE;
    if pages == 0 || pages as u32 >= handshake.msg_count {
        return None;
    }

    // Build the message in a contiguous scratch buffer first, then memcpy it into
    // the ring (which may wrap).  The ring is 0x40000 = 64 pages, big enough.
    let mut buf = Vec::<u8>::with_capacity(aligned);
    buf.resize(aligned, 0);
    {
        // msg envelope: checksum@0x20 (zeroed for now), sequence@0x24, elem_count@0x28.
        put_u32(&mut buf, 0x24, sequence);
        put_u32(&mut buf, 0x28, pages as u32);
        // nvfw_gsp_rpc v03.00 header at +0x30.
        let r = GSP_MSG_HEADER_SIZE;
        put_u32(&mut buf, r + 0x00, 0x0300_0000);
        put_u32(&mut buf, r + 0x04, GSP_RPC_SIGNATURE);
        put_u32(&mut buf, r + 0x08, rpc_length as u32);
        put_u32(&mut buf, r + 0x0c, function);
        put_u32(&mut buf, r + 0x10, u32::MAX);
        put_u32(&mut buf, r + 0x14, u32::MAX);
        let data_off = r + GSP_RPC_HEADER_SIZE;
        buf[data_off..data_off + payload.len()].copy_from_slice(payload);
    }
    // checksum = xor of every u64 in the aligned span, with checksum field 0.
    let mut xor = 0u64;
    for i in 0..(aligned / 8) {
        let mut b8 = [0u8; 8];
        b8.copy_from_slice(&buf[i * 8..i * 8 + 8]);
        xor ^= u64::from_le_bytes(b8);
    }
    let checksum = (xor as u32) ^ ((xor >> 32) as u32);
    put_u32(&mut buf, 0x20, checksum);

    // Copy contiguously across ring slots, wrapping at msg_count.
    let entries = unsafe { cmdq.add(GSP_QUEUE_ENTRY_OFF) };
    let mut off = 0usize;
    let mut slot = write_ptr;
    while off < aligned {
        let dst = unsafe { entries.add(slot as usize * GSP_PAGE_SIZE) };
        unsafe {
            core::ptr::copy_nonoverlapping(buf.as_ptr().add(off), dst, GSP_PAGE_SIZE);
        }
        off += GSP_PAGE_SIZE;
        slot += 1;
        if slot == handshake.msg_count {
            slot = 0;
        }
    }

    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    unsafe { core::ptr::write_volatile(cmdq.add(0x10) as *mut u32, slot); }
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    Some(checksum)
}

/// RPC function numbers we issue after GspInitDone.
/// (nova-r570-bindings.rs: NV_VGPU_MSG_FUNCTION_GET_GSP_STATIC_INFO = 65)
pub const RPC_GET_GSP_STATIC_INFO: u32 = 65;

/// Send one RPC on the command queue once GSP-RM is up.
///
/// This is the M2 entry point: the preboot handshake proved we can *stage*
/// messages GSP consumes during init; this proves we can drive the live
/// request/response protocol.  `sequence` must keep advancing across calls.
///
/// # Safety
/// `handshake` must describe queues that are still mapped and owned by us.
pub unsafe fn rpc_send(
    handshake: &GspBootHandshake,
    function: u32,
    payload: &[u8],
    sequence: u32,
) -> Option<u32> {
    unsafe { cmdq_push_rpc(handshake, function, payload, sequence) }
}

/// Like `rpc_send` but declares an argument size (the reply struct size) larger
/// than the bytes we physically carry - required for read-style RPCs such as
/// GET_GSP_STATIC_INFO where GSP fills `argc` bytes of reply.
///
/// # Safety
/// Same as `rpc_send`.
pub unsafe fn rpc_send_argc(
    handshake: &GspBootHandshake,
    function: u32,
    payload: &[u8],
    argc: usize,
    sequence: u32,
) -> Option<u32> {
    unsafe { cmdq_push_rpc_argc(handshake, function, payload, argc, sequence) }
}

/// Ring the GSP doorbell so it notices a freshly pushed command.
///
/// Verified against nouveau r535_gsp_cmdq_push, which after publishing the new
/// write pointer does exactly:
///     wmb(); (*gsp->cmdq.wptr) = wptr; mb();
///     nvkm_falcon_wr32(&gsp->falcon, 0xc00, 0x00000000);
/// i.e. a write of 0 to GSP falcon base + 0xc00.
pub fn rpc_doorbell<W: Fn(u64, u32)>(gsp_falcon_base: u64, w: &W) {
    w(gsp_falcon_base + 0xc00, 0);
}

/// Queue the two asynchronous RPCs Nouveau/OpenRM put on the command ring
/// before starting GSP-RM.  GSP consumes them during early init, before it can
/// emit a successful GspInitDone.
pub fn preload_init_rpcs(
    handshake: &GspBootHandshake,
    bar0: u64,
    bar1: u64,
    bar3: u64,
    pci_device_id: u16,
    pci_id: u32,
    pci_subsystem_id: u32,
    pci_revision_id: u32,
) -> Option<GspInitRpcState> {
    let mut system = [0u8; GSP_SYSTEM_INFO_SIZE];
    put_u64(&mut system, 0x00, bar0);
    put_u64(&mut system, 0x08, bar1);
    put_u64(&mut system, 0x10, bar3);
    put_u64(&mut system, 0x20, pci_device_id as u64);
    put_u64(&mut system, 0x48, 0x0000_8000_0000_0000); // x86-64 TASK_SIZE
    put_u32(&mut system, 0x50, 0x0008_8000);
    put_u32(&mut system, 0x54, 0x0000_1000);
    // r570's GspSystemInfo carries the same PCI identity dwords CPU-RM reads
    // from config space.  Do not leave them as an all-zero synthetic device.
    put_u32(&mut system, 0x58, pci_id);
    put_u32(&mut system, 0x5c, pci_subsystem_id);
    put_u32(&mut system, 0x60, pci_revision_id);
    // The probe boots this UEFI payload under QEMU/KVM with the physical GPU
    // attached through VFIO.  Match OpenRM's passthrough detection instead of
    // describing the platform as bare-host/Xen (the zero-initialized values).
    put_u32(&mut system, 0x344, 3); // OS_HYPERVISOR_KVM
    system[0x348] = 1;              // GspSystemInfo::bIsPassthru
    put_u64(&mut system, 0x398, GSP_PAGE_SIZE as u64); // hostPageSize

    let system_checksum = unsafe {
        cmdq_push_rpc(handshake, GSP_RPC_SET_SYSTEM_INFO, &system, 0)?
    };

    // PACKED_REGISTRY_TABLE with Nouveau's two required DWORD entries plus
    // the r570 firmware's supported headless mode.  The probe VM has no VGA
    // device, so do not initialize the display engine's unavailable DMA path.
    let mut registry = [0u8; 115];
    put_u32(&mut registry, 0x00, 115);
    put_u32(&mut registry, 0x04, 3);
    put_u32(&mut registry, 0x08, 56);
    registry[0x0c] = 1;
    put_u32(&mut registry, 0x10, 1);
    put_u32(&mut registry, 0x14, 4);
    put_u32(&mut registry, 0x18, 76);
    registry[0x1c] = 1;
    put_u32(&mut registry, 0x20, 1);
    put_u32(&mut registry, 0x24, 4);
    put_u32(&mut registry, 0x28, 98);
    registry[0x2c] = 1;
    put_u32(&mut registry, 0x30, 1);
    put_u32(&mut registry, 0x34, 4);
    registry[56..76].copy_from_slice(b"RMSecBusResetEnable\0");
    registry[76..98].copy_from_slice(b"RMForcePcieConfigSave\0");
    registry[98..115].copy_from_slice(b"RmDisableDisplay\0");
    let registry_checksum = unsafe {
        cmdq_push_rpc(handshake, GSP_RPC_SET_REGISTRY, &registry, 1)?
    };

    let cmdq = unsafe { handshake.shared.host.add(handshake.cmdq_offset) };
    let write_ptr = unsafe { core::ptr::read_volatile(cmdq.add(0x10) as *const u32) };
    Some(GspInitRpcState { write_ptr, system_checksum, registry_checksum })
}

fn libos_id8(name: &[u8]) -> u64 {
    let mut id = 0u64;
    for &c in name.iter().take(8) {
        id = (id << 8) | c as u64;
    }
    id
}

unsafe fn init_log_pte_array(region: DmaRegion) {
    // Word 0 is the producer put pointer.  The 4K page DMA array starts at
    // word 1, as in r535_gsp_libos_init/create_pte_array.
    let pages = (region.size + GSP_PAGE_SIZE - 1) / GSP_PAGE_SIZE;
    let p = region.host as *mut u64;
    for i in 0..pages {
        unsafe {
            core::ptr::write_volatile(p.add(1 + i), region.bus + (i * GSP_PAGE_SIZE) as u64);
        }
    }
}

/// Allocate and initialize the shared command/status rings, RMARGS, LIBOS
/// memory-region list, and early logging buffers consumed by GSP-RM at boot.
pub fn stage_boot_handshake<A: DmaAllocator>(alloc: &mut A) -> Option<GspBootHandshake> {
    let queue_pages = (2 * GSP_QUEUE_SIZE) / GSP_PAGE_SIZE;
    // The PTE list maps itself as well as both queues.  For 0x80000 of queues,
    // r570/r535 therefore use 128 + one PTE page = 129 entries/pages.
    let mut pte_count = queue_pages;
    pte_count += (pte_count * core::mem::size_of::<u64>() + GSP_PAGE_SIZE - 1) / GSP_PAGE_SIZE;
    let pte_bytes = (pte_count * core::mem::size_of::<u64>() + GSP_PAGE_SIZE - 1)
        & !(GSP_PAGE_SIZE - 1);
    let cmdq_offset = pte_bytes;
    let msgq_offset = cmdq_offset + GSP_QUEUE_SIZE;
    let shared_pages = pte_bytes / GSP_PAGE_SIZE + queue_pages;
    let shared = alloc.alloc_pages(shared_pages)?;
    let msg_count = ((GSP_QUEUE_SIZE - GSP_QUEUE_ENTRY_OFF) / GSP_PAGE_SIZE) as u32;

    unsafe {
        let ptes = shared.host as *mut u64;
        for i in 0..pte_count {
            core::ptr::write_volatile(ptes.add(i), shared.bus + (i * GSP_PAGE_SIZE) as u64);
        }
        core::ptr::write_volatile(
            shared.host.add(cmdq_offset) as *mut MsgqTxHeader,
            MsgqTxHeader {
                version: 0,
                size: GSP_QUEUE_SIZE as u32,
                msg_size: GSP_PAGE_SIZE as u32,
                msg_count,
                write_ptr: 0,
                flags: 1,
                rx_hdr_off: core::mem::size_of::<MsgqTxHeader>() as u32,
                entry_off: GSP_QUEUE_ENTRY_OFF as u32,
            },
        );
    }

    let rmargs = alloc.alloc_pages(1)?;
    unsafe {
        core::ptr::write_volatile(
            rmargs.host as *mut GspArgumentsCached,
            GspArgumentsCached {
                message_queue: MessageQueueInitArguments {
                    shared_mem_phys_addr: shared.bus,
                    page_table_entry_count: pte_count as u32,
                    _padding0: [0; 4],
                    cmd_queue_offset: cmdq_offset as u64,
                    stat_queue_offset: msgq_offset as u64,
                },
                dmem_stack: 1,
                ..GspArgumentsCached::default()
            },
        );
    }

    let loginit = alloc.alloc_pages(0x10)?;
    let logintr = alloc.alloc_pages(0x10)?;
    let logrm = alloc.alloc_pages(0x10)?;
    unsafe {
        init_log_pte_array(loginit);
        init_log_pte_array(logintr);
        init_log_pte_array(logrm);
    }

    let libos = alloc.alloc_pages(1)?;
    unsafe {
        let args = libos.host as *mut LibosMemoryRegionInitArgument;
        let entries = [
            (b"LOGINIT" as &[u8], loginit),
            (b"LOGINTR" as &[u8], logintr),
            (b"LOGRM" as &[u8], logrm),
            (b"RMARGS" as &[u8], rmargs),
        ];
        for (i, (name, region)) in entries.iter().enumerate() {
            core::ptr::write_volatile(args.add(i), LibosMemoryRegionInitArgument {
                id8: libos_id8(name),
                pa: region.bus,
                size: region.size as u64,
                kind: 1, // LIBOS_MEMORY_REGION_CONTIGUOUS
                loc: 1,  // LIBOS_MEMORY_REGION_LOC_SYSMEM
                _padding0: [0; 6],
            });
        }
    }

    Some(GspBootHandshake {
        shared, libos, rmargs, loginit, logintr, logrm,
        pte_count: pte_count as u32,
        cmdq_offset, msgq_offset, msg_count,
    })
}

const _: () = assert!(core::mem::size_of::<MsgqTxHeader>() == 0x20);
const _: () = assert!(core::mem::size_of::<MessageQueueInitArguments>() == 0x20);
const _: () = assert!(core::mem::size_of::<GspArgumentsCached>() == 0x48);
const _: () = assert!(core::mem::size_of::<LibosMemoryRegionInitArgument>() == 0x20);

/// A built radix3 page table. `level0.bus` is what goes into
/// WprMeta.sysmemAddrOfRadix3Elf (nova: radix3_dma_address() = level0 DMA addr).
///
/// Structure (verified from nova firmware/gsp.rs):
///   Level 0: 1 page, 1 entry  -> DMA addr of first Level 1 page
///   Level 1: entries          -> each = DMA addr of a Level 2 page
///   Level 2: entries          -> each = DMA addr of a firmware (ELF) page
///   Each entry = full 8-byte little-endian DMA byte address (NOT pfn>>12).
pub struct Radix3 {
    pub level0: DmaRegion,  // root (single page, 1 entry)
    pub level1: DmaRegion,
    pub level2: DmaRegion,  // points at the firmware pages
    pub data: DmaRegion,    // the ELF image, page-aligned in sysmem
    pub elf_size: usize,
}

/// Build the 3-level radix page table mapping the GSP-RM ELF image, matching
/// nova-core's `GspFirmware` construction. `elf` = raw gsp-570.144.bin bytes.
pub fn build_radix3<A: DmaAllocator>(alloc: &mut A, elf: &[u8]) -> Option<Radix3> {
    let elf_size = elf.len();
    let per_page = GSP_PAGE_SIZE / core::mem::size_of::<u64>(); // 512 entries/page

    // --- stage the ELF into page-aligned coherent memory (contiguous) ---
    let data_pages = (elf_size + GSP_PAGE_SIZE - 1) / GSP_PAGE_SIZE;
    let data = alloc.alloc_pages(data_pages)?;
    unsafe {
        core::ptr::copy_nonoverlapping(elf.as_ptr(), data.host, elf_size);
        let tail = data_pages * GSP_PAGE_SIZE - elf_size;
        if tail > 0 { core::ptr::write_bytes(data.host.add(elf_size), 0, tail); }
    }

    // --- Level 2: one entry per firmware page = full DMA byte address ---
    let l2_pages = (data_pages + per_page - 1) / per_page;
    let level2 = alloc.alloc_pages(l2_pages)?;
    unsafe {
        let p = level2.host as *mut u64;
        for i in 0..data_pages {
            let addr = data.bus + (i * GSP_PAGE_SIZE) as u64;
            core::ptr::write_volatile(p.add(i), addr.to_le());
        }
    }

    // --- Level 1: one entry per Level-2 page = its DMA byte address ---
    let l1_pages = (l2_pages + per_page - 1) / per_page;
    let level1 = alloc.alloc_pages(l1_pages)?;
    unsafe {
        let p = level1.host as *mut u64;
        for i in 0..l2_pages {
            let addr = level2.bus + (i * GSP_PAGE_SIZE) as u64;
            core::ptr::write_volatile(p.add(i), addr.to_le());
        }
    }

    // --- Level 0: single page, one entry = DMA addr of first Level-1 page ---
    let level0 = alloc.alloc_pages(1)?;
    unsafe {
        let p = level0.host as *mut u64;
        core::ptr::write_volatile(p, level1.bus.to_le());
    }

    Some(Radix3 { level0, level1, level2, data, elf_size })
}

/// Minimal ELF64 sanity check + entry point (for later BOOTVEC use).
pub fn elf64_entry(elf: &[u8]) -> Option<u64> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" || elf[4] != 2 {
        return None;
    }
    // e_entry at offset 24, 8 bytes LE
    let mut e = [0u8; 8];
    e.copy_from_slice(&elf[24..32]);
    Some(u64::from_le_bytes(e))
}

fn elf_u16(elf: &[u8], off: usize) -> Option<u16> {
    let b = elf.get(off..off.checked_add(2)?)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn elf_u32(elf: &[u8], off: usize) -> Option<u32> {
    let b = elf.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn elf_u64(elf: &[u8], off: usize) -> Option<u64> {
    let b = elf.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

/// Return the file range occupied by a named ELF64 section.  GSP-RM is a
/// relocatable ELF container; nouveau maps only `.fwimage` through radix3 and
/// stages `.fwsignature_ga10x` separately for the SEC2 booter.
pub fn elf64_section_range(elf: &[u8], wanted: &[u8]) -> Option<(usize, usize)> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return None;
    }
    let shoff = usize::try_from(elf_u64(elf, 0x28)?).ok()?;
    let shentsize = elf_u16(elf, 0x3a)? as usize;
    let shnum = elf_u16(elf, 0x3c)? as usize;
    let shstrndx = elf_u16(elf, 0x3e)? as usize;
    if shentsize < 64 || shnum == 0 || shstrndx >= shnum {
        return None;
    }
    let sht_bytes = shentsize.checked_mul(shnum)?;
    elf.get(shoff..shoff.checked_add(sht_bytes)?)?;

    let str_hdr = shoff.checked_add(shstrndx.checked_mul(shentsize)?)?;
    let str_off = usize::try_from(elf_u64(elf, str_hdr + 0x18)?).ok()?;
    let str_size = usize::try_from(elf_u64(elf, str_hdr + 0x20)?).ok()?;
    let strings = elf.get(str_off..str_off.checked_add(str_size)?)?;

    for index in 0..shnum {
        let hdr = shoff.checked_add(index.checked_mul(shentsize)?)?;
        let name_off = elf_u32(elf, hdr)? as usize;
        let tail = strings.get(name_off..)?;
        let name_len = tail.iter().position(|&b| b == 0)?;
        if &tail[..name_len] == wanted {
            let off = usize::try_from(elf_u64(elf, hdr + 0x18)?).ok()?;
            let size = usize::try_from(elf_u64(elf, hdr + 0x20)?).ok()?;
            elf.get(off..off.checked_add(size)?)?;
            return Some((off, size));
        }
    }
    None
}

/// Parsed payload descriptor from bootloader-570.144.bin.  The outer
/// nvfw_bin_hdr points at RM_RISCV_UCODE_DESC and at the 0x6000-byte payload
/// that nouveau copies into a coherent DMA buffer.
#[derive(Debug, Clone, Copy)]
pub struct RiscvBootloader {
    pub payload_offset: usize,
    pub payload_size: usize,
    pub code_offset: u32,
    pub data_offset: u32,
    pub manifest_offset: u32,
    pub app_version: u32,
}

pub fn parse_riscv_bootloader(blob: &[u8]) -> Option<RiscvBootloader> {
    if blob.len() < 24 || elf_u32(blob, 0)? != 0x10de {
        return None;
    }
    let desc = elf_u32(blob, 0x0c)? as usize;
    let payload_offset = elf_u32(blob, 0x10)? as usize;
    let payload_size = elf_u32(blob, 0x14)? as usize;
    // The fields used by nouveau are common to descriptor versions 1..5.
    let app_version = elf_u32(blob, desc.checked_add(0x1c)?)?;
    let manifest_offset = elf_u32(blob, desc.checked_add(0x20)?)?;
    let data_offset = elf_u32(blob, desc.checked_add(0x28)?)?;
    let data_size = elf_u32(blob, desc.checked_add(0x2c)?)? as usize;
    let code_offset = elf_u32(blob, desc.checked_add(0x30)?)?;
    let code_size = elf_u32(blob, desc.checked_add(0x34)?)? as usize;
    let payload = blob.get(payload_offset..payload_offset.checked_add(payload_size)?)?;
    payload.get(manifest_offset as usize..)?;
    payload.get(data_offset as usize..(data_offset as usize).checked_add(data_size)?)?;
    payload.get(code_offset as usize..(code_offset as usize).checked_add(code_size)?)?;
    Some(RiscvBootloader {
        payload_offset,
        payload_size,
        code_offset,
        data_offset,
        manifest_offset,
        app_version,
    })
}

pub const GSP_FW_WPR_META_MAGIC: u64 = 0xDC3AAE21371A60B3;
pub const GSP_FW_WPR_META_REVISION: u64 = 1;

/// GspFwWprMeta, r570.144 layout (256 bytes / 0x100), verified against the
/// matching OpenRM header and its `ct_assert(sizeof(*pWprMeta) == 256)`.
/// Unions are flattened to their zero-initialized storage;
/// the alternate arms are zeroed anyway. #[repr(C)] preserves exact offsets.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GspFwWprMeta {
    pub magic: u64,                        // 0x00
    pub revision: u64,                     // 0x08
    pub sysmem_addr_of_radix3_elf: u64,    // 0x10
    pub size_of_radix3_elf: u64,           // 0x18
    pub sysmem_addr_of_bootloader: u64,    // 0x20
    pub size_of_bootloader: u64,           // 0x28
    pub bootloader_code_offset: u64,       // 0x30
    pub bootloader_data_offset: u64,       // 0x38
    pub bootloader_manifest_offset: u64,   // 0x40
    pub sysmem_addr_of_signature: u64,     // 0x48 (union arm 1)
    pub size_of_signature: u64,            // 0x50
    pub gsp_fw_rsvd_start: u64,            // 0x58
    pub non_wpr_heap_offset: u64,          // 0x60
    pub non_wpr_heap_size: u64,            // 0x68
    pub gsp_fw_wpr_start: u64,             // 0x70
    pub gsp_fw_heap_offset: u64,           // 0x78
    pub gsp_fw_heap_size: u64,             // 0x80
    pub gsp_fw_offset: u64,                // 0x88
    pub boot_bin_offset: u64,              // 0x90
    pub frts_offset: u64,                  // 0x98
    pub frts_size: u64,                    // 0xa0
    pub gsp_fw_wpr_end: u64,               // 0xa8
    pub fb_size: u64,                      // 0xb0
    pub vga_workspace_offset: u64,         // 0xb8
    pub vga_workspace_size: u64,           // 0xc0
    pub boot_count: u64,                   // 0xc8
    pub partition_rpc_union: [u8; 32],     // 0xd0 (union, zeroed)
    pub gsp_fw_heap_vf_partition_count: u8,// 0xf0
    pub flags: u8,                         // 0xf1
    pub padding: [u8; 2],                  // 0xf2
    pub pmu_reserved_size: u32,            // 0xf4
    pub verified: u64,                     // 0xf8
}                                          // total 0x100 = 256

impl Default for GspFwWprMeta {
    fn default() -> Self {
        // SAFETY: all-zero is a valid bit pattern for this POD struct.
        unsafe { core::mem::zeroed() }
    }
}

/// r570.144 OpenRM has a hard compile-time assertion that this is 256 bytes.
const _: () = assert!(core::mem::size_of::<GspFwWprMeta>() == 0x100);

// FB layout constants (GA102/GA104 non-vGPU), from nova fb/hal/{ga102,tu102}.rs.
const SZ_1M: u64 = 1 << 20;
const SZ_128K: u64 = 128 * 1024;
const FRTS_SIZE: u64 = SZ_1M;              // frts_size_tu102
const NON_WPR_HEAP_SIZE: u64 = SZ_1M;      // non_wpr_heap_size_tu102
const VGA_WORKSPACE_SIZE: u64 = SZ_128K;   // 128 KiB at top of FB
const WPR_META_SIZE: u64 = 0x100;          // sizeof(GspFwWprMeta)

const fn align_down(v: u64, a: u64) -> u64 { v & !(a - 1) }

/// Compute gspFwHeapSize per nova LibosParams::wpr_heap_size (LIBOS3 baremetal).
fn gsp_fw_heap_size(fb_size: u64) -> u64 {
    let carveout: u64 = 22 * SZ_1M;            // LIBOS3 baremetal carveout (~22MiB)
    let base_rm: u64 = 8 * SZ_1M;              // base_rm TU10X
    let client_alloc: u64 = 96 * SZ_1M;        // client alloc, rounded to 1MiB
    let per_gb: u64 = 96 * 1024;               // 98304 B per GiB of FB
    let gib = (fb_size + (1 << 30) - 1) >> 30; // ceil(fb/1GiB)
    let overhead = {
        let o = gib * per_gb;
        (o + SZ_1M - 1) & !(SZ_1M - 1)         // round up to 1MiB
    };
    let sum = carveout + base_rm + client_alloc + overhead;
    // clamp to [88MiB, 280MiB)
    let min = 88 * SZ_1M;
    let max = 280 * SZ_1M;
    if sum < min { min } else if sum >= max { max - SZ_1M } else { sum }
}

/// The computed FB region layout (byte offsets within the framebuffer).
#[derive(Debug, Clone, Copy, Default)]
pub struct FbLayout {
    pub fb_size: u64,
    pub vga_off: u64, pub vga_size: u64,
    pub frts_off: u64, pub frts_size: u64,
    pub boot_off: u64,
    pub elf_off: u64,
    pub wpr2_heap_off: u64, pub wpr2_heap_size: u64,
    pub wpr2_start: u64, pub wpr2_end: u64,
    pub non_wpr_off: u64, pub non_wpr_size: u64,
}

/// Compute the top-down FB layout (nova fb.rs) given fb_size, the GSP ELF size,
/// and the bootloader size.
pub fn compute_fb_layout(fb_size: u64, elf_size: u64, bootloader_size: u64) -> FbLayout {
    compute_fb_layout_with_vga(fb_size, elf_size, bootloader_size, fb_size - VGA_WORKSPACE_SIZE)
}

/// Compute the layout with the VGA workspace base read from hardware.  The
/// common GA10x case reserves 128 KiB, while fused/display-disabled boards may
/// reserve the full final 1 MiB PRAMIN window.
pub fn compute_fb_layout_with_vga(
    fb_size: u64,
    elf_size: u64,
    bootloader_size: u64,
    vga_off: u64,
) -> FbLayout {
    let heap = gsp_fw_heap_size(fb_size);
    let fb_end = fb_size;
    let vga_off = core::cmp::min(vga_off, fb_end);
    let vga_size = fb_end - vga_off;
    let frts_end = align_down(vga_off, SZ_128K);
    let frts_off = frts_end - FRTS_SIZE;
    let boot_off = align_down(frts_off - bootloader_size, 0x1000);
    // Nouveau's TU102/GA102 GSP path aligns the `.fwimage` carveout to 64 KiB.
    let elf_off = align_down(boot_off - elf_size, 0x10000);
    let wpr2_heap_off = align_down(elf_off - heap, SZ_1M);
    let wpr2_heap_end = align_down(elf_off, SZ_1M);
    let wpr2_start = align_down(wpr2_heap_off - WPR_META_SIZE, SZ_1M);
    let non_wpr_off = wpr2_start - NON_WPR_HEAP_SIZE;
    FbLayout {
        fb_size,
        vga_off, vga_size,
        frts_off, frts_size: FRTS_SIZE,
        boot_off,
        elf_off,
        wpr2_heap_off, wpr2_heap_size: wpr2_heap_end - wpr2_heap_off,
        wpr2_start, wpr2_end: frts_end,
        non_wpr_off, non_wpr_size: NON_WPR_HEAP_SIZE,
    }
}

/// Populate a GspFwWprMeta from the FB layout + DMA addresses of the staged
/// radix3 ELF, bootloader, and signature.
pub fn make_wpr_meta(
    fb: &FbLayout,
    radix3_dma: u64, radix3_size: u64,
    bootloader_dma: u64, bootloader_size: u64,
    bl_code_off: u64, bl_data_off: u64, bl_manifest_off: u64,
    signature_dma: u64, signature_size: u64,
) -> GspFwWprMeta {
    let mut m = GspFwWprMeta::default();
    m.magic = GSP_FW_WPR_META_MAGIC;
    m.revision = GSP_FW_WPR_META_REVISION;
    m.sysmem_addr_of_radix3_elf = radix3_dma;
    m.size_of_radix3_elf = radix3_size;
    m.sysmem_addr_of_bootloader = bootloader_dma;
    m.size_of_bootloader = bootloader_size;
    m.bootloader_code_offset = bl_code_off;
    m.bootloader_data_offset = bl_data_off;
    m.bootloader_manifest_offset = bl_manifest_off;
    m.sysmem_addr_of_signature = signature_dma;
    m.size_of_signature = signature_size;
    m.gsp_fw_rsvd_start = fb.non_wpr_off;
    m.non_wpr_heap_offset = fb.non_wpr_off;
    m.non_wpr_heap_size = fb.non_wpr_size;
    m.gsp_fw_wpr_start = fb.wpr2_start;
    m.gsp_fw_heap_offset = fb.wpr2_heap_off;
    m.gsp_fw_heap_size = fb.wpr2_heap_size;
    m.gsp_fw_offset = fb.elf_off;
    m.boot_bin_offset = fb.boot_off;
    m.frts_offset = fb.frts_off;
    m.frts_size = fb.frts_size;
    m.gsp_fw_wpr_end = fb.wpr2_end;
    m.fb_size = fb.fb_size;
    m.vga_workspace_offset = fb.vga_off;
    m.vga_workspace_size = fb.vga_size;
    m
}
