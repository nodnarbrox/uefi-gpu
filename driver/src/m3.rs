//! Post-allocation channel bring-up. Wire sizes are explicit, not Rust ABI layouts.
use alloc::{format, string::String, vec::Vec};
use core::{ptr::{read_volatile, write_volatile}, sync::atomic::{fence, Ordering}};
use crate::gsp;

fn put(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

pub fn control<W: Fn(u64, u32)>(
    hs: &gsp::GspBootHandshake, wr: &W, falcon: u64,
    client: u32, object: u32, cmd: u32, params: &mut [u8], seq: u32,
    name: &str, log: &mut Vec<String>,
) -> bool {
    let mut request = alloc::vec![0u8; 0x18 + params.len()];
    put(&mut request, 0, client);
    put(&mut request, 4, object);
    put(&mut request, 8, cmd);
    put(&mut request, 0x10, params.len() as u32);
    request[0x18..].copy_from_slice(params);
    if unsafe { gsp::rpc_send_argc(hs, 76, &request, request.len(), seq) }.is_none() {
        log.push(format!("  M3 {}: RPC queue full", name));
        return false;
    }
    gsp::rpc_doorbell(falcon, wr);
    unsafe {
        let cmdq = hs.shared.host.add(hs.cmdq_offset);
        let msgq = hs.shared.host.add(hs.msgq_offset);
        let count = read_volatile(msgq.add(0x0c) as *const u32);
        let entryoff = read_volatile(msgq.add(0x1c) as *const u32) as usize;
        if count == 0 || entryoff < 0x40 { return false; }
        for _ in 0..30000 {
            let w = read_volatile(msgq.add(0x10) as *const u32);
            let r = read_volatile(cmdq.add(0x20) as *const u32);
            if w != r {
                if r >= count { return false; }
                fence(Ordering::SeqCst);
                let elem = msgq.add(entryoff + r as usize * 0x1000);
                let get = |off: usize| read_volatile(elem.add(off) as *const u32);
                let func = get(0x3c);
                let result = get(0x40);
                let len = get(0x38);
                let ec = get(0x28).max(1);
                if ec > count { return false; }
                if func == 0x1020 {
                    let mut source=[0u8;65]; let mut engine=[0u8;65];let mut diag=[0u8;256];
                    for i in 0..65 {source[i]=read_volatile(elem.add(0x68+i));engine[i]=read_volatile(elem.add(0xb8+i));}
                    for i in 0..256 {diag[i]=read_volatile(elem.add(0x104+i));}
                    log.push(format!("  M4 NOCAT: rec={} bug={:08x} subsystem={:08x} error={:08x}{:08x} tdr={:08x} source={} engine={} diag={}",
                        get(0x60)&255,get(0x64),get(0xac),get(0xb4),get(0xb0),get(0xfc),
                        crate::diagnostic_ascii(&source,65),crate::diagnostic_ascii(&engine,65),
                        crate::diagnostic_hex(&diag,(get(0x100) as usize).min(256))));
                }
                let matches = func == 76 && len >= (0x38 + params.len()) as u32
                    && get(0x50) == client && get(0x54) == object && get(0x58) == cmd;
                let status = if matches { get(0x5c) } else { u32::MAX };
                if matches && ec == 1 {
                    for (i, b) in params.iter_mut().enumerate() {
                        *b = read_volatile(elem.add(0x68 + i));
                    }
                }
                log.push(format!("  *** M3 {} reply: matched={} seq={} fn={} cmd=0x{:08x} rpc_result=0x{:08x} ctrl_status=0x{:08x} len=0x{:x} ***",
                    name, matches, get(0x24), func, get(0x58), result, status, len));
                fence(Ordering::SeqCst);
                write_volatile(cmdq.add(0x20) as *mut u32, (r + ec) % count);
                if matches { return result == 0 && status == 0 && ec == 1; }
            }
            uefi::boot::stall(100);
        }
    }
    log.push(format!("  M3 {}: reply TIMEOUT", name));
    false
}

pub fn run<R: Fn(u64) -> u32, W: Fn(u64, u32)>(
    hs: &gsp::GspBootHandshake, rd: &R, wr: &W, bar0: u64, falcon: u64,
    client: u32, subdevice: u32, channel: u32, engine: u32, data_phys: u64, userd_phys: u64,
    map_va: u64, dma: &mut crate::RbioDma, llm_fb: ([u64;4],u32), log: &mut Vec<String>,
) -> bool {
    if !control(hs, wr, falcon, client, channel, 0xa06f0104,
        &mut engine.to_le_bytes(), 11, "BIND", log) { return false; }
    let mut token_bytes = [0u8; 4];
    if !control(hs, wr, falcon, client, channel, 0xc36f0108,
        &mut token_bytes, 12, "WORK_TOKEN", log) { return false; }
    let token = u32::from_le_bytes(token_bytes);
    if !control(hs, wr, falcon, client, channel, 0xa06f0103,
        &mut [1, 0], 13, "SCHEDULE", log) { return false; }

    // Only the existing, mapped scratch page and the allocated USERD slot.
    // Never access WPR, firmware, or any other framebuffer allocation.
    if data_phys.checked_add(0x1000).is_none_or(|v| v > 0x2f4000000)
        || userd_phys.checked_add(0x200).is_none_or(|v| v > 0x2f4000000) {
        log.push(String::from("  M3 semaphore ABORT: scratch overlaps protected FB"));
        return false;
    }
    let window_reg = bar0 + 0x1700;
    let saved = rd(window_reg);
    let aperture = |phys: u64| {
        wr(window_reg, ((phys & !0xfffff) >> 16) as u32);
        bar0 + 0x700000 + (phys & 0xfffff)
    };
    let pb_va = map_va + 0x200;
    let sem_va = map_va + 0xf00;
    let expected = 0x4d33424fu32;
    // NOP, then 4-word incrementing SEMAPHORE A/B/C/D. RELEASE, 4-byte, no WFI.
    let push = [0x20010002, 0, 0x20040004, (sem_va >> 32) as u32,
        sem_va as u32, expected, 0x01100002];
    wr(aperture(data_phys + 0xf00), 0);
    for (i, value) in push.iter().enumerate() {
        wr(aperture(data_phys + 0x200 + i as u64 * 4), *value);
    }
    wr(aperture(data_phys), pb_va as u32 & !3);
    wr(aperture(data_phys + 4), (pb_va >> 32) as u32 | (push.len() as u32) << 10);
    fence(Ordering::SeqCst);
    let initial = rd(aperture(data_phys + 0xf00));
    let ring0 = rd(aperture(data_phys));
    let ring1 = rd(aperture(data_phys + 4));
    wr(aperture(userd_phys + 0x8c), 1);
    fence(Ordering::SeqCst);
    let put = rd(aperture(userd_phys + 0x8c));
    wr(window_reg, saved);
    // GA100 VFN priv 0xb80000 + user 0x30000 + NOTIFY_CHANNEL_PENDING 0x90.
    wr(bar0 + 0xbb0090, token);
    log.push(format!("  M3 SUBMIT: token=0x{:08x} doorbell=0xbb0090 pb_va=0x{:x} words={} ring=[{:08x},{:08x}] USERD=0x{:x} GPPut={} semaphore_initial=0x{:08x}",
        token, pb_va, push.len(), ring0, ring1, userd_phys, put, initial));
    let mut seen = initial;
    for _ in 0..20000 {
        seen = rd(aperture(data_phys + 0xf00));
        if seen == expected { break; }
        uefi::boot::stall(100);
    }
    let get = rd(aperture(userd_phys + 0x88));
    let put = rd(aperture(userd_phys + 0x8c));
    wr(window_reg, saved);
    log.push(format!("  *** M3 SEMAPHORE completion={} expected=0x{:08x} observed=0x{:08x} GPGet={} GPPut={} ***",
        initial == 0 && seen == expected, expected, seen, get, put));
    for rq in 0..2u64 {
        log.push(format!("  M3 PBDMA{}: intr0={:08x} intr1={:08x} status={:08x}", rq,
            rd(bar0 + 0x40108 + rq * 0x800), rd(bar0 + 0x40148 + rq * 0x800),
            rd(bar0 + 0x40120 + rq * 0x800)));
    }
    let completed = initial == 0 && seen == expected;
    if completed {
        crate::m4::run(hs, rd, wr, bar0, falcon, client, subdevice, channel,
            data_phys, userd_phys, map_va, token, dma, llm_fb, log);
    }
    completed
}
