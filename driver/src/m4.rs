//! First compute proof: 4x4 FP32 matmul, launched via QMD 3.0 on sm_86.
use alloc::{format, string::String, vec::Vec};
use core::{ptr::{read_volatile, write_volatile}, sync::atomic::{fence, Ordering}};
use crate::{gsp, m3};

fn alloc_compute<W: Fn(u64,u32)>(hs: &gsp::GspBootHandshake, wr: &W,
    falcon: u64, client: u32, channel: u32, log: &mut Vec<String>) -> bool {
    let mut req = [0u8; 0x30];
    for (off, val) in [(0,client),(4,channel),(8,0xc7c00001),(12,0xc7c0),
        (20,16),(32,2),(40,16)] {
        req[off..off+4].copy_from_slice(&val.to_le_bytes());
    }
    if unsafe { gsp::rpc_send_argc(hs,103,&req,req.len(),16) }.is_none() {return false;}
    gsp::rpc_doorbell(falcon,wr);
    unsafe {
        let cq=hs.shared.host.add(hs.cmdq_offset);
        let mq=hs.shared.host.add(hs.msgq_offset);
        let count=read_volatile(mq.add(12) as *const u32);
        let off=read_volatile(mq.add(28) as *const u32) as usize;
        for _ in 0..50000 {
            let r=read_volatile(cq.add(32) as *const u32);
            if r!=read_volatile(mq.add(16) as *const u32) {
                if count==0 || r>=count {return false;}
                fence(Ordering::SeqCst);
                let e=mq.add(off+r as usize*4096);
                let get=|o:usize| read_volatile(e.add(o) as *const u32);
                let matched=get(0x3c)==103 && get(0x50)==client && get(0x58)==0xc7c00001;
                let status=get(0x60); let rpc=get(0x40); let step=get(0x28).max(1);
                log.push(format!("  *** M4 COMPUTE_ALLOC reply: matched={} seq={} fn={} rpc_result=0x{:08x} alloc_status=0x{:08x} class=0xc7c0 ***",
                    matched,get(0x24),get(0x3c),rpc,status));
                fence(Ordering::SeqCst);
                write_volatile(cq.add(32) as *mut u32,(r+step)%count);
                if matched {return status==0 && rpc==0;}
            }
            uefi::boot::stall(100);
        }
    }
    log.push(String::from("  M4 COMPUTE_ALLOC TIMEOUT"));false
}

pub(crate) fn field(q: &mut [u32;64], lo:usize, hi:usize, value:u64) {
    assert!(hi-lo<64 && (hi-lo==63 || value>>(hi-lo+1)==0));
    for bit in lo..=hi {
        let mask=1u32<<(bit%32);
        q[bit/32]=(q[bit/32]&!mask)|(((value>>(bit-lo)) as u32&1)<<(bit%32));
    }
}

fn promote<R:Fn(u64)->u32,W:Fn(u64,u32)>(hs:&gsp::GspBootHandshake,rd:&R,wr:&W,
    bar0:u64,falcon:u64,client:u32,channel:u32,data_phys:u64,info:&[u8],log:&mut Vec<String>) -> bool {
    let saved=rd(bar0+0x1700);
    let ap=|p:u64| {wr(bar0+0x1700,((p&!0xfffff)>>16) as u32);bar0+0x700000+(p&0xfffff)};
    let write64=|p:u64,v:u64| {wr(ap(p),v as u32);wr(ap(p+4),(v>>32) as u32);};
    let pdb=data_phys-0x5000;
    // Context tables in non-WPR scratch; context storage in known free FB[1].
    // 0x05000000..0x08000000 virtual: PD0 slots 40..63, 4KB small pages.
    for slot in 0..24u64 {
        let pt=pdb+0x30000+slot*4096;
        for d in 0..1024 {wr(ap(pt+d*4),0);}
        write64(pdb+0x3000+(40+slot)*16,0);
        write64(pdb+0x3000+(40+slot)*16+8,(pt>>4)|2);
    }
    let mut ctrl=[0u8;560];
    for (off,value) in [(0,1u32),(12,client),(16,channel)] {ctrl[off..off+4].copy_from_slice(&value.to_le_bytes());}
    let mut pa=0x04000000u64;let mut va=0x05000000u64;let mut count=0;
    // Nouveau r535 mapping: info engine index -> PROMOTE buffer ID, initialize.
    for (idx,id,init,nonmapped) in [(0,0,true,false),(16,2,true,false),(17,3,false,false),
        (18,4,false,false),(19,5,false,false),(20,6,false,false),(23,9,true,false),
        (24,10,true,true),(24,11,true,false)] {
        let mut size=u32::from_le_bytes(info[idx*8..idx*8+4].try_into().unwrap()) as u64;
        if size==0 || size==u32::MAX as u64 {continue;}
        if id==0 {size=((size+4095)&!4095)+64*4096;}
        let page_align=if size>=2*1024*1024 {2*1024*1024} else if size>=65536 {65536} else {4096};
        let align=if id==5 {size.next_power_of_two()} else {page_align};
        pa=(pa+page_align-1)&!(page_align-1);va=(va+align-1)&!(align-1);
        let rounded=(size+4095)&!4095;
        if pa+rounded>0x08000000 || va+rounded>0x08000000 {wr(bar0+0x1700,saved);return false;}
        // Only buffers initialized by RM require host zeroing, per API contract.
        if init {for d in 0..rounded/4 {wr(ap(pa+d*4),0);}}
        if !nonmapped {for off in (0..rounded).step_by(4096) {
            let v=va+off;let slot=(v-0x05000000)>>21;let ptei=(v>>12)&511;
            // PRIV PTE bit5, RO bit6 for the privileged access map.
            let flags=1|0x20|if id==11 {0x40} else {0};
            write64(pdb+0x30000+slot*4096+ptei*8,((pa+off)>>4)|flags);
        }}
        let base=48+count*32;
        if init {
            ctrl[base..base+8].copy_from_slice(&pa.to_le_bytes());
            ctrl[base+16..base+24].copy_from_slice(&size.to_le_bytes());
            ctrl[base+24..base+28].copy_from_slice(&4u32.to_le_bytes());
        }
        if !nonmapped {ctrl[base+8..base+16].copy_from_slice(&va.to_le_bytes());}
        ctrl[base+28..base+30].copy_from_slice(&(id as u16).to_le_bytes());
        ctrl[base+30]=init as u8;ctrl[base+31]=nonmapped as u8;
        log.push(format!("  M4 GR_CTX buffer={} size=0x{:x} PA=0x{:x} VA=0x{:x} init={} nonmapped={}",id,size,pa,va,init,nonmapped));
        count+=1;pa+=rounded;if !nonmapped {va+=rounded;}
    }
    ctrl[40..44].copy_from_slice(&(count as u32).to_le_bytes());
    fence(Ordering::SeqCst);wr(bar0+0x1700,saved);
    wr(bar0+0xb830a0,(pdb>>8) as u32);wr(bar0+0xb830a4,(pdb>>40) as u32);
    wr(bar0+0xb830b0,0x80000007);
    for _ in 0..20000 {if rd(bar0+0xb830b0)&0x80000000==0 {break;}uefi::boot::stall(10);}
    m3::control(hs,wr,falcon,client,0x5d1d0000,0x2080012b,&mut ctrl,15,"GR_PROMOTE",log)
}

pub fn run<R:Fn(u64)->u32,W:Fn(u64,u32)>(hs:&gsp::GspBootHandshake,rd:&R,wr:&W,
    bar0:u64,falcon:u64,client:u32,subdevice:u32,channel:u32,data_phys:u64,userd:u64,
    va:u64,token:u32,dma:&mut crate::RbioDma,llm_fb:([u64;4],u32),log:&mut Vec<String>) -> bool {
    let mut ctxinfo=[0u8;8*26*8];
    if !m3::control(hs,wr,falcon,0xc2000006,0xabcd2080,0x20800a32,
        &mut ctxinfo,14,"GR_CTXINFO",log) {return false;}
    for id in 0..26 {
        let size=u32::from_le_bytes(ctxinfo[id*8..id*8+4].try_into().unwrap());
        let align=u32::from_le_bytes(ctxinfo[id*8+4..id*8+8].try_into().unwrap());
        log.push(format!("  M4 GR_CTXINFO id={} size=0x{:x} alignment=0x{:x}",id,size,align));
    }
    if !promote(hs,rd,wr,bar0,falcon,client,channel,data_phys,&ctxinfo,log) {return false;}
    if !alloc_compute(hs,wr,falcon,client,channel,log) {return false;}
    if !m3::control(hs,wr,falcon,client,channel,0xa06f0103,&mut [1,0],17,"COMPUTE_SCHEDULE",log) {return false;}
    let extra_phys=data_phys+0x5000;
    if va!=0x04000000 || extra_phys+0x10000>0x2f4000000 {return false;}
    let saved=rd(bar0+0x1700);
    let ap=|p:u64| {wr(bar0+0x1700,((p&!0xfffff)>>16) as u32);bar0+0x700000+(p&0xfffff)};
    // Extend the existing small-page table, never overwrite instance/RAMFC/USERD.
    for page in 0..16u64 {
        let phys=extra_phys+page*4096;
        for dw in 0..1024 {wr(ap(phys+dw*4),0);}
        let pte=(phys>>4)|1;
        wr(ap(data_phys-0x1000+(page+1)*8),pte as u32);
        wr(ap(data_phys-0x1000+(page+1)*8+4),(pte>>32) as u32);
    }
    let code=include_bytes!("../kernels/matmul4.sass");
    let c2=include_bytes!("../kernels/matmul4.c2");
    for (i,b) in code.chunks_exact(4).enumerate() {
        wr(ap(extra_phys+i as u64*4),u32::from_le_bytes(b.try_into().unwrap()));
    }
    for dw in 0..128 {wr(ap(data_phys+0x900+dw*4),0);}
    wr(ap(data_phys+0x900+40),0xfffdc0);
    for (i,b) in c2.chunks_exact(4).enumerate() {
        wr(ap(data_phys+0xb00+i as u64*4),u32::from_le_bytes(b.try_into().unwrap()));
    }
    let mut a=[0f32;16];let mut b=[0f32;16];let mut expected=[0f32;16];
    for i in 0..16 {
        a[i]=(i+1) as f32;b[i]=((i*3+1)%7) as f32-2.0;
        wr(ap(extra_phys+0x2000+i as u64*4),a[i].to_bits());
        wr(ap(extra_phys+0x2100+i as u64*4),b[i].to_bits());
        wr(ap(extra_phys+0x2200+i as u64*4),0x7fc00000);
    }
    for row in 0..4 {for col in 0..4 {for k in 0..4 {
        expected[row*4+col]+=a[row*4+k]*b[k*4+col];
    }}}
    let mut q=[0u32;64];
    for (lo,hi,v) in [
        (128,133,0x3f),(134,134,1),(186,191,0x3f),(368,369,1),
        (378,378,1),(382,382,1),(384,415,1),(416,431,1),(448,463,1),
        (544,561,0),(562,567,9),(569,574,26),(580,583,3),
        (592,607,16),(608,623,1),(624,639,1),(640,640,1),(642,642,1),
        (648,656,23),(657,662,9),(763,767,1),
        (768,799,va+0xf10),(819,819,1),(823,823,1),(830,831,1),
        (832,863,0x4d34434f),
        (1024,1055,va+0x900),(1074,1074,1),(1075,1087,0x160/16),
        (1152,1183,va+0xb00),(1202,1202,1),(1203,1215,2),
        (1536,1567,va+0x1000),(256,287,(va+0x1000)>>8),
        (1641,1649,(code.len()/256) as u64),(1656,1663,0x86)] {
        field(&mut q,lo,hi,v);
    }
    for (i,v) in q.iter().enumerate() {wr(ap(data_phys+0x800+i as u64*4),*v);}
    wr(ap(data_phys+0xf10),0);
    fence(Ordering::SeqCst);
    wr(bar0+0x1700,saved);
    wr(bar0+0xb830a0,((data_phys-0x5000)>>8) as u32);
    wr(bar0+0xb830a4,((data_phys-0x5000)>>40) as u32);
    wr(bar0+0xb830b0,0x80000007);
    let mut flushed=false;
    for _ in 0..20000 {if rd(bar0+0xb830b0)&0x80000000==0 {flushed=true;break;} uefi::boot::stall(10);}
    if !flushed {log.push(String::from("  M4 mapping flush TIMEOUT"));return false;}
    let mut pb=Vec::<u32>::new();
    let mut method=|addr:u32,vals:&[u32]| {
        pb.push(0x20000000|((vals.len() as u32)<<16)|(1<<13)|(addr>>2));pb.extend_from_slice(vals);
    };
    method(0,&[0xc7c0]);
    method(0x7b0,&[0x7293,0]);method(0x2a0,&[0x7294,0]);
    method(0x1698,&[0x1011]); // instruction bit0, global bit4, constant bit12
    method(0x2b4,&[((va+0x800)>>8) as u32]);method(0x2c0,&[9]);
    for (i,v) in pb.iter().enumerate() {wr(ap(data_phys+0x200+i as u64*4),*v);}
    wr(ap(data_phys+8),(va+0x200) as u32);
    wr(ap(data_phys+12),(pb.len() as u32)<<10);
    fence(Ordering::SeqCst);let _=rd(ap(data_phys+12));
    wr(ap(userd+0x8c),2);let _=rd(ap(userd+0x8c));
    wr(bar0+0x1700,saved);wr(bar0+0xbb0090,token);
    let mut completed=false;
    for _ in 0..50000 {if rd(ap(data_phys+0xf10))==0x4d34434f {completed=true;break;}uefi::boot::stall(100);}
    let sem=rd(ap(data_phys+0xf10));let gpget=rd(ap(userd+0x88));
    let mut matched=0;
    for row in 0..4 {
        let mut seen=[0f32;4];
        for col in 0..4 {let i=row*4+col;seen[col]=f32::from_bits(rd(ap(extra_phys+0x2200+i as u64*4)));
            if seen[col]==expected[i] {matched+=1;}}
        log.push(format!("  M4 MATMUL row{}: GPU={:?} expected={:?}",row,seen,&expected[row*4..row*4+4]));
    }
    wr(bar0+0x1700,saved);
    log.push(format!("  *** M4 MATMUL completion={} semaphore=0x{:08x} GPGet={} matches={}/16 PASS={} ***",
        completed,sem,gpget,matched,completed&&matched==16));
    for rq in 0..2u64 {log.push(format!("  M4 PBDMA{} intr0={:08x} intr1={:08x}",rq,rd(bar0+0x40108+rq*0x800),rd(bar0+0x40148+rq*0x800)));}
    if !completed || matched!=16 {
        for off in [0x400100,0x400108,0x400110,0x400118,0x400130,0x400700,0x400704,0x400708] {
            log.push(format!("  M4 GR reg[{:06x}]={:08x}",off,rd(bar0+off)));
        }
        let mut tok=[0u8;4];
        m3::control(hs,wr,falcon,client,channel,0xc36f0108,&mut tok,18,"POST_COMPUTE",log);
        unsafe {
            let base=hs.logrm.host as *const u64;
            let n=hs.logrm.size/8-1;let put=read_volatile(base);
            for idx in put.saturating_sub(64)..put {
                log.push(format!("  M4 LOGRM[{}]=0x{:016x}",idx,read_volatile(base.add(1+idx as usize%n))));
            }
        }
    }
    if completed&&matched==16 {
        // Successful compute setup ends at outbound sequence17. Keep every
        // subsequent control contiguous, including optional diagnostic queries.
        let mut next_sequence=18;
        let boosted=crate::frontier_perf::begin(hs,wr,falcon,client,subdevice,&mut next_sequence,log);
        if boosted{crate::llm::run(rd,wr,bar0,data_phys,userd,token,dma,&q,llm_fb,log);}
        else{uefi::println!("FRONTIER_PERF_EXPERIMENT skipped=true reason=boost_not_accepted");}
        crate::frontier_perf::end(hs,wr,falcon,client,subdevice,&mut next_sequence,log);
    }
    completed&&matched==16
}
