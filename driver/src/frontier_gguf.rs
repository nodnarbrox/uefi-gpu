//! Bounded zero-copy GGUF directory parser. Tensor bodies stay on storage.
//! `logical_bytes` is the validated complete file size, not prefix.len().
extern crate alloc;
use alloc::vec::Vec;
use super::frontier_core::{row_bytes, Error as QuantError};

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Error { Truncated, Magic, Version, Limit, Type, Utf8, Alignment,
    Shape, Overflow, TensorBounds, Overlap, Duplicate, Missing, Split }

struct Reader<'a> { data:&'a[u8], pos:usize }
impl<'a> Reader<'a> {
    fn take(&mut self,n:usize)->Result<&'a[u8],Error> {
        let end=self.pos.checked_add(n).ok_or(Error::Overflow)?;
        let value=self.data.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos=end;Ok(value)
    }
    fn u32(&mut self)->Result<u32,Error> {Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))}
    fn u64(&mut self)->Result<u64,Error> {Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))}
    fn string(&mut self)->Result<&'a str,Error> {
        let n=usize::try_from(self.u64()?).map_err(|_|Error::Limit)?;
        if n>16*1024*1024 {return Err(Error::Limit);}
        core::str::from_utf8(self.take(n)?).map_err(|_|Error::Utf8)
    }
    fn skip(&mut self,kind:u32,nested:bool)->Result<(),Error> {
        match kind {
            0|1|7=>{self.take(1)?;},2|3=>{self.take(2)?;},4|5|6=>{self.take(4)?;},
            10|11|12=>{self.take(8)?;},8=>{self.string()?;},
            9=>{
                if nested {return Err(Error::Type);}
                let element=self.u32()?;let count=self.u64()?;
                if count>1_000_000 {return Err(Error::Limit);}
                for _ in 0..count {self.skip(element,true)?;}
            },
            _=>return Err(Error::Type),
        }Ok(())
    }
}

#[derive(Debug,Clone,Copy)]
pub struct Value<'a> {pub kind:u32, data:&'a[u8]}
impl<'a> Value<'a> {
    pub fn unsigned(&self)->Result<u64,Error> {
        let mut r=Reader{data:self.data,pos:0};
        match self.kind {
            0=>Ok(r.take(1)?[0] as u64),
            2=>Ok(u16::from_le_bytes(r.take(2)?.try_into().unwrap()) as u64),
            4=>Ok(r.u32()? as u64),10=>r.u64(),_=>Err(Error::Type),
        }
    }
    pub fn string(&self)->Result<&'a str,Error> {
        if self.kind!=8 {return Err(Error::Type);}
        Reader{data:self.data,pos:0}.string()
    }
    pub fn float(&self)->Result<f32,Error> {
        if self.kind!=6 {return Err(Error::Type);}
        Ok(f32::from_bits(Reader{data:self.data,pos:0}.u32()?))
    }
    pub fn array(&self)->Result<Array<'a>,Error> {
        if self.kind!=9 {return Err(Error::Type);}
        let mut r=Reader{data:self.data,pos:0};
        let kind=r.u32()?;let count=r.u64()? as usize;
        Ok(Array{reader:r,kind,left:count})
    }
}
pub struct Array<'a> {reader:Reader<'a>,pub kind:u32,pub left:usize}
impl<'a> Iterator for Array<'a> {
    type Item=Value<'a>;
    fn next(&mut self)->Option<Self::Item> {
        if self.left==0 {return None;}
        // Parent parse validated the complete array, so this cannot truncate.
        let start=self.reader.pos;
        if self.reader.skip(self.kind,true).is_err() {self.left=0;return None;}
        self.left-=1;
        Some(Value{kind:self.kind,data:&self.reader.data[start..self.reader.pos]})
    }
}

#[derive(Debug,Clone,Copy)]
pub struct Tensor<'a> {
    pub name:&'a str,pub dims:[u64;4],pub rank:usize,pub kind:u32,
    pub offset:u64,pub bytes:u64,pub row_bytes:u64,
}
impl Tensor<'_> {
    /// Absolute file range for contiguous rows (including an expert's rows).
    pub fn rows(&self,start:u64,count:u64)->Result<(u64,u64),Error> {
        let total=self.bytes/self.row_bytes;
        if count==0 || start.checked_add(count).ok_or(Error::Overflow)?>total {
            return Err(Error::TensorBounds);
        }
        let offset=self.offset.checked_add(start.checked_mul(self.row_bytes).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        Ok((offset,count.checked_mul(self.row_bytes).ok_or(Error::Overflow)?))
    }
    pub fn expert(&self,id:u32)->Result<(u64,u64),Error> {
        if self.rank!=3 || id as u64>=self.dims[2] {return Err(Error::Shape);}
        self.rows((id as u64).checked_mul(self.dims[1]).ok_or(Error::Overflow)?,self.dims[1])
    }
}

#[derive(Debug)]
pub struct Shard<'a> {
    pub metadata:Vec<(&'a str,Value<'a>)>,pub tensors:Vec<Tensor<'a>>,
    pub data_offset:u64,pub logical_bytes:u64,pub split_no:u64,pub split_count:u64,
    pub total_tensors:u64,
}

/// Complete directory spanning all split files; no tensor body is retained.
pub struct ModelIndex<'a> {pub shards:Vec<Shard<'a>>}
impl<'a> ModelIndex<'a> {
    pub fn new(mut shards:Vec<Shard<'a>>)->Result<Self,Error> {
        if shards.is_empty() {return Err(Error::Split);}
        shards.sort_unstable_by_key(|s|s.split_no);
        let total=shards[0].total_tensors;let count=shards.len() as u64;
        let mut names=alloc::collections::BTreeSet::new();
        for (i,s) in shards.iter().enumerate() {
            if s.split_no!=i as u64 || s.split_count!=count || s.total_tensors!=total {return Err(Error::Split);}
            for t in &s.tensors {if !names.insert(t.name) {return Err(Error::Duplicate);}}
        }
        if names.len() as u64!=total {return Err(Error::Split);}
        Ok(Self{shards})
    }
    pub fn tensor(&self,name:&str)->Result<(usize,&Tensor<'a>),Error> {
        for (i,s) in self.shards.iter().enumerate() {
            if let Ok(t)=s.tensor(name) {return Ok((i,t));}
        }Err(Error::Missing)
    }
    /// Independent counters for planning bounded caches; compressed on-disk bytes.
    pub fn inventory(&self)->Result<[u64;3],Error> {
        let mut counts=[0u64;3];
        for s in &self.shards {for t in &s.tensors {
            let category=if t.name=="per_layer_token_embd.weight" {1}
                else if t.name.ends_with("_exps.weight") {2}else{0};
            counts[category]=counts[category].checked_add(t.bytes).ok_or(Error::Overflow)?;
        }}Ok(counts)
    }
}
impl<'a> Shard<'a> {
    pub fn value(&self,key:&str)->Result<Value<'a>,Error> {
        self.metadata.iter().find(|(k,_)|*k==key).map(|(_,v)|*v).ok_or(Error::Missing)
    }
    pub fn tensor(&self,name:&str)->Result<&Tensor<'a>,Error> {
        self.tensors.iter().find(|t|t.name==name).ok_or(Error::Missing)
    }
    pub fn parse(prefix:&'a[u8],logical_bytes:u64)->Result<Self,Error> {
        if prefix.len()>16*1024*1024 {return Err(Error::Limit);}
        let mut r=Reader{data:prefix,pos:0};
        if r.take(4)?!=b"GGUF" {return Err(Error::Magic);}
        if r.u32()?!=3 {return Err(Error::Version);}
        let nt=r.u64()?;let nk=r.u64()?;
        if nt>4096 || nk>1024 {return Err(Error::Limit);}
        let mut metadata=Vec::with_capacity(nk as usize);
        for _ in 0..nk {
            let key=r.string()?;if key.len()>256 {return Err(Error::Limit);}
            if metadata.iter().any(|(k,_)|*k==key) {return Err(Error::Duplicate);}
            let kind=r.u32()?;let start=r.pos;r.skip(kind,false)?;
            metadata.push((key,Value{kind,data:&prefix[start..r.pos]}));
        }
        let mut shard=Self{metadata,tensors:Vec::with_capacity(nt as usize),data_offset:0,
            logical_bytes,split_no:0,split_count:1,total_tensors:nt};
        if let Ok(v)=shard.value("split.no") {shard.split_no=v.unsigned()?;}
        if let Ok(v)=shard.value("split.count") {shard.split_count=v.unsigned()?;}
        if let Ok(v)=shard.value("split.tensors.count") {
            // This field is int32 in standard split GGUFs.
            shard.total_tensors=if v.kind==5 {
                let n=i32::from_le_bytes(v.data.try_into().map_err(|_|Error::Type)?);
                if n<0 {return Err(Error::Split);}n as u64
            }else{v.unsigned()?};
        }
        if shard.split_count==0 || shard.split_count>64 || shard.split_no>=shard.split_count ||
            shard.total_tensors<nt || shard.total_tensors>65536 {return Err(Error::Split);}
        let alignment=match shard.value("general.alignment") {Ok(v)=>v.unsigned()?,Err(_)=>32};
        if !alignment.is_power_of_two() || alignment>4096 {return Err(Error::Alignment);}
        for _ in 0..nt {
            let name=r.string()?;if name.len()>256 {return Err(Error::Limit);}
            if shard.tensors.iter().any(|t|t.name==name) {return Err(Error::Duplicate);}
            let rank=r.u32()? as usize;
            if !(1..=4).contains(&rank) {return Err(Error::Shape);}
            let mut dims=[1;4];for d in &mut dims[..rank] {*d=r.u64()?;if *d==0 {return Err(Error::Shape);}}
            let kind=r.u32()?;let offset=r.u64()?;
            if offset%alignment!=0 {return Err(Error::Alignment);}
            let width=usize::try_from(dims[0]).map_err(|_|Error::Overflow)?;
            let rb=row_bytes(kind,width).map_err(|e|match e {
                QuantError::QuantType=>Error::Type,QuantError::Bounds=>Error::Overflow,_=>Error::Shape})? as u64;
            let bytes=dims[1..rank].iter().try_fold(rb,|n,d|n.checked_mul(*d)).ok_or(Error::Overflow)?;
            shard.tensors.push(Tensor{name,dims,rank,kind,offset,bytes,row_bytes:rb});
        }
        shard.data_offset=(r.pos as u64).checked_add(alignment-1).ok_or(Error::Overflow)? & !(alignment-1);
        if shard.data_offset>logical_bytes {return Err(Error::TensorBounds);}
        let mut ranges=Vec::with_capacity(nt as usize);
        for t in &mut shard.tensors {
            t.offset=t.offset.checked_add(shard.data_offset).ok_or(Error::Overflow)?;
            let end=t.offset.checked_add(t.bytes).ok_or(Error::Overflow)?;
            if end>logical_bytes {return Err(Error::TensorBounds);}
            ranges.push((t.offset,end));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|w|w[0].1>w[1].0) {return Err(Error::Overlap);}
        Ok(shard)
    }
}

#[cfg(test)] mod tests {
    use super::*;
    const META:&[u8]=include_bytes!("../../reference/q3-metadata.gguf");
    const SECOND:&[u8]=include_bytes!("../../reference/shard2-header.gguf");
    const THIRD:&[u8]=include_bytes!("../../reference/shard3-header.gguf");
    #[test] fn metadata_shard() {
        let s=Shard::parse(META,10946624).unwrap();
        assert_eq!(s.data_offset,10946624);assert_eq!(s.tensors.len(),0);
        assert_eq!((s.split_no,s.split_count,s.total_tensors),(0,3,1224));
        assert_eq!(s.value("general.architecture").unwrap().string(),Ok("qwen4exp"));
        assert_eq!(s.value("qwen4exp.embedding_length").unwrap().unsigned(),Ok(2560));
        assert_eq!(s.value("qwen4exp.attention.layer_norm_rms_epsilon").unwrap().float(),Ok(1e-6));
        let tokens=s.value("tokenizer.ggml.tokens").unwrap().array().unwrap();
        assert_eq!(tokens.left,248320);assert_eq!(tokens.kind,8);
        assert_eq!(tokens.count(),248320);
        let mult:Vec<u64>=s.value("qwen4exp.ple.layer_multipliers").unwrap().array().unwrap()
            .map(|v|v.unsigned().unwrap()).collect();
        assert_eq!(mult,[23703573157769,20109073645365,8052911324071]);
    }
    #[test] fn third_shard_and_expert_ranges() {
        let s=Shard::parse(THIRD,39992153376).unwrap();
        assert_eq!(s.data_offset,52512);assert_eq!(s.tensors.len(),822);
        assert_eq!((s.split_no,s.split_count,s.total_tensors),(2,3,1224));
        let mut experts=0;
        for t in &s.tensors {
            assert_eq!(t.rows(0,t.bytes/t.row_bytes).unwrap(),(t.offset,t.bytes));
            assert_eq!(t.rows(t.bytes/t.row_bytes,1),Err(Error::TensorBounds));
            if t.rank==3 {
                assert_eq!(t.dims[2],512);experts+=1;
                let first=t.expert(0).unwrap();let last=t.expert(511).unwrap();
                assert_eq!(first.0,t.offset);assert_eq!(last.0+last.1,t.offset+t.bytes);
                assert_eq!(t.expert(512),Err(Error::Shape));
            }
        }
        // 65 IQ3_XXS + 29 IQ4_NL + 3 Q8_0 tensors, from the pinned directory.
        assert_eq!(experts,97);
    }
    #[test] fn reject_truncation_overflow_and_bad_offsets() {
        for n in [0,3,4,7,8,23,24,1024,50000] {
            assert!(Shard::parse(&THIRD[..n],39992153376).is_err());
        }
        assert_eq!(Shard::parse(THIRD,1048576).unwrap_err(),Error::TensorBounds);
        let mut bad=THIRD.to_vec();bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(Shard::parse(&bad,39992153376).unwrap_err(),Error::Limit);
        bad=THIRD.to_vec();bad[4]=2;
        assert_eq!(Shard::parse(&bad,39992153376).unwrap_err(),Error::Version);
    }
    fn synthetic(offsets:&[u64],width:u64)->Vec<u8> {
        let mut out=Vec::new();out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(offsets.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        for (i,offset) in offsets.iter().enumerate() {
            let name=alloc::format!("tensor.{i}");out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&width.to_le_bytes());out.extend_from_slice(&2u64.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());out.extend_from_slice(&offset.to_le_bytes());
        }out
    }
    #[test] fn malformed_ranges() {
        assert_eq!(Shard::parse(&synthetic(&[0,32],32),10000).unwrap_err(),Error::Overlap);
        assert_eq!(Shard::parse(&synthetic(&[1],32),10000).unwrap_err(),Error::Alignment);
        assert_eq!(Shard::parse(&synthetic(&[u64::MAX-31],32),u64::MAX).unwrap_err(),Error::Overflow);
        assert_eq!(Shard::parse(&synthetic(&[0],u64::MAX-31),u64::MAX).unwrap_err(),Error::Overflow);
        assert_eq!(Shard::parse(&synthetic(&[0],33),10000).unwrap_err(),Error::Shape);
        assert!(Shard::parse(&synthetic(&[0,96],32),10000).is_ok());
    }
    #[test] fn whole_model_directory() {
        let shards=alloc::vec![Shard::parse(META,10946624).unwrap(),
            Shard::parse(SECOND,49983253824).unwrap(),Shard::parse(THIRD,39992153376).unwrap()];
        let model=ModelIndex::new(shards).unwrap();
        assert_eq!(model.inventory().unwrap(),[5351626240,28800138240,55823564800]);
        let (disk,lookup)=model.tensor("per_layer_token_embd.weight").unwrap();
        assert_eq!(disk,1);assert_eq!(lookup.kind,20);assert_eq!(lookup.row_bytes,90);
        assert_eq!(lookup.dims[..2],[160,320001536]);
        let rows=super::super::frontier_core::ple_rows(42,[None,None]);
        for row in rows {let (start,n)=lookup.rows(row as u64,1).unwrap();
            assert_eq!(n,90);assert!(start+n<=model.shards[disk].logical_bytes);}
        for layer in 0..48 {for part in ["gate","up","down"] {
            let name=alloc::format!("blk.{layer}.ffn_{part}_exps.weight");
            let (_,t)=model.tensor(&name).unwrap();assert_eq!(t.dims[2],512);
            assert_eq!(t.expert(511).unwrap().1,t.bytes/512);
        }}
        let mut missing=model.shards;missing.pop();
        assert!(matches!(ModelIndex::new(missing),Err(Error::Split)));
    }
}
