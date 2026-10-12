//! Correctness-first qwen4exp decode graph. Linears are supplied by the GPU
//! backend; CPU state math is explicit so it can be checked before fusion.
//! Short context only: <=1024 positions, where QSA selects all valid KV tokens.
extern crate alloc;
use alloc::{format,string::String,vec,vec::Vec};
use super::frontier_core::ple_rows;
const D:usize=2560;const HC:usize=4;const W:usize=D*HC;const R:usize=320;
const HD:usize=128;const KH:usize=16;const VH:usize=48;const FF:usize=640;
pub const VOCAB:usize=248320;pub const LIMIT:usize=1024;
#[derive(Debug)] pub enum Error { Backend, Shape, NonFinite, Context }
type Result<T>=core::result::Result<T,Error>;
pub struct LinearRequest<'a>{pub name:&'a str,pub expert:Option<usize>,pub input:&'a[f32]}

pub trait Weights {
    fn phase_mark(&self)->(u64,u64){(0,0)}
    fn phase_end(&mut self,_stage:usize,_mark:(u64,u64)){}
    fn linear(&mut self,name:&str,expert:Option<usize>,x:&[f32])->Result<Vec<f32>>;
    fn vector(&mut self,name:&str)->Result<Vec<f32>>;
    fn row(&mut self,name:&str,row:usize)->Result<Vec<f32>>;
    fn batch(&mut self,jobs:&[LinearRequest<'_>])->Result<Vec<Vec<f32>>>{
        jobs.iter().map(|j|self.linear(j.name,j.expert,j.input)).collect()
    }
    fn trace(&mut self,_layer:usize,_stage:&str,_x:&[f32]) {}
    fn recurrent_on_gpu(&self)->bool {false}
    fn recurrent(&mut self,_layer:usize,_position:usize,conv:&[f32],alpha:&[f32],beta:&[f32],a:&[f32],dt:&[f32],state:&mut[f32])->Result<Vec<f32>> {
        delta_step(conv,alpha,beta,a,dt,state)
    }
}
fn exp(x:f32)->f32 {super::frontier_cpu_expf::exp(x)}
fn sqrt(x:f32)->f32 {libm::sqrtf(x)}
fn sigmoid(x:f32)->f32 {super::frontier_cpu_expf::activation(x,false)}
fn silu(x:f32)->f32 {super::frontier_cpu_expf::activation(x,true)}
fn softplus(x:f32)->f32 {if x>20.0{x}else{libm::log1pf(exp(x))}}
fn finite(x:&[f32])->Result<()> {if x.iter().all(|v|v.is_finite()){Ok(())}else{Err(Error::NonFinite)}}
fn norm(x:&[f32],gamma:&[f32],group:usize)->Result<Vec<f32>> {
    if group==0||x.len()%group!=0||gamma.len()!=x.len(){return Err(Error::Shape);}
    let mut y=vec![0.;x.len()];
    if let Some(ok)=super::frontier_cpu_math::norm(x,gamma,&mut y,group){return if ok{Ok(y)}else{Err(Error::NonFinite)};}
    for base in (0..x.len()).step_by(group){let mut sum=0f64;
        for &v in &x[base..base+group]{sum+=(v*v)as f64;}
        let scale=1.0/sqrt((sum/group as f64)as f32+1e-6);
        for i in base..base+group{y[i]=(x[i]*scale)*gamma[i];}}
    finite(&y)?;Ok(y)
}
fn l2(x:&mut[f32]) {let mut sum=0f64;for &v in x.iter(){sum+=(v*v)as f64;}
    let scale=1.0/sqrt(sum as f32).max(1e-6);for v in x{*v*=scale;}}
fn softmax(x:&mut[f32])->Result<()> {finite(x)?;
    let max=x.iter().copied().fold(f32::NEG_INFINITY,f32::max);let mut sum=0f32;
    for v in x.iter_mut(){*v=exp(*v-max);sum+=*v;}
    if sum<=0.0||!sum.is_finite(){return Err(Error::NonFinite);}
    for v in x{*v/=sum;}Ok(())}

struct LayerState {conv:Vec<f32>,delta:Vec<f32>,keys:Vec<f32>,values:Vec<f32>}
pub struct Forward<B:Weights> {pub backend:B,states:Vec<LayerState>,ple_history:Vec<f32>,
    previous:[Option<u32>;2],pub position:usize}
impl<B:Weights> Forward<B> {
    pub fn new(backend:B)->Self {
        let gpu_state=backend.recurrent_on_gpu();
        let states=(0..48).map(|i|if i%4==3 {LayerState{conv:Vec::new(),delta:Vec::new(),keys:Vec::new(),values:Vec::new()}}
            else{LayerState{conv:vec![0.;3*10240],delta:if gpu_state{Vec::new()}else{vec![0.;HD*HD*VH]},keys:Vec::new(),values:Vec::new()}}).collect();
        Self{backend,states,ple_history:vec![0.;9*W],previous:[None,None],position:0}
    }
    fn vector(&mut self,name:&str,size:usize)->Result<Vec<f32>> {
        let x=self.backend.vector(name)?;if x.len()!=size{return Err(Error::Shape);}finite(&x)?;Ok(x)
    }
    fn linear(&mut self,name:&str,expert:Option<usize>,x:&[f32],size:usize)->Result<Vec<f32>> {
        let y=self.backend.linear(name,expert,x)?;if y.len()!=size{return Err(Error::Shape);}finite(&y)?;Ok(y)
    }
    fn projections(&mut self,names:&[String],x:&[f32],sizes:&[usize])->Result<Vec<Vec<f32>>> {
        if names.len()!=sizes.len(){return Err(Error::Shape);}
        let requests:Vec<_>=names.iter().map(|name|LinearRequest{name,expert:None,input:x}).collect();
        let output=self.backend.batch(&requests)?;
        if output.len()!=sizes.len(){return Err(Error::Shape);}
        for (y,&size) in output.iter().zip(sizes){if y.len()!=size{return Err(Error::Shape);}finite(y)?;}
        Ok(output)
    }
    fn mix(&mut self,hidden:&[f32],prefix:&str,inject:bool)->Result<(Vec<f32>,Vec<f32>)> {
        let gamma=self.vector(&format!("{prefix}_norm.weight"),W)?;
        let xn=norm(hidden,&gamma,D)?;
        let (mut lo,inj)=if inject {
            let mut projected=self.projections(&[format!("{prefix}_down.weight"),format!("{prefix}_inject.weight")],&xn,&[R,HC])?;
            let inj=projected.pop().ok_or(Error::Shape)?;(projected.pop().ok_or(Error::Shape)?,inj)
        }else{(self.linear(&format!("{prefix}_down.weight"),None,&xn,R)?,Vec::new())};
        for v in &mut lo{*v=silu(*v*0.25);}
        let gate=self.linear(&format!("{prefix}_up.weight"),None,&lo,W)?;
        let mut out=vec![0.;D];
        for i in 0..D {out[i]=xn[i]*sigmoid(gate[i]);
            for c in 1..HC{out[i]+=xn[c*D+i]*sigmoid(gate[c*D+i]);}out[i]*=0.25;}
        Ok((out,inj))
    }
    fn combine(hidden:&mut[f32],out:&[f32],inject:&[f32]) {
        for c in 0..HC {let scale=2.0*sigmoid(inject[c]*0.25);
            if super::frontier_cpu_axpy::axpy(&mut hidden[c*D..(c+1)*D],out,scale).is_none(){
                for i in 0..D {hidden[c*D+i]+=out[i]*scale;}}}
    }
    fn ple(&mut self,token:u32,hidden:&mut[f32])->Result<()> {
        let mut emb=Vec::with_capacity(D);
        for row in ple_rows(token,self.previous){let values=self.backend.row("per_layer_token_embd.weight",row as usize)?;
            if values.len()!=160{return Err(Error::Shape);}emb.extend_from_slice(&values);}
        let mut projected=self.projections(&[String::from("blk.1.ple_key.weight"),String::from("blk.1.ple_value.weight")],&emb,&[W,D])?;
        let value=projected.pop().ok_or(Error::Shape)?;let key=projected.pop().ok_or(Error::Shape)?;
        let nk=self.vector("blk.1.ple_norm_key.weight",W)?;let nq=self.vector("blk.1.ple_norm_query.weight",W)?;
        let key=norm(&key,&nk,D)?;let query=norm(hidden,&nq,D)?;
        let mut gated=vec![0.;W];
        for c in 0..HC {let mut dot=0f32;for i in 0..D{dot+=key[c*D+i]*query[c*D+i];}
            dot/=sqrt(D as f32);let mag=sqrt(dot.abs().max(1e-6));
            let gate=sigmoid(if dot>0.0{mag}else if dot<0.0{-mag}else{0.0});
            for i in 0..D{gated[c*D+i]=value[i]*gate;}}
        let nc=self.vector("blk.1.ple_norm_conv.weight",W)?;let normalized=norm(&gated,&nc,D)?;
        let kernel=self.vector("blk.1.ple_conv1d.weight",4*W)?;
        for i in 0..W {let mut cv=0.;for k in 0..4 {let lag=(3-k)*3;
            cv+=kernel[i*4+k]*if lag==0{normalized[i]}else{self.ple_history[(9-lag)*W+i]};}
            hidden[i]+=gated[i]+silu(cv);}
        self.ple_history.copy_within(W..9*W,0);self.ple_history[8*W..].copy_from_slice(&normalized);
        finite(hidden)
    }
    fn linear_attention(&mut self,layer:usize,x:&[f32])->Result<Vec<f32>> {
        let name=|s:&str|format!("blk.{layer}.{s}");
        let mut projected=self.projections(&[name("attn_qkv.weight"),name("attn_gate.weight"),name("ssm_alpha.weight"),name("ssm_beta.weight")],x,&[10240,VH*HD,VH,VH])?;
        let beta=projected.pop().ok_or(Error::Shape)?;let alpha=projected.pop().ok_or(Error::Shape)?;
        let z=projected.pop().ok_or(Error::Shape)?;let qkv=projected.pop().ok_or(Error::Shape)?;
        let a=self.vector(&name("ssm_a"),VH)?;let dt=self.vector(&name("ssm_dt.bias"),VH)?;
        let kernel=self.vector(&name("ssm_conv1d.weight"),4*10240)?;
        let gamma=self.vector(&name("ssm_norm.weight"),HD)?;
        let s=&mut self.states[layer];let mut conv=vec![0.;10240];
        for i in 0..10240 {let mut v=0.;for k in 0..4 {v+=kernel[i*4+k]*if k==3{qkv[i]}else{s.conv[k*10240+i]};}
            conv[i]=silu(v);}
        s.conv.copy_within(10240..3*10240,0);s.conv[2*10240..].copy_from_slice(&qkv);
        for h in 0..KH*2 {l2(&mut conv[h*HD..(h+1)*HD]);}
        let mut output=self.backend.recurrent(layer,self.position,&conv,&alpha,&beta,&a,&dt,&mut s.delta)?;
        for h in 0..VH {let normalized=norm(&output[h*HD..(h+1)*HD],&gamma,HD)?;
            for i in 0..HD{output[h*HD+i]=normalized[i]*sigmoid(z[h*HD+i]);}}
        self.linear(&name("ssm_out.weight"),None,&output,D)
    }
    fn attention(&mut self,layer:usize,x:&[f32])->Result<Vec<f32>> {
        let name=|s:&str|format!("blk.{layer}.{s}");
        let mut projected=self.projections(&[name("attn_q.weight"),name("attn_k.weight"),name("attn_v.weight")],x,&[12288,512,512])?;
        let value=projected.pop().ok_or(Error::Shape)?;let mut key=projected.pop().ok_or(Error::Shape)?;
        let full=projected.pop().ok_or(Error::Shape)?;
        let qnorm=self.vector(&name("attn_q_norm.weight"),256)?;let knorm=self.vector(&name("attn_k_norm.weight"),256)?;
        let mut query=vec![0.;6144];
        for h in 0..24 {query[h*256..(h+1)*256].copy_from_slice(&norm(&full[h*512..h*512+256],&qnorm,256)?);}
        for h in 0..2 {let nk=norm(&key[h*256..(h+1)*256],&knorm,256)?;key[h*256..(h+1)*256].copy_from_slice(&nk);}
        // Text positions are equal on all IMRoPE axes; partial rotary width64.
        for v in [&mut query,&mut key] {for head in v.chunks_exact_mut(256){for i in 0..32 {
            let theta=self.position as f32*libm::powf(10_000_000.,-((2*i)as f32)/64.);
            let (sn,cs)=libm::sincosf(theta);let a=head[i];let b=head[i+32];
            head[i]=a*cs-b*sn;head[i+32]=a*sn+b*cs;}}}
        let s=&mut self.states[layer];s.keys.extend_from_slice(&key);s.values.extend_from_slice(&value);
        let mut out=vec![0.;6144];
        for h in 0..24 {let kh=h/12;let mut scores=vec![0.;self.position+1];
            for p in 0..=self.position {
                let sum=super::frontier_cpu_math::dot(&query[h*256..(h+1)*256],&s.keys[p*512+kh*256..p*512+kh*256+256],1)
                    .unwrap_or_else(||{let mut sum=0.;for i in 0..256{sum+=query[h*256+i]*s.keys[p*512+kh*256+i];}sum});
                scores[p]=sum/16.;}
            softmax(&mut scores)?;
            for i in 0..256 {
                let sum=super::frontier_cpu_math::dot(&scores,&s.values[kh*256+i..],512)
                    .unwrap_or_else(||{let mut sum=0.;for p in 0..=self.position{sum+=scores[p]*s.values[p*512+kh*256+i];}sum});
                out[h*256+i]=sum*sigmoid(full[h*512+256+i]);}
        }
        self.linear(&name("attn_output.weight"),None,&out,D)
    }
    fn ffn(&mut self,layer:usize,x:&[f32])->Result<Vec<f32>> {
        let name=|s:&str|format!("blk.{layer}.{s}");
        let mut scores=self.linear(&name("ffn_gate_inp.weight"),None,x,512)?;softmax(&mut scores)?;
        let mut ids:Vec<usize>=(0..512).collect();ids.sort_unstable_by(|a,b|scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
        let sum:f32=ids[..10].iter().map(|i|scores[*i]).sum();if sum<=0.0{return Err(Error::NonFinite);}
        let gate=name("ffn_gate_exps.weight");let up=name("ffn_up_exps.weight");
        let shared_gate=name("ffn_gate_shexp.weight");let shared_up=name("ffn_up_shexp.weight");
        let shared_scale=name("ffn_gate_inp_shexp.weight");
        let mut requests=Vec::new();
        for &e in &ids[..10] {
            requests.push(LinearRequest{name:&gate,expert:Some(e),input:x});
            requests.push(LinearRequest{name:&up,expert:Some(e),input:x});
        }
        requests.push(LinearRequest{name:&shared_gate,expert:None,input:x});
        requests.push(LinearRequest{name:&shared_up,expert:None,input:x});
        requests.push(LinearRequest{name:&shared_scale,expert:None,input:x});
        let projected=self.backend.batch(&requests)?;
        if projected.len()!=23||projected[..22].iter().any(|v|v.len()!=FF)||projected[22].len()!=1{return Err(Error::Shape);}
        for v in &projected{finite(v)?;}
        let middle:Vec<Vec<f32>>=(0..11).map(|i|projected[i*2].iter().zip(&projected[i*2+1])
            .map(|(&g,&u)|silu(g)*u).collect()).collect();
        let down=name("ffn_down_exps.weight");let shared_down=name("ffn_down_shexp.weight");
        let mut requests=Vec::new();
        for (i,&e) in ids[..10].iter().enumerate(){requests.push(LinearRequest{name:&down,expert:Some(e),input:&middle[i]});}
        requests.push(LinearRequest{name:&shared_down,expert:None,input:&middle[10]});
        let values=self.backend.batch(&requests)?;
        if values.len()!=11||values.iter().any(|v|v.len()!=D){return Err(Error::Shape);}
        for v in &values{finite(v)?;}
        let mut out=vec![0.;D];
        for (e,&id) in ids[..10].iter().enumerate(){let weight=scores[id]/sum;
            if super::frontier_cpu_axpy::axpy(&mut out,&values[e],weight).is_none(){
                for i in 0..D{out[i]+=weight*values[e][i];}}}
        let weight=sigmoid(projected[22][0]);
        if super::frontier_cpu_axpy::axpy(&mut out,&values[10],weight).is_none(){
            for i in 0..D{out[i]+=weight*values[10][i];}}finite(&out)?;Ok(out)
    }
    pub fn token(&mut self,token:u32)->Result<Vec<f32>> {
        self.advance(token,true)
    }
    /// Prefill may omit its unused vocabulary head; all sequence state still
    /// advances identically. Decode and the final prompt token request logits.
    pub fn advance(&mut self,token:u32,want_logits:bool)->Result<Vec<f32>> {
        if token as usize>=VOCAB{return Err(Error::Shape);}
        if self.position>=LIMIT{return Err(Error::Context);}
        let mark=self.backend.phase_mark();
        let emb=self.backend.row("token_embd.weight",token as usize)?;if emb.len()!=D{return Err(Error::Shape);}
        let mut hidden=Vec::with_capacity(W);for _ in 0..HC{hidden.extend_from_slice(&emb);}
        self.backend.phase_end(4,mark);
        for layer in 0..48 {
            if layer==1{let mark=self.backend.phase_mark();self.ple(token,&mut hidden)?;self.backend.phase_end(3,mark);}
            let mark=self.backend.phase_mark();
            let (x,inj)=self.mix(&hidden,&format!("blk.{layer}.hc_attn"),true)?;
            self.backend.phase_end(0,mark);
            let mark=self.backend.phase_mark();
            let attn=if layer%4==3{self.attention(layer,&x)?}else{self.linear_attention(layer,&x)?};
            Self::combine(&mut hidden,&attn,&inj);
            self.backend.phase_end(1,mark);
            let mark=self.backend.phase_mark();
            let (x,inj)=self.mix(&hidden,&format!("blk.{layer}.hc_ffn"),true)?;
            self.backend.phase_end(0,mark);
            let mark=self.backend.phase_mark();
            let output=self.ffn(layer,&x)?;Self::combine(&mut hidden,&output,&inj);
            finite(&hidden)?;self.backend.trace(layer,"residual",&hidden);
            self.backend.phase_end(2,mark);
        }
        let mark=self.backend.phase_mark();
        let logits=if want_logits{
            let (x,_)=self.mix(&hidden,"output_hc",false)?;
            self.linear("output.weight",None,&x,VOCAB)?
        }else{Vec::new()};
        self.backend.phase_end(5,mark);
        self.previous=[Some(token),self.previous[0]];self.position+=1;Ok(logits)
    }
}

/// Autoregressive GDN recurrence, also checked against an independent GGML graph.
pub fn delta_step(conv:&[f32],alpha:&[f32],beta:&[f32],a:&[f32],dt:&[f32],state:&mut[f32])->Result<Vec<f32>> {
    if conv.len()!=10240||alpha.len()!=VH||beta.len()!=VH||a.len()!=VH||dt.len()!=VH||state.len()!=HD*HD*VH {return Err(Error::Shape);}
        let mut output=vec![0.;VH*HD];let qscale=1.0/sqrt(HD as f32);
        for h in 0..VH {let kh=h%KH;let decay=exp(a[h]*softplus(alpha[h]+dt[h]));let b=sigmoid(beta[h]);
            for v in 0..HD {let row=&mut state[(h*HD+v)*HD..(h*HD+v+1)*HD];let mut sk=0.;
                for k in 0..HD {row[k]*=decay;sk+=row[k]*conv[KH*HD+kh*HD+k];}
                let delta=(conv[2*KH*HD+h*HD+v]-sk)*b;let mut result=0.;
                for k in 0..HD {row[k]+=conv[KH*HD+kh*HD+k]*delta;result+=row[k]*(conv[kh*HD+k]*qscale);}
                output[h*HD+v]=result;}
        }
    finite(&output)?;Ok(output)
}
