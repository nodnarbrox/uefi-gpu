"""Offline NVRTC compilation only; never opens or executes a GPU."""
import ctypes as c, pathlib, sys, struct, os
# Public export: provide the path to the NVRTC shared library explicitly.
lib=c.CDLL(os.environ['NVRTC_LIBRARY'])
prog=c.c_void_p(); src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
def check(rc):
    if rc: raise RuntimeError(f'NVRTC error {rc}')
check(lib.nvrtcCreateProgram(c.byref(prog),src.read_bytes(),src.name.encode(),0,None,None))
opts=[b'--gpu-architecture=sm_86',b'--fmad=false',b'--std=c++14',b'--device-as-default-execution-space']
rc=lib.nvrtcCompileProgram(prog,len(opts),(c.c_char_p*len(opts))(*opts))
size=c.c_size_t();check(lib.nvrtcGetProgramLogSize(prog,c.byref(size)))
log=c.create_string_buffer(size.value);check(lib.nvrtcGetProgramLog(prog,log));print(log.value.decode())
check(rc);check(lib.nvrtcGetCUBINSize(prog,c.byref(size)))
buf=c.create_string_buffer(size.value);check(lib.nvrtcGetCUBIN(prog,buf));out.write_bytes(buf.raw)
check(lib.nvrtcDestroyProgram(c.byref(prog)))
# Extract independent text/constant sections; reject runtime relocations, local
# memory, or global data requiring an unimplemented ELF loader.
b=buf.raw;shoff=struct.unpack_from('<Q',b,40)[0];ent,n,ni=struct.unpack_from('<HHH',b,58)
hs=[struct.unpack_from('<IIQQQQIIQQ',b,shoff+i*ent)for i in range(n)]
names=b[hs[ni][4]:hs[ni][4]+hs[ni][5]]
def name(h):return names[h[0]:].split(b'\0',1)[0].decode()
for h in hs:
    if h[1]in(4,9)and h[5]:assert name(hs[h[7]]).startswith('.debug'),name(h)
    if name(h).startswith('.text.'):
        kernel=name(h)[6:]; path=out.parent/kernel
        path.with_suffix('.sass').write_bytes(b[h[4]:h[4]+h[5]])
        cs=[s for s in hs if name(s)=='.nv.constant2.'+kernel]
        path.with_suffix('.c2').write_bytes(b[cs[0][4]:cs[0][4]+cs[0][5]]if cs else b'')
        print('KERNEL',kernel,'BYTES',h[5],'REGISTERS',h[7]>>24,'C2',cs[0][5]if cs else 0)
