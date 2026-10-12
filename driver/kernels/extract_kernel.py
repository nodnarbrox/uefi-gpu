"""Extract a CUDA ELF code section; reject relocation-bearing kernels."""
import argparse
import pathlib
import struct

p = argparse.ArgumentParser()
p.add_argument('cubin', type=pathlib.Path)
p.add_argument('output', type=pathlib.Path)
p.add_argument('--name', default='matmul4')
a = p.parse_args()
b = a.cubin.read_bytes()
assert b[:6] == b'\x7fELF\x02\x01'
shoff = struct.unpack_from('<Q', b, 40)[0]
entsz, count, names_index = struct.unpack_from('<HHH', b, 58)
headers = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * entsz) for i in range(count)]
nh = headers[names_index]
names = b[nh[4]:nh[4] + nh[5]]
code = None
a.output.with_suffix('.c2').write_bytes(b'')
for h in headers:
    name = names[h[0]:].split(b'\0', 1)[0].decode()
    print(name, 'size', h[5], 'info', hex(h[7]))
    if h[1] in (4, 9) and h[5]:
        target = headers[h[7]]
        target_name = names[target[0]:].split(b'\0', 1)[0]
        assert target_name.startswith(b'.debug'), 'runtime relocations need a loader'
    if name.startswith('.nv.constant2'):
        a.output.with_suffix('.c2').write_bytes(b[h[4]:h[4] + h[5]])
    if name == '.text.' + a.name:
        code = b[h[4]:h[4] + h[5]]
        print('KERNEL registers', h[7] >> 24)
assert code and len(code) <= 4096
a.output.write_bytes(code)
print('SASS_BYTES', len(code))
