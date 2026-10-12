# Source snapshot and build inputs

The `driver/` tree is the archived `GOOD-frontier-axpy-1p912toks-20260908`
inference checkpoint. Rust source files are copied byte for byte. This is a
research snapshot with hardware-specific assumptions, not a turnkey portable
driver. The USB release, when present, is a separately identified archived boot
artifact; it is not the EFI that produced the 177B benchmark.

## Contents

- Rust NVIDIA bring-up, GSP communication, GPU memory/command submission,
  GGUF parsing, quantization, expert caches and inference.
- CUDA/PTX kernel sources and the checkpoint's extracted `.sass`/`.c2` files.
  These extracted kernels are project compute code, not the NVIDIA GSP firmware.
- Cargo dependency lockfile and input-only prompt fixtures.
- Public benchmark writeup and selected historical evidence.

## External inputs

Signed GPU firmware and model-derived fixtures are not included in the source
tree. [external-inputs.json](../external-inputs.json) enumerates each missing
`include_bytes!` input, its retained checkpoint hash/size where available and
the referencing modules. Files under `reference/` sit alongside `driver/`.
Obtain the matching model/tokenizer and generate these fixture inputs yourself;
empty files or substitute weights are not valid inputs.

The checkpoint embeds a signed booter and the 570.144 GA102-family GSP
bootloader. NVIDIA's [official extraction tool and documentation](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/570.144/nouveau)
provide firmware provenance. Additional full GSP/model runtime disks were
supplied by the original host. Retain applicable NVIDIA terms when obtaining
or redistributing firmware; firmware is not covered by a project source license.

With the exact external inputs installed, use a Rust toolchain with the
`x86_64-unknown-uefi` target, then run from `driver/`:

```sh
cargo build --locked --release --target x86_64-unknown-uefi
```

The original checkpoint uses Rust 1.91.0. A clean public checkout alone cannot
build the full EFI until these external inputs are supplied. No claim of a
fresh GPU run or independently reproduced public build accompanies publication.

## Offline checks and kernel generation

Existing platform-independent tests can run without firmware or model disks:

```sh
rustc --edition=2021 --test driver/src/frontier_core.rs -o core-tests
./core-tests
rustc --edition=2021 --test driver/src/frontier_cache.rs -o cache-tests
./cache-tests
```

For offline CUDA kernel generation, set `NVRTC_LIBRARY` to the local NVRTC
shared library path and pass input `.cu` and output `.cubin` paths to
`driver/kernels/compile_ops.py`. The helper targets `sm_86`, extracts independent
text/constants and rejects unsupported relocation/data sections. Compiler version
and register counts matter for this driver; recompilation is not automatically
equivalent to the archived kernels. Its private absolute path was replaced with
an environment setting for publication. `prepare_scale.py` similarly uses its
own directory instead of the original host path; neither change alters runtime
driver/inference code.

## Attribution

This is AI-assisted engineering directed and validated by Brandon Davis.
Public NVIDIA interfaces and open driver code informed the work. Preserve the
Sun Microsystems notice in `frontier_cpu_expf.rs`. GGML-derived quantization
tables/layouts retain their upstream notice in `third_party/LICENSE.ggml`.
Dependency licenses remain with their respective authors. Publication does not
assign a new blanket license to third-party firmware, model data or dependencies.
