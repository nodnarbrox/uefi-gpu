# Skills demonstrated and boundaries

AI-assisted engineering: direction, implementation review, measurement and
debugging—not a claim that every line was hand-written.

| Area | Concrete evidence |
|---|---|
| Rust / UEFI | Custom driver and inference app targeting x86_64-unknown-uefi |
| GPU systems | GSP RPC, VRAM mappings, GPFIFO/QMD submission and completion |
| Protocol debugging | Eight-byte serialization fault; input variants and RM-log decoding |
| Inference performance | Weight residency, GPU operations/fusion, DP4A, MMA and lossless layout experiments |
| Measurement | Warm/cold cohorts, precision caveats, PTIMER counters, rejected regressions |
| MoE memory management | 176.94B generation with bounded CPU/VRAM caches and storage-backed lookup |
| Linux / virtualization | KVM/VFIO, systemd, journal and buffered/direct-read forensics |
| Python | Evaluation, HTTP adapters, tokenization/validation and evidence extraction |
| Cross-platform | Related Dart/FFI worker/coordinator work; fleet results separately attributed |
| Firmware networking | DHCP/HTTP memory-test reporting verified in QEMU/OVMF |
| Reliability | Fail-closed admission when actual bytes failed; no silent substitute |

## Boundaries

- Linux host/hypervisor present. No guest OS/CUDA runtime is accurate; no driver
  anywhere is not.
- Specialized driver, not every NVIDIA graphics/production subsystem.
- Studying CUDA kernels is not operating vLLM/TensorRT-LLM in production.
- Large-model warm benchmark: 1.748 prefill / 1.912 decode tok/s.
  Large-model HTTP serving has not passed hardware acceptance.
- Expert caching implemented; guaranteed layer-ahead prediction, native
  SSD-to-GPU DMA and persistent-megakernel gains not demonstrated.
- Checkpoint hashes are provenance, not proven byte-identical public builds,
  independent replication or audited security.
- No profitable economics, reliable WAN expert streaming or provider SLA established.

## Résumé-ready statements

- Directed and validated an AI-assisted Rust UEFI driver that runs compute on
  an RTX 3060 using signed GSP firmware, without a guest OS or CUDA runtime.
- Improved Qwen2.5-1.5B decode from roughly 2.08 to 164 tok/s; matched pinned
  stock llama.cpp decode at Q8 weight parity.
- Implemented storage-backed 176.94B inference with bounded expert caches;
  verified two repeatable regressions and measured warm throughput.
- Diagnosed an eight-byte request-layout error and a later host-integrity failure;
  retained a fail-closed gate rather than claiming deployment from synthetic tests.
