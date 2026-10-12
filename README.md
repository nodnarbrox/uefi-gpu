# A 177B model generating on an RTX 3060 from a UEFI guest

I directed and validated an AI-assisted systems project that implements a minimal
Rust NVIDIA GPU driver and LLM runtime in UEFI, using signed, unmodified GSP firmware.

**Scope up front:** the measured runs use an RTX 3060 12 GB passed through under
Linux/KVM/VFIO. There is no guest OS, CUDA runtime or NVIDIA OS driver inside the
guest. There is a custom driver, UEFI firmware, GPU firmware and a Linux host.
This is not a verified physical-platform inference boot.

## Source and downloads

- [Driver and inference source](driver/src/)
- [Kernel sources and extracted kernels](driver/kernels/)
- [Build inputs and provenance](docs/BUILD.md)
- [USB image releases](https://github.com/nodnarbrox/uefi-gpu/releases)

USB artifacts are labeled separately from the later inference checkpoint.

## Results worth examining

| Result | Observed outcome | Limit |
|---|---|---|
| GPU bring-up | GSP RPC, GPU memory, command-channel allocation and compute completion | Necessary subset, not NVIDIA's complete driver |
| Qwen2.5-1.5B Q8 decode | About 164 tok/s, effectively tied with pinned stock llama.cpp | Short context, unlocked clocks |
| Later Q8 prefill optimization | 3,255 tok/s pooled versus historical stock 3,421 | About 4.8% behind, not a prefill win |
| Full 176.94B model | Coherent sentence and correct five-riddle; repeatable IDs | Two regression cases, not general quality evaluation |
| Large-model warm throughput | **1.748 prefill / 1.912 decode tok/s** | Cached evaluation, not fresh-request speed |
| Large-model HTTP chat | Adapter deployed; deliberately returns unavailable | Host-integrity failure unresolved |

The large artifact is Qwen3.8-Flash-Next UD-Q3_K_XL: **89,986,353,824 bytes across
three shards**, mixed quantization. “120B” was a project nickname; 176.94B is the
recorded parameter count. Size alone does not establish frontier-level quality.

The archived GPU evaluation generated:

> The sky appears blue because air molecules scatter shorter blue wavelengths of sunlight more effectively than longer red wavelengths.

For the arithmetic riddle:

> You are the number five.

These are **archived evaluation outputs, not a successful chat exchange**.
See [the full result](docs/FRONTIER-177B.md) and
[exact counters, IDs and hashes](evidence/frontier-177b-ptimer.json).

## What I built and learned

The custom driver establishes GSP communication, resource-management objects,
GPU virtual mappings and a GPFIFO command channel. Public NVIDIA headers and
open-driver implementations informed the work; it was not implemented without
reference material or through an exploit.

A persistent channel INVALID_ARGUMENT was traced to an **eight-byte serialization
offset** using input variants and GSP RM-log decoding. Matching logical values
was not matching the wire layout. Real replies confirmed the corrected
allocation and scheduling.

Small-model decode progressed roughly **2.08 → 4.4 → 114 → 164 tok/s** through
VRAM weight residency, GPU attention/norms/KV and fusion, quantized kernels and
command-stream refinements. These are cumulative historical checkpoints, not
controlled attribution of each gain to one change.

Later Q8 prefill gains came from **lossless weight-fragment packing and activation
layout**. Larger tiles and several vector-load experiments regressed and were
rejected. The whole prefill stream was already submitted as one prebuilt entry.

Direct ring control removes this implementation's CUDA API calls, but does not
make GPU scheduling free or establish an exclusive capability. CUDA graphs
also reduce repeated host launch costs.
[NVIDIA CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## How the large model runs with 12 GB VRAM

Always-on weights and bounded expert caches occupy the GPU. Additional hot
experts live in guest CPU RAM; backing weights and token-indexed lookup rows
remain on storage. The measured configuration uses a 20 GiB guest, a 16 GiB CPU
expert-payload cap, and a 5 GiB GPU expert arena in addition to always-on allocations.
Host page cache matters too.

**Verified storage path: UEFI BlockIO → CPU staging → GPU copies.** It is not a
demonstrated native SSD-to-GPU DMA path. MoE routing depends on intermediate
activations; it must not be confused with token-indexed lookup rows or
advertised as guaranteed layer-ahead expert prediction.

## Knowing when not to ship

Later model checks found buffered and O_DIRECT reads of the same range
disagreeing; direct reads matched its reference hash. Kernel bad-page warnings
and an Oops were also recorded. This establishes an unsafe host/data path,
not a uniquely diagnosed DIMM or proof that all storage is healthy.

Marking two logged pages HWPOISON did **not** repair that disagreement. An actual
chat request returned 503, with no fabricated answer or silent small-model
substitution. Verified recovery remains necessary.

A separate UEFI memory tester with DHCP/HTTP reporting was built and tested in
QEMU/OVMF. **No physical-host memory-test result is claimed here.**

## Distributed work and evidence limits

Related fleet tests favored independent whole-request routing over a particular
two-node layer split. That supports the chosen architecture, not a theorem that
model parallelism always loses. Whole-model workers increase aggregate capacity;
they do not pool VRAM for one larger model or automatically accelerate one request.
Fleet figures and provenance limits are separated in [BENCHMARKS](docs/BENCHMARKS.md).

This was AI-assisted engineering: I made architectural decisions, directed coding
agents, reviewed failures and required probe evidence. I do not claim to have
hand-written every line. The archived driver and inference source is now available in [driver/](driver/).

This release includes the archived Rust source, kernel sources and extracted kernels, plus the evidence/writeup package. **External firmware and model fixtures are required to build the full inference checkpoint**; see [BUILD](docs/BUILD.md).
Hashes establish identity, not independent attestation of a physical run. The
177B extraction verifies selected archived files, raw timing records, mirrored-log
deduplication, pinned-tokenizer decoding and regression text. Readers can
recompute the rates.

- [Benchmarks and comparison methodology](docs/BENCHMARKS.md)
- [177B result and serving limitations](docs/FRONTIER-177B.md)
- [Technical overview](docs/HOW-IT-WORKS.md)
- [Evidence and verification limits](evidence/serial-logs.md)
- [Skills demonstrated](SKILLS.md)

The published driver snapshot includes expert caching and CPU pooling. Fleet coordinator code, credentials and infrastructure addresses are excluded. No audited production security,
profitable economics or public-provider readiness is claimed.
