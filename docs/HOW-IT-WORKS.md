# How it works: a custom GPU driver inside UEFI

Measured environment: UEFI under OVMF/KVM, RTX 3060 passed through with VFIO.
No guest OS or CUDA runtime; a Linux host exists. The custom Rust code is a driver.

## Bring-up and submission

PCI/MMIO access and NVIDIA's signed GSP firmware establish GPU control. Public
headers and open-driver code provide reference definitions. Register access
still matters; NVIDIA's proprietary driver is not the only possible implementation.

The implemented sequence establishes firmware communication, resource-manager
objects, a virtual address space and mappings, then a GPFIFO channel. Replies
are checked for matching sequence/function and actual status. Application logs
format those observations; compute additionally needs observed completion writes.

An eight-byte serialization offset caused channel INVALID_ARGUMENT. Logical
field-value tests did not expose the wire-layout error. Corrected replies
reported successful allocation and channel ID 3.

Command buffers/QMDs describe work; GPFIFO entries reference them. Doorbells
submit work and completion synchronization checks results. This is the needed
subset, not a general graphics/compute driver.

## Small-model execution

Weights, KV and activations reside in VRAM. Quantized linears, attention, norms,
RoPE and greedy kernels reduce CPU round trips. Decode uses DP4A; batched prefill
uses integer MMA. Q8 weights do not imply identical intermediate arithmetic
to every llama.cpp path.

The execution sequence is prebuilt rather than launched/waited per operation.
Dispatch, dependencies, memory traffic and synchronization still cost time.
CUDA graphs also reduce repeated launch costs:
[NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
Removing the OS was not isolated as the cause of performance.

Later prefill gains came from fragment-friendly Q8 weight/activation packing.
Quant values and scales are preserved; extra VRAM/setup buys faster timed loads.
Bigger tiles alone regressed.

## The large-model memory hierarchy

The 176.94B artifact is a mixed-quant MoE model with token-indexed lookup data.
Its three-shard backing store is about 90 decimal GB.

| Tier | Measured role |
|---|---|
| Always-on GPU allocations | About 5.35 decimal GB |
| GPU expert arena | 5 GiB |
| CPU expert-payload cap | 16 GiB within a 20 GiB guest |
| Backing storage / host cache | Shards, missing experts, sparse lookup rows |

This is not the full model resident in VRAM. The payload cap is not the whole
application's memory footprint. MoE choices depend on current activations;
prior tokens do not guarantee knowledge of future expert choices.

Verified path: **UEFI virtio BlockIO → CPU staging → GPU copies**.
Native SSD-to-GPU DMA has not been demonstrated. Guest read counters do not
directly measure physical SSD traffic because host caching intervenes.

The 48-layer forward combines GPU linears/GDN with CPU graph operations.
The latest measured improvement uses explicit SSE2 CPU accumulation with
operation order preserved. Warm throughput: 1.748 prefill / 1.912 decode tok/s.

## HTTP and safety

The host adapter tokenizes a request, binds it to an identity, runs the UEFI
evaluation and validates returned IDs. The MVP returns an answer after completion;
SSE keepalives are not live generated tokens.

Current state: buffered/direct model reads disagree, with recorded kernel
page faults. Requests return 503. Tokenizer and synthetic transport tests are
not accepted large-model serving.

No confidential-compute boundary, deterministic maximum clocks, free hardware
launches, physical boot-to-answer time, or public-service readiness is claimed.
Host, hypervisor, UEFI and signed GSP firmware remain system dependencies.
