# Benchmarks: measurements, cohorts and limits

Updated September 8, 2026. RTX 3060 12 GB, verified PCI identity 10de:2487.
UEFI runs use OVMF/KVM passthrough, a custom guest driver and signed GSP firmware.
UEFI inference uses PTIMER; llama.cpp uses synchronized host timers; HTTP tests
use wall time. Loading and end-to-end chat latency are separate. Clocks unlocked.

## Historical small-model progression

Qwen2.5-1.5B, Q8 weights:

| Cumulative checkpoint | Decode tok/s |
|---|---:|
| Initial system-RAM weight path | ~2.08 |
| Weights uploaded once to VRAM | 4.405 |
| GPU attention/norms/KV plus fusion | ~114 |
| Later quantized-kernel / command-stream path | ~164 |

About 79x over the initial implementation, not an isolated causal measurement
for every technique. Both weight traffic and later CPU/launch costs mattered.

## Same Q8 weights, same GPU: pinned llama.cpp comparison

Identical Qwen2.5-1.5B Q8_0 GGUF. llama.cpp commit
`22cae832188a1f08d18bd0a707a4ba5cd03c7349`.

| Original comparison cohort | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| UEFI: three post-Linux controls | 1786.483 | 164.097 |
| Stock llama-bench: ten repetitions/test | 3420.628 | 164.039 |
| Matched application loop: five repetitions/prompt | 3412.414 | 152.906 |

Decode ties stock: a 0.036% difference is not meaningful. UEFI is 7.32% ahead of
this particular matched application loop, not every CUDA deployment.

Method differences remain:

- Original prompts: 28/53 tokens; 15/9 decode steps including final EOS prediction.
  Rates pool tokens divided by pooled time.
- UEFI: block-32 INT8 activation quantization, DP4A decode, INT8 MMA prefill,
  FP32 attention/KV, GPU greedy.
- Matched llama.cpp: exact prompts, batched last-logit-only prefill, FP32 KV,
  flash attention disabled, CPU greedy, one CPU thread, full GPU offload.
- Stock: synthetic benchmark tokens, FP16 KV, flash attention enabled, no sampling.
  [Pinned benchmark documentation](https://github.com/ggml-org/llama.cpp/blob/22cae832188a1f08d18bd0a707a4ba5cd03c7349/tools/llama-bench/README.md).
- OS/CUDA/GSP versions and kernels differ. This does not isolate OS overhead.

**Initial variability retained:** the first pre-Linux probe decoded at
46.395/45.719 tok/s. The complete three-probe pre-Linux cohort was
674.330 prefill / 88.495 decode. The later control cohort is explicitly identified,
not presented as all runs. The slowdown's cause was not established.

## Later pure-Q8 layout optimization

| Build | Pooled prefill tok/s | Pooled decode tok/s |
|---|---:|---:|
| Later weight/activation fragment layout | **3255.338** | **164.218** |
| Earlier stock comparison, for context | 3420.628 | 164.039 |

Three frozen-EFI probes, pooling 243 prompt tokens and 72 decode steps.
Individual prefill: 3246.478 / 3257.135 / 3262.441; decode:
164.056 / 164.260 / 164.337. Sentence/five-riddle regressions passed.

About **4.8% behind** historical stock prefill, effectively tied decode.
Stock was not rerun alongside every later candidate. Individual 53/139-token
rates around 3654/4147 do not establish a pooled apples-to-apples win.
Some were pure Q8; dismissing them all as mixed-Q4 would also be wrong.

Lossless packing adds about 1.47 GB of weights; setup/repacking/readback is
excluded. Larger shared tiles and several vector-load variants regressed.
No cp.async or persistent-kernel speedup was demonstrated.

## Size curve: one generalized Q8 backend

Three probes per model, same 28/53-token inputs. Output lengths differ.
This is a different generalized checkpoint, not either 1.5B control above.

| Qwen2.5 model | Prefill tok/s | Decode tok/s | Original short riddle |
|---|---:|---:|---|
| 0.5B Q8 | 5012.145 | 365.268 | 6 — wrong |
| 1.5B Q8 | 1812.939 | 164.317 | five — correct |
| 3B Q8 | 653.953 | 88.636 | 10 — wrong |

Same-GGUF greedy llama.cpp on CPU **and CUDA** also answered 10 for the exact
3B short prompt. With a different reasoning instruction, UEFI and CUDA reached
5. The complete UEFI reasoning ended at EOS after 191 tokens: separately,
719.580 prefill / 84.395 decode. This resolves that specific concern as shared
model/prompt behavior, not proof of universal numerical correctness.
The 0.5B wrong short answer also reproduced in the reference.

## Full 176.94B mixed-quant model

**Headline: measured warm prefill 1.748 tok/s; decode 1.912 tok/s on the full
176.94B model itself.** Source is the successful September 8, 09:28 EDT archived
real-GPU evaluation, revalidated offline—not a fresh overnight run or live chat.
Later host-integrity failures prohibit treating this as current serving readiness.

| Archived warm case | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Sentence, 24 prompt / 20 decode calls | 2.052510 | 2.158881 |
| Riddle, 40 prompt / 6 decode calls | 1.604700 | 1.383350 |
| **Pooled** | **1.747689** | **1.911574** |

Different architecture, precision and residency: not a fourth point in the
controlled dense-Q8 curve. CPU/VRAM expert caches plus SSD backing; warm CPU
expert misses zero. First/warming-round decode: 0.363 sentence / 0.617 riddle.
Build/load and HTTP latency are excluded even from those timings.

No successful matched 177B CUDA baseline exists. Community 24–26 tok/s is an
external unmatched target, **not our measurement**.
[Details](FRONTIER-177B.md); [exact evidence](../evidence/frontier-177b-ptimer.json).

## Separately reported fleet experiments

These figures came from related fleet work. Its raw per-request traces are not
in this package and are **not independently reverified here**.

Qwen2.5-0.5B-Q4, reported 0.3 ms LAN:

| Layout | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Solo | 248 | 55.8 |
| Two-node RPC layer split | 139 | 42.5 |

That split lost: 0.56x prefill / 0.76x decode. This does not mean every layer split
incurs a hop for each layer or cannot help capacity on other interconnects.

Reported per-worker decode: M40 CUDA 161, Apple Silicon Metal 110, CPU VM 57,
Intel MacBook CPU 39 tok/s. Rounded rates sum to **367**, approximately the
reported 368, or 2.3x the fastest. A sum of separate rates is **nominal capacity**,
not proof of concurrent pool throughput.

A separately reported 60-request test gave 4.37 / 5.05 / 5.10 / 5.11 req/s as
weaker workers were added; another two-worker test reported 1.56x. These cannot
be combined into one scaling law. Slower workers can add capacity; routing and
tail latency determine user benefit.

## Provenance

Private records include the original Q8 report/results, later layout experiments
and size-curve logs. This is not their complete public reproduction bundle.
The 177B JSON exposes exact numerators, denominators, IDs, prompts, source line
numbers and artifact hashes. Hashes do not replace independent hardware replication.
