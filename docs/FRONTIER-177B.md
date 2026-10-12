# 176.94B-model generation: measured, with limits

Recovered from the successful September 8, 2026, 09:28 EDT evaluation archive.
This is **not a new run after the host-memory intervention**, and not an HTTP
chat success. Model label: Qwen3.8-Flash-Next UD-Q3_K_XL; 176.94B total parameters,
89,986,353,824 bytes in three shards. Mixed quantization, not uniform Q3 or Q8.

## Actual generation

Prompt:

> In one short sentence, explain why the sky appears blue.

Recorded IDs, decoded with the pinned tokenizer:

> The sky appears blue because air molecules scatter shorter blue wavelengths of sunlight more effectively than longer red wavelengths.

Prompt:

> I am a number. If you double me and add two, you get twelve. What number am I? Answer in one short sentence.

Recorded output:

> You are the number five.

Both reached EOS and reproduced identical IDs in the warming and warm rounds.
One probe, two cases, two rounds each. The application's mirrored serial replay
does not create additional trials. Host token decoding and semantic review
establish these regressions, not arbitrary-prompt quality or full CUDA logit parity.

## PTIMER benchmark

| Phase / case | Prompt tokens | Prefill tok/s | Decode calls | Decode tok/s |
|---|---:|---:|---:|---:|
| Warming / sentence | 24 | 0.572 | 20 | 0.363 |
| Warm / sentence | 24 | 2.053 | 20 | 2.159 |
| Warming / riddle | 40 | 0.603 | 6 | 0.617 |
| Warm / riddle | 40 | 1.605 | 6 | 1.383 |
| **Pooled warm** | **64** | **1.748** | **26** | **1.912** |

Exact pooled warm arithmetic:

```
prefill = 64 * 1e9 / 36,619,781,632 = 1.7476892856 tok/s
decode  = 26 * 1e9 / 13,601,353,984 = 1.9115743940 tok/s
```

The first output prediction belongs to prefill. Subsequent decode calls include
the terminal EOS. Prefill is sequential with only the final output-head evaluation.
No serial writes inside the timed phases. Building, model preparation/loading,
warmup, teardown and HTTP latency are not included in the warm result.

This is about **523 ms per warm decode call**, not a fresh-chat latency guarantee.
Short prompts, one measured round per case, retained caches, no confidence interval
or long-context soak. Clocks were not locked; a finite performance-boost request
was accepted and cleared during the archived run.

## Memory and storage

| Resource | Allocation / cap |
|---|---:|
| Always-on GPU allocations | 5,352,296,448 bytes |
| GPU expert cache arena | 5,368,709,120 bytes (5 GiB) |
| CPU expert-payload cap | 17,179,869,184 bytes (16 GiB) |
| Guest RAM | 20 GiB |
| Complete model backing | 89,986,353,824 bytes |

The payload cap excludes other guest memory. GPU arena allocation is not identical
to occupied payload. The whole model is not resident on the 12 GB card, and the
host's page cache contributes additional caching beyond the guest budget.

The logged warm sentence CPU/GPU expert payloads were 16,316,979,200 /
3,696,640,000 bytes; warm riddle: 16,655,436,800 / 3,732,403,200 bytes.
Warm CPU expert misses were zero; GPU expert evictions/copies still occurred.

Guest BlockIO transferred about 7.438 / 7.368 decimal GB during the two warming
phases, versus 563,712 / 587,776 bytes in the respective warm phases. These are
guest/backend counters, **not a physical SSD-throughput measurement**. Host caching
can satisfy reads. The path is UEFI virtio BlockIO → CPU staging → GPU copies;
direct native SSD-to-GPU DMA and fully hidden expert prefetch are not demonstrated.

## What changed in this checkpoint

Explicit SSE2 CPU expert-weighted accumulation and residual injection replaced
software-float scalar loops while preserving operation order. The recorded prior
checkpoint pooled 1.591447 prefill / 1.738649 decode; this one recorded
1.747689 / 1.911574: about 9.82% / 9.95% improvement. GPU kernels and quantization
were unchanged in that comparison. It is an observed short-workload gain, not a
confidence-bounded general speedup.

## Comparison and current product state

The reported community 24–26 tok/s is **unmatched and not independently replicated
on this configuration**. Our warm result is about 12.6–13.6 times lower numerically.
No successful matched 177B CUDA baseline exists; no claim of beating that fork.

A large-model HTTP adapter is deployed but admission remains blocked. Actual
buffered/direct model-read disagreement persisted after two logged pages were
marked HWPOISON. A real request returned 503 rather than a model reply. The gate
was not relaxed, and no substitute model or archived answer was served as live.

The adapter's intended MVP is fresh evaluation per request and buffered response
publication; SSE keepalives are not incremental token generation. Its TTFT and
fresh-request throughput remain unmeasured. This is not production-ready chat.

## Evidence and what can be checked

[Machine-readable evidence](../evidence/frontier-177b-ptimer.json) contains all
four unique timing records, their raw lines/occurrence numbers, exact prompts and
IDs, decoded outputs, tokenizer hashes, selected artifact hashes and pooled math.
The offline exporter validated nine artifacts against the retained checkpoint
manifest, checked prompt IDs and decoded the output with the pinned tokenizer.
It does not rehash the 90 GB model or rerun the GPU tonight.

The archived raw completion still says `correctness=UNVERIFIED` /
`HOST_REVIEW_REQUIRED`; the host review is explicitly separate. Hashes establish
artifact identity, not public reproducibility or independent hardware attestation.
The archived driver/inference implementation is available in [driver/src](../driver/src/). Fleet coordinator code is not included. External build inputs are documented in [BUILD](BUILD.md).
