# Evidence: exact records and interpretation

Selected evidence, not all historical logs. Archived implementation source is in [driver/src](../driver/src/). Hashes support
identity/recomputation, not independent hardware attestation. PTIMER records
are printed by the UEFI application, not by a GPU serial port.

## Archived 177B verification

The offline exporter checked nine checkpoint artifacts against their manifest,
including EFI, status, serial/run logs and case definitions. It decoded recorded
IDs with the pinned tokenizer and checked prompt IDs. No GPU was run.

- EFI SHA256:
  `a8fd3d44abb499cd0f7eaba4a09681219c984cd7e44a34887c1354945f8a9c2e`
- Full serial SHA256:
  `6d39b46f15f36a8216d75d10fede09c0ef2b271b8f7cd8b5bbd915f611262391`
- [Exact records, prompts, IDs, line numbers and hashes](frontier-177b-ptimer.json).

The serial buffer is replayed. Eight occurrences are **four unique case/round
records**, not eight trials or four independent boots. Occurrence line numbers
are retained; only identical records are deduplicated.

## Exact channel/schedule replies from that archive

```text
*** M3 GPFIFO reply: matched=true seq=13 fn=103 rpc_result=0x00000000 alloc_status=0x00000000 len=0x1b0 cid=3 ***
*** M3 COMPUTE_SCHEDULE reply: matched=true seq=23 fn=76 cmd=0xa06f0103 rpc_result=0x00000000 ctrl_status=0x00000000 len=0x3a ***
```

Application-formatted GSP response fields. Matching logical input fields did
not mean matching wire bytes in the earlier failed allocation.

## Exact completion markers, caveat preserved

```text
FRONTIER_EVAL_REPEAT case=sentence generated_ids_equal=true
FRONTIER_EVAL_REPEAT case=riddle generated_ids_equal=true
FRONTIER_EVAL_COMPLETE cases=2 measured_rounds=2 correctness=HOST_REVIEW_REQUIRED
*** FRONTIER_FULL_FORWARD completed=true correctness=UNVERIFIED ***
```

The application did not claim general correctness. Subsequent pinned host
detokenization/review established the displayed regression outputs. Repeatability
alone does not prove arbitrary-prompt correctness; operator oracles are not
model-wide CUDA logit parity.

## Exact warm timing arithmetic

| Case | Prefill tokens / ns | Decode calls / ns |
|---|---|---|
| Sentence | 24 / 11,693,000,672 | 20 / 9,264,056,544 |
| Riddle | 40 / 24,926,780,960 | 6 / 4,337,297,440 |
| Pooled | 64 / 36,619,781,632 | 26 / 13,601,353,984 |

`tokens * 1e9 / nanoseconds` gives **1.747689 / 1.911574 tok/s**.
Serial writes were outside timing. Loading/HTTP latency excluded.
Slower warming rounds are retained in the JSON.

## Live serving acceptance: not achieved

A later actual large-model POST returned HTTP 503. Selected response fields:

```json
{"error":{"type":"host_unavailable","code":"host_unavailable"}}
```

Not model output. The private full response also names the integrity hold.
Buffered/direct disagreement persists after two logged pages were marked
HWPOISON; the second hard-offline write returned EBUSY, recovery Ignored.
A matching direct-read hash for one range is not proof that all storage is
healthy or that a DIMM is the unique cause.

Other historical comparisons and fleet reports have separate provenance
limits in [BENCHMARKS](../docs/BENCHMARKS.md). They are not disguised as raw
serial excerpts. Unverified register interpretations, prose posing as logs,
and physical memtest claims have been removed.
