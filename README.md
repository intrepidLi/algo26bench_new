# algo26bench v2

`algo26bench_v2` is a compact, fidelity-aware research harness for comparing
industrial ranking architectures under a canonical academic protocol.  It
implements the algorithmic computation graph; production-only kernels,
serving caches, quantization, MFU claims, and proprietary scheduling are out
of scope unless a recipe explicitly says otherwise.

The first release contains two recipes:

- **HyFormer**: semantic non-sequential tokens, independent multi-sequence
  query generation/decoding, layer-wise sequence encoding, and parameter-free
  RankMixer-style query boosting with per-token FFNs.
- **OneTrans**: timestamp-aware or ordered multi-sequence fusion, Auto-Split
  non-sequential tokenization, RMS pre-normalization, mixed shared/token-specific
  causal attention and FFNs, and tail-query pyramid reduction.

## Design boundaries

- Datasets expose raw categorical IDs, dense values, timestamps, labels, and
  stable sample IDs.  They do not create learned embeddings.
- Every recipe owns its batch layout and collator.  The engine only consumes
  the recipe's objective and metric packets.
- `True` always means a valid token.
- Fidelity is reported per recipe and per run; omitted systems mechanisms are
  never presented as reproduced results.

## Quick smoke run

```bash
PYTHONPATH=src python -m algo26bench.bench.run --config configs/hyformer_smoke.json
PYTHONPATH=src python -m algo26bench.bench.run --config configs/hyformer_longer_smoke.json
PYTHONPATH=src python -m algo26bench.bench.run --config configs/onetrans_smoke.json
```

HyFormer's `sequence_encoder` selects among the three encoding strategies of the
paper's Sec. 3.4.1: `transformer` (Eq. 5), `longer` (Eq. 6, compressing the
sequence onto its most recent `num_short_tokens` behaviours) and `swiglu`
(Eq. 7). The fidelity card names whichever one is active.

Each synthetic command writes no checkpoint unless `output_dir` is set in the
training configuration.

## PCVR (academic_2000w)

The real TAAC2026 parquet is read by `algo26bench.data.pcvr`.  It reuses the
`algo26bench` `schema.json` layout, but emits `RawExample` rows: original
timestamps are kept, sequences stay ragged, and padding is left to the recipe
collator.  Truncation keeps the **most recent** `max_len` events (legacy
`algo26bench` kept the head).

`d_model` for HyFormer must be divisible by `4 * num_queries + num_ns_tokens`.
With the default user/item grouping plus one dense token that is 7, so the
smoke config uses `d_model=56`.

```bash
PYTHONPATH=src python -m algo26bench.bench.run --config configs/hyformer_pcvr_smoke.json
PYTHONPATH=src python -m algo26bench.bench.run --config configs/onetrans_pcvr_smoke.json
```

Point `data.data_dir` at a `native_parquet_v2` directory that contains
`schema.json` and `*.parquet`.  The default path is the same
`academic_2000w/train/native_parquet_v2` tree used by `algo26bench/run.sh`.
`train_row_groups` selects an evenly spaced training subset the same way as
the legacy loader.
