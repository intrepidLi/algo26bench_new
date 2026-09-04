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
PYTHONPATH=src python -m algo26bench.bench.run --config configs/onetrans_smoke.json
```

Each command uses deterministic synthetic data and writes no checkpoint unless
`output_dir` is set in the training configuration.
