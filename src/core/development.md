# src/core — development.md

## Goal
Engine core: inference loop (tokenize → forward → sample → detokenize),
configuration, logging and top-level orchestration of the model, streaming and
backend subsystems.

## Status
**Not started (P0 scaffold).** No sources yet.

## Scope
- In: engine context lifecycle, generation loop, sampling (greedy/top-p/temp),
  tokenizer integration, chat template handling.
- Out (for now): multimodal input pipeline (P6).

## Next steps
1. Define `fatcat_engine` context struct and config header in `include/fatcat/`.
2. Wire the generation loop once `src/model/` produces a forward pass (P2).
