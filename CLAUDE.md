# moment-miner — working notes

Architecture, benchmarks, labeling format: see README. Roadmap decisions live
in Valya's session memory, not here.

## Rules

- Dev is Docker-only: the WSL host has no ffmpeg, GPU, or Python env. Use
  `make test` / `make index` / `make serve` (Makefile wraps all docker runs).
- Named volumes: `mm-hf-cache` (model weights), `mm_data` (indexes). Deleting
  one means re-download / re-index.
- `examples/` (git-ignored): Valya's real Canon 4K clips + `labels.csv` hand
  labels. Use it for real-footage smoke tests; never commit footage.
- `refs/` (git-ignored) and `/proj/lossless-cut`: cloned reference repos,
  read-only.
- Valya runs `git commit` / `git push` herself — never commit.
- Embedding backends stay pluggable behind `embeddings/base.py`; vectors must
  be L2-normalized (search ranks by L2 assuming it equals cosine).
- After any search-quality change, run `mm eval` on `examples/labels.csv`
  (once labels exist) — no shipping on vibes.
- `.llc` files reference media relative to their own location (LosslessCut
  joins dirname(project) + mediaFileName) — relative paths are intentional.
- Whisper output is filtered (no_speech_prob / avg_logprob) because it
  hallucinates captions on music; don't remove that filter without eval.
- Prefer an existing library over hand-rolled code, and let it be the only
  implementation. Cutting is `smartcut` (MIT, pinned `==1.7`, optional
  extra): deprecated upstream 2026-02, so if PyAV ever breaks it, vendor
  the four modules used — do not reimplement with raw ffmpeg calls.
- Verify cut accuracy by pixel alignment on the copied interior, never by
  duration alone: a clip can be several frames out and still have a
  plausible duration. Test at 50 fps; the 15 fps fixture hides it.
