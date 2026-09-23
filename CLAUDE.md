# moment-miner — working notes

Architecture, benchmarks, labeling format: see README. Roadmap decisions live in Valya's session memory, not here.

## Rules

- Dev is Docker-only: the WSL host has no ffmpeg, GPU, or Python env. Use `make test` / `make index` / `make serve` (Makefile wraps all docker runs).
- Named volumes: `mm-hf-cache` (model weights), `mm_data` (indexes). Deleting one means re-download / re-index.
- `footage/` (git-ignored): Valya's real Canon 4K clips under `footage/src/`, hand labels at `footage/labels.csv` (12 filled rows, local videos only) and `footage/labels-all.csv` (198 rows harvested from `.llc` projects, of which only 14 name a video present here — measured 2026-09-09, and their durations match `labels.csv`, so it adds ~2 usable rows, not 186). Use it for real-footage smoke tests; never commit footage.
- **Labels are edit points, not action extents.** Every row is padded on purpose, so a 12.1 s `pull-ups` row wraps a ~3–4 s pull-up. Fine for `mm eval`, whose hit rule is overlap-based; useless for judging whether a cut is tight. Anything measuring `mm locate` needs a separate file of tight boundaries — never add them to `labels.csv`, which every recorded number depends on staying comparable.
- `examples/` (git-ignored) is **empty on purpose** — reserved for exemplar clips, one folder per label (`examples/kong_vault/clip1.mp4`). Exemplars must never cover a moment that `labels.csv` also marks, or a with-exemplars run scores higher for free.
- `refs/` (git-ignored) and `/proj/lossless-cut`: cloned reference repos, read-only.
- Embedding backends stay pluggable behind `embeddings/base.py`; vectors must be L2-normalized (search ranks by L2 assuming it equals cosine).
- After any search-quality change, run `make eval VIDEOS=footage DATA=mm_data` (scores `footage/labels.csv`) — no shipping on vibes. Index with `--no-asr` to stay comparable with the recorded baseline.
- `.llc` files reference media relative to their own location (LosslessCut joins dirname(project) + mediaFileName) — relative paths are intentional.
- Whisper output is filtered (no_speech_prob / avg_logprob) because it hallucinates captions on music; don't remove that filter without eval.
- Prefer an existing library over hand-rolled code, and let it be the only implementation. Cutting is `smartcut` (MIT, pinned `==1.7`, optional extra): deprecated upstream 2026-02, so if PyAV ever breaks it, vendor the four modules used — do not reimplement with raw ffmpeg calls.
- Verify cut accuracy by pixel alignment on the copied interior, never by duration alone: a clip can be several frames out and still have a plausible duration. Test at 50 fps; the 15 fps fixture hides it.
