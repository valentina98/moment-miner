# Moment Miner

Search large video archives (parkour / sports / event footage) with natural
language, get back exact ranked timestamps, and export the moments as lossless
clips. No AI-edited output — the source files are never re-encoded.

## Quickstart

Everything runs in Docker — no local Python or ffmpeg needed (only Docker
and make):

```bash
make test                    # build the image + run the test suite
make index VIDEOS=examples   # index a folder (first run downloads models
                             #   into the mm-hf-cache volume, ~2 GB)
make search Q="athlete performs a kong vault"
make mine Q="kong vault" OUT=clips   # top matches as lossless clips in clips/
make shell                   # bash inside the container, mm on PATH
```

The `mm` CLI itself — inside `make shell`, on a GPU box, or any machine with
the package and ffmpeg installed (`pip install -e ".[asr,siglip,smartcut]"`):

```
$ mm index /archive/parkour
$ mm search "athlete performs a kong vault"
/archive/parkour/video_001.mp4
  00:12:15 - 00:12:28  (score 0.0312)
$ mm export /archive/parkour/video_001.mp4 00:12:15 00:12:28
$ mm search "kong vault" --llc   # LosslessCut projects in llc/ next to the
                                 # videos: open, adjust cuts, export losslessly
$ mm mine "kong vault" /archive/parkour -k 5
                                 # one shot: index-if-needed → search → top-5
                                 # lossless clips in
                                 # /archive/parkour_mined/kong-vault_<timestamp>/
```

`*_mined` folders are never re-ingested by the indexer, so exports can't
pollute the index.

## Commands

| Command | Purpose |
|---|---|
| `mm index FOLDER` | scan recursively, index new/changed videos (`--reindex`, `--no-asr`, `--backend`) |
| `mm search QUERY` | ranked timestamps (`-k`, `--json`, `--llc`, `--llc-dir`) |
| `mm mine QUERY FOLDER` | index-if-needed → search → top-k clips (`-o`, `-k`, `--no-smart`) |
| `mm annotate FOLDER` | interactive ground-truth labeling (`--template`) |
| `mm eval LABELS.csv` | recall@k + MRR against hand labels |
| `mm export VIDEO T0 T1` | one clip (`--pad`, `--no-smart`, `--no-snap`) |
| `mm serve` | resident HTTP search daemon (`/search`, `/health`) |
| `mm status` | index statistics (`--errors` lists failed files) |
| `mm help [COMMAND]` | this list / per-command help (also `--version`) |

All commands take `--data-dir` (default `./mm_data`) before the subcommand.

## Architecture: two-tier "index cheap, verify smart"

```
folder scan ─► SQLite manifest (size/mtime/partial-hash → incremental, resumable)
  ├─ ffprobe metadata
  ├─ ASR: faster-whisper → word-timestamped transcript
  ├─ TIER 1: embeddings per 8s overlapping window ─► LanceDB (vector + FTS)
  └─ TIER 2 (planned): Qwen3-VL verifies/reranks top candidates only,
       refines start/end, writes descriptions + confidence

query (text | example clip) ─► ANN + BM25 ─► reciprocal-rank fusion
  ─► overlap merge ─► ranked (video, t0, t1, snippet)

export ─► smartcut: frame-accurate at both ends, only the boundary GOPs
  recoded, interior passed through bit-identically · --no-smart falls back
  to ffmpeg -c copy snapped to a keyframe (may start one GOP early)
```

Embedding backends are pluggable (`moment_miner/embeddings/`):

| Backend  | What                                          | Use                                  |
|----------|-----------------------------------------------|--------------------------------------|
| `siglip` | SigLIP2 mean-pooled frame embeddings          | CPU/GPU baseline, zero-shot          |
| `mock`   | deterministic hash vectors                    | tests                                |
| planned  | Meta PE-AV / InternVideo2 (native video+audio)| motion-defined actions (flips, vaults)|

## Lossless export: smart cut by default

A stream copy can only begin on a keyframe, so cutting by copy alone snaps the start backwards and opens with up to one GOP of footage nobody asked for — 0.48 s on the Canon 4K sources here.

`mm export` therefore uses [smartcut](https://github.com/skeskinen/smartcut) (MIT) by default: it recodes only the GOPs at each boundary and passes everything between them through untouched. Measured on `examples/2M4A2436.MP4`, asking for `1.2` to `3.0` (1.80 s):

| | start | clip duration | interior |
|---|---|---|---|
| default (smart) | **1.200 s exact** | 1.808 s | decodes bit-identically to source |
| `--no-smart` | 1.000 s | 2.122 s | whole clip byte-identical |

Boundary GOPs are recoded at `NEAR_LOSSLESS` (CRF 3). Verify start accuracy by pixel alignment on the copied interior, never by duration alone — a clip can be several frames out and still have a plausible duration, which is exactly how an earlier hand-rolled version passed its own tests while cutting four frames early.

`--no-smart` remains available and is the faster, fully byte-identical path when a keyframe-aligned start is good enough.

**Dependency note:** smartcut is an optional extra (`pip install -e ".[smartcut]"`, pinned to `1.7`). The package was deprecated in February 2026 but is MIT, pure Python, and depends only on PyAV; if a future PyAV breaks it, vendoring the four modules used here is the escape hatch. Core `mm index` / `mm search` never pull it in.

## Install

Dependencies are declared once in `pyproject.toml`; `requirements.txt` /
`requirements-dev.txt` select the extras (`asr` = faster-whisper,
`siglip` = torch+transformers+pillow, `smartcut` = frame-accurate export,
`dev` = pytest). System `ffmpeg`/`ffprobe` are required in every variant.

**Docker (recommended — same image locally and on Vast.ai):**

```bash
docker build --target runtime -t moment-miner .
docker run --rm --gpus all \
  -v /path/to/archive:/videos:ro -v mm_data:/data \
  moment-miner index /videos
docker run --rm --gpus all -v mm_data:/data moment-miner search "kong vault"

docker build -t moment-miner:dev .   # dev image (adds tests + pytest)
docker run --rm moment-miner:dev     # run the test suite
```

Day to day, the Makefile wraps all of this: `make test`, `make index
VIDEOS=/path`, `make search Q="kong vault"`, `make mine Q="..." OUT=clips`,
`make annotate TEMPLATE=parkour`, `make eval`, `make serve`, `make shell` —
add `GPUS=--gpus=all` on a GPU box.

**Vast.ai:** either push the image (`docker tag moment-miner
<user>/moment-miner && docker push <user>/moment-miner`) and use it as the
instance image, or launch a stock `pytorch/pytorch` template and put this in
the on-start script:

```bash
apt-get update && apt-get install -y ffmpeg
git clone <repo-url> /app && pip install -e "/app[asr,siglip,smartcut]"
```

CUDA is picked up automatically by both faster-whisper and torch.
RTX 4090 ≈ $0.29–0.59/hr (2026).

**Local virtualenv (no Docker):**

```bash
python -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU box
pip install -r requirements-dev.txt
```

## Self-hosted vs cloud APIs

| | Self-hosted (this project) | Twelve Labs (Marengo+Pegasus) | Gemini-based (e.g. SentrySearch cloud mode) |
|---|---|---|---|
| Cost per 100 h footage | ≈ $3–8 GPU time (tier 1) | ≈ $252 indexing + recurring infra | ≈ $284 (~$2.84/h indexed) |
| Data locality | footage never leaves your machines | upload TBs to their cloud | frames go to Google |
| Quality | good baseline; VLM rerank tier planned | best-in-class out of the box | strong |
| Search-by-example | native (same embedding space) | supported | limited |
| Custom taxonomy (kong vs dash vault) | prototype vectors + linear probe on your labels | no | no |
| Effort | you run the pipeline | trivial | low |

Reference projects studied (cloned under `refs/`, git-ignored):
[SentrySearch](https://github.com/ssrajadh/sentrysearch),
[GSoC video-search-engine](https://github.com/AkashKumar7902/video-seach-engine),
[VideoRAG](https://arxiv.org/abs/2502.01549).

## Performance

Measured on ~50 s of real 4K footage (5 Canon clips, 937 MB, ~110 Mbps) inside
the dev container. Laptop = AMD Ryzen 5 3500U, 6 cores exposed to WSL2, 12 GB
RAM, no GPU (Docker on WSL2). GPU column to be filled from the first Vast.ai
run.

**Full indexing (whisper-small ASR + SigLIP2 embeddings) by footage duration**
(single 4K file, built by lossless concat of the example clips):

| Footage | Laptop (CPU) | ×realtime | RTX 4090 (Vast.ai) |
|---|---|---|---|
| 15 s | 77 s | 5.2× | pending |
| 30 s | 79 s | 2.7× | pending |
| 1 min | 2 m 54 s | 2.9× | pending |
| 5 min | 13 m 08 s | 2.6× | pending |
| 10 min | 25 m 02 s | 2.5× | pending |

Cost is roughly *fixed overhead + ~2.5× realtime*: model loading (~40 s) plus
per-file pipeline setup dominate short files, while long footage converges to
~2.5× realtime on this CPU. Extrapolated, 100 h of archive ≈ 250 CPU-hours —
this is exactly what the GPU rental is for.

**Per-action breakdown** (~50 s across 5 separate 4K files — per-file overhead
makes this slower than the single-file numbers above):

| Action | Laptop (CPU) | ×realtime | RTX 4090 (Vast.ai) |
|---|---|---|---|
| Scan + fingerprint (5 files, 937 MB) | 0.2 s | — | — (IO-bound) |
| Index: SigLIP2 embeddings only | 161 s | 3.2× | pending |
| Index: full (whisper-small ASR + embeddings) | 222 s | 4.5× | pending |
| &nbsp;&nbsp;└ ASR share (whisper-small) | ≈61 s | 1.2× | pending |
| Search, one CLI invocation | 16–20 s | — | pending |
| Export 3 s lossless clip (stream copy) | 3.6 s | — | — (CPU/IO-bound) |

Search latency is almost entirely SigLIP model load — each CLI call starts a
fresh process. A resident daemon (planned alongside the M2 web UI) brings warm
queries under a second. On the indexing side, decoding 4K sources is a large
share of the cost; GPU NVDEC + batched inference is where a rented card pays
off.

## Roadmap

- [x] **M0** — manifest + incremental scan, ASR, tier-1 embeddings, hybrid CLI
      search, keyframe-snapped lossless export
- [ ] **M0.5** — eval harness (~20 hand-labeled moments, recall@10) + real
      GPU-hrs/hr-footage cost table on Vast.ai
- [ ] **M1** — Qwen3-VL rerank + temporal grounding (`--full-captioning`
      ingest option), confidence scores, descriptions
- [ ] **M2** — search-by-example + prototype move classes, minimal web UI
  - [x] smart cut: frame-accurate export, via the `smartcut` package
- [ ] **M3** — feedback capture → linear probe for parkour move taxonomy,
      highlight scoring (audio salience + VLM judge)

## Search daemon

Each `mm search` CLI call reloads the models (~20 s on CPU). `mm serve` loads
them once and answers over HTTP — warm queries are sub-second:

```bash
mm serve --port 7700          # or containerized: make serve
curl "localhost:7700/search?q=kong+vault&k=5"
curl localhost:7700/health    # backend, device, segment count
```

## Evaluating search quality

`mm eval` scores the index against hand-labeled ground truth (recall@k + MRR):

```bash
mm eval examples/labels.csv          # columns: query,video,start,end
```

### How to label videos

Labels always live **with the footage** as `<folder>/labels.csv` — that's
where `mm eval` and `make eval` look. Templates (shipped in
`moment_miner/templates/`) are read-only blueprints; they get copied, never
edited.

The guided way — `mm annotate` creates the file from a template and appends
rows interactively (watch the video in any player, type the times here).
Containerized: `make annotate VIDEOS=examples TEMPLATE=parkour` (mounts the
footage writable, since the labels file lives next to it):

```text
$ mm annotate examples --template parkour
created examples/labels.csv from template 'parkour'
videos:
  [1] 2M4A2341.MP4
  [2] 2M4A2377.MP4
query (empty to finish): kong vault over an obstacle
video [1-2]: 2
start (M:SS or seconds): 0:04
end   (M:SS or seconds): 0:09
added (1 this session)
query (empty to finish):
done — 1 label(s) appended to examples/labels.csv
```

Or by hand: copy a template next to your footage
(`cp moment_miner/templates/labels_parkour.csv examples/labels.csv`) and fill
in `video,start,end` per moment in any editor:

```csv
query,video,start,end
a person performs a backflip,2M4A2341.MP4,0:04,0:09
kong vault over an obstacle,2M4A2407.MP4,0:02,0:06
kong vault over an obstacle,2M4A2407.MP4,0:12,0:16
```

**Duplicate the row for every additional occurrence** of the same query.
Rows you never fill are skipped automatically — no need to delete them. Add
your own query rows freely; delete queries you can't label confidently.

Rules that keep the metric honest:

- Label **every** matching moment for each query you use — an unlabeled true
  moment counts as a miss and silently corrupts recall. If you can't
  enumerate all instances of a query, drop that query.
- Time ranges tight but generous: start just before the action, end just
  after. Scoring is overlap-based, so ±1–2 s doesn't matter.
- Mix moment types: specific actions, outcomes (fails/crashes), context
  (talking to camera, crowd reactions), and 2–3 paraphrases of one thing.
- Queries with only 1–2 true moments across many files test discrimination
  best; 15–20 filled queries over 30+ minutes of footage give stable numbers.
- Videos are matched by filename, so keep filenames unique across folders.

Available templates (`moment_miner/templates/`, also listed by
`mm annotate --template nope`): parkour, wedding, concert, conference,
travel (planned: skate/BMX, team sports, climbing, birthdays/family).

## Tests

```bash
pytest            # ffmpeg-dependent tests auto-skip if ffmpeg is missing
```
