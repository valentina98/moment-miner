# Moment Miner

Search any large video archive with natural language, get back exact ranked
timestamps, and export the moments as lossless clips. No AI-edited output —
the source files are never re-encoded.

Nothing in the tool is specific to one kind of footage. The worked examples
below search a parkour archive, because that is what it was built and
measured against; sports, events, talks, travel and family archives work the
same way, and `mm annotate` ships label templates for several of them. Read
"kong vault" throughout as "whatever you would type".

## Quickstart

Everything runs in Docker — no local Python or ffmpeg needed (only Docker
and make):

```bash
make test                    # build the image + run the test suite
make index VIDEOS=footage/src  # index a folder (first run downloads models
                             #   into the mm-hf-cache volume, ~2 GB)
make search Q="athlete performs a kong vault"
make mine Q="kong vault" OUT=clips   # top matches as lossless clips in clips/
make shell                   # bash inside the container, mm on PATH
```

The `mm` CLI itself — inside `make shell`, on a GPU box, or any machine with
the package and ffmpeg installed (`pip install -e ".[asr,caption,rank,siglip,smartcut]"`):

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
| `mm index FOLDER` | scan recursively, index new/changed videos and any whose segment geometry changed (`--reindex`, `--no-asr`, `--backend`, `--caption`) |
| `mm search QUERY` | ranked timestamps (`-k`, `--json`, `--llc`, `--llc-dir`, `--static`/`--moving`, `--static-max`) |
| `mm mine QUERY FOLDER` | index-if-needed → search → top-k clips (`-o`, `-k`, `--no-smart`) |
| `mm locate -q Q -v VIDEO` | where inside one video the action is (`--labels`/`--videos` to sweep a CSV, `--fps`, `--locator`, `--duration`, `--top`, `-o`) |
| `mm annotate FOLDER` | interactive ground-truth labeling (`--template`) |
| `mm eval LABELS.csv` | recall@k + MRR against hand labels |
| `mm caption FOLDER` | caption already-indexed segments without re-embedding (`--model`, `--frame-size`, `--caption-frames`, `--force`) |
| `mm caption-compare A.csv B.csv` | several caption sets as one blind judging page |
| `mm export VIDEO T0 T1` | one clip (`--pad`, `--no-smart`, `--no-snap`) |
| `mm serve` | resident HTTP search daemon (`/search`, `/health`) |
| `mm status` | index statistics (`--errors` lists failed files) |
| `mm help [COMMAND]` | this list / per-command help (also `--version`) |

All commands take `--data-dir` (default `./mm_data`) before the subcommand.

**Static vs moving is a filter, not a search term.** Nobody types "the camera
moves", so it never enters the caption text — each segment carries a motion
value instead, and `mm search "..." --static` or `--moving` narrows a result
set by it. The value is the fraction of pixels that change between frames, 0.0
to 1.0, which is a proxy: it reads a whip-pan and a close-up filling the frame
alike. Segments with no motion value drop out of a filtered search rather than
being guessed at. The measurement is in `moment_miner/motion.py`.

**The measurement is stored; the judgement is not.** Motion lives in its own
`motion` table rather than in `segments_<backend>`, because it is a property of
the decoded frames and not of the embedding — one table serves every backend,
and the tag can be dropped or recomputed without rewriting the vectors. The
static/moving cut is applied at search time via `--static-max` (default 0.35),
so retuning it costs nothing; it used to be baked in at index time, which made
retuning cost a full re-index. The default separates cut clips from raw camera
clips cleanly on the reference footage, but it was calibrated on one shooter's
material — a good default, not a constant.

## Architecture: two-tier "index cheap, verify smart"

```
folder scan ─► SQLite manifest (size/mtime/partial-hash → incremental, resumable;
  records the window/stride/raster/backend each video was indexed under, so
  changing any of them re-indexes it without --reindex)
  ├─ ffprobe metadata
  ├─ ASR: faster-whisper → word-timestamped transcript
  ├─ TIER 1: embeddings per 8s overlapping window ─► LanceDB (vector + FTS)
  ├─ TIER 1.5: mm locate re-scores one shortlisted video frame by frame and
  │    picks the region to cut — same embedding, no re-index, resolution
  │    chosen per query rather than baked into the segment geometry
  └─ TIER 2 (planned): Qwen3-VL verifies/reranks top candidates only,
       writes descriptions + confidence

query (text | example clip) ─► ANN + BM25 ─► reciprocal-rank fusion
  ─► overlap merge ─► ranked (video, t0, t1, snippet)

export ─► smartcut: frame-accurate at both ends, only the boundary GOPs
  recoded, interior passed through bit-identically · --no-smart falls back
  to ffmpeg -c copy snapped to a keyframe (may start one GOP early)
```

Embedding backends are pluggable (`moment_miner/embeddings/`):

| Backend  | What                                          | Use                                  |
|----------|-----------------------------------------------|--------------------------------------|
| `siglip` | SigLIP2 frame embeddings, mean-pooled per segment | CPU/GPU baseline, zero-shot       |
| `mock`   | deterministic hash vectors                    | tests                                |
| planned  | Meta PE-AV / InternVideo2 (native video+audio)| motion-defined actions (flips, vaults)|

## Lossless export: smart cut by default

A stream copy can only begin on a keyframe, so cutting by copy alone snaps the start backwards and opens with up to one GOP of footage nobody asked for — 0.48 s on the Canon 4K sources here.

`mm export` therefore uses [smartcut](https://github.com/skeskinen/smartcut) (MIT) by default: it recodes only the GOPs at each boundary and passes everything between them through untouched. Measured on `footage/src/canon_50fps/2M4A2436.MP4`, asking for `1.2` to `3.0` (1.80 s):

| | start | clip duration | interior |
|---|---|---|---|
| default (smart) | **1.200 s exact** | 1.808 s | decodes bit-identically to source |
| `--no-smart` | 1.000 s | 2.122 s | whole clip byte-identical |

Boundary GOPs are recoded at `NEAR_LOSSLESS` (CRF 3). Verify start accuracy by pixel alignment on the copied interior, never by duration alone — a clip can be several frames out and still have a plausible duration, which is exactly how an earlier hand-rolled version passed its own tests while cutting four frames early.

`--no-smart` remains available and is the faster, fully byte-identical path when a keyframe-aligned start is good enough.

**Dependency note:** smartcut is an optional extra (`pip install -e ".[smartcut]"`, pinned to `1.7`). The package was deprecated in February 2026 but is MIT, pure Python, and depends only on PyAV; if a future PyAV breaks it, vendoring the four modules used here is the escape hatch. Core `mm index` / `mm search` never pull it in.

## Finding the cut: `mm locate`

Search ranks whole segments. It says *which* eight seconds are worth looking
at, never where inside them the action starts. `mm locate` answers the second
question by scoring one video frame by frame against the query and picking the
region worth cutting:

```bash
mm locate -q "kong vault over an obstacle" -v footage/src/2M4A2407.MP4 --fps 5
mm locate --labels footage/labels.csv --videos footage/src -o curves.html
```

It needs no re-index and no finer segmentation. The per-frame vectors already
exist — `embed_segment` is the mean of `embed_frames`, and the mean is the only
reason they were unreachable. Resolution is a `--fps` knob set per query rather
than a segment geometry baked into the index, which is the point: halving the
stride would double index size and embedding time for every video, forever, to
buy resolution you need on the handful of clips you are actually cutting.

**Region finders are interchangeable** (`moment_miner/locate.py`), because
extent is content-dependent — one pull-up or ten, three corks or fifteen:

| `--locator` | What it does |
|---|---|
| `maxsub` (default) | longest contiguous run beating the clip's own baseline; length comes from the curve, never asserted |
| `window` | highest-scoring window of exactly `--duration` seconds, for when you know the length and want it fixed |

`maxsub` is a max-subarray (Kadane) over the baseline-centered curve. Mean over
a window always prefers the single best frame and sum always prefers the whole
clip; centering makes above-baseline frames pay in and below-baseline frames
pay out, so a run ends where the action does. For a repetition query that
length *is* the answer — no single frame distinguishes three corks from
fifteen, only how long the run lasts.

**The baseline defaults to the mean, not the median.** Most of a clip is not
the action, so on a flat floor the median lands *on* it: every floor frame
centers to exactly zero, a zero neither extends a run nor ends one, and two
separate bumps merge across the dead stretch between them. The mean is pulled
above the floor by the bump. Pass a percentile instead when the action fills
most of the clip, which is the case the mean gets wrong.

Absolute cosines carry almost no information here — a live search returned
top-three scores five ten-thousandths apart — so every locator reads the shape
of the curve after centering, never a raw threshold.


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
`make annotate TEMPLATE=parkour`, `make eval`, `make probe PROMPTS=my.txt`,
`make serve`, `make shell` — add `GPUS=--gpus=all` on a GPU box. API keys go in
`.env`, git-ignored; copy `.env.example` and fill in what you need.

**Vast.ai:** either push the image (`docker tag moment-miner
<user>/moment-miner && docker push <user>/moment-miner`) and use it as the
instance image, or launch a stock `pytorch/pytorch` template and put this in
the on-start script:

```bash
apt-get update && apt-get install -y ffmpeg
git clone <repo-url> /app && pip install -e "/app[asr,caption,rank,siglip,smartcut]"
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
| Custom taxonomy (e.g. kong vs dash vault) | prototype vectors + linear probe on your labels | no | no |
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

These figures were taken **2026-09-02** and have not been re-taken since. Two
changes have landed that move indexing in opposite directions and roughly
cancel: the extraction raster went 320x180 → 456x256 (+4.3%, measured), and
frame extraction became one decode pass per video instead of one per window
(−7% on a 10-minute corpus, indicative). Treat the table as the right order of
magnitude, not as current.

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
- [ ] **M3** — feedback capture → linear probe for a custom class taxonomy
      (parkour moves as the worked example), highlight scoring (audio
      salience + VLM judge)

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
mm eval footage/labels.csv           # columns: query,video,start,end
```

### How to label videos

Labels always live **with the footage** as `<folder>/labels.csv` — that's
where `mm eval` and `make eval` look. Templates (shipped in
`moment_miner/templates/`) are read-only blueprints; they get copied, never
edited.

Labels are ground truth and stay hand-written. `examples/` is a different
thing and holds no labels: it is for **exemplar clips**, one folder per
label (`examples/kong_vault/clip1.mp4`), used to search by example rather
than by text. An exemplar must never cover a moment that `labels.csv` also
marks — the run would score higher for free and measure nothing.

The guided way — `mm annotate` creates the file from a template and appends
rows interactively (watch the video in any player, type the times here).
Containerized: `make annotate VIDEOS=footage TEMPLATE=parkour` (mounts the
footage writable, since the labels file lives next to it):

```text
$ mm annotate footage --template parkour
created footage/labels.csv from template 'parkour'
videos:
  [1] 2M4A2341.MP4
  [2] 2M4A2377.MP4
query (empty to finish): kong vault over an obstacle
video [1-2]: 2
start (M:SS or seconds): 0:04
end   (M:SS or seconds): 0:09
added (1 this session)
query (empty to finish):
done — 1 label(s) appended to footage/labels.csv
```

Or by hand: copy a template next to your footage
(`cp moment_miner/templates/labels_parkour.csv footage/labels.csv`) and fill
in `video,start,end` per moment in any editor:

```csv
query,category,video,start,end
backflip,flips,2M4A2341.MP4,0:04,0:09
kong vault over an obstacle,PK,2M4A2407.MP4,0:02,0:06
kong vault over an obstacle,PK,2M4A2407.MP4,0:12,0:16
```

`query,video,start,end` are the required columns. Extra ones are carried
along and ignored by `mm eval` — the parkour template ships a `category`
column so a long vocabulary stays navigable while you fill it in, and
`mm annotate` appends against whatever header the file already has. CSV has
no comment syntax, so group your ideas with that column rather than with
`#` lines, which parse as data rows.

**Duplicate the row for every additional occurrence** of the same query.
Rows you never fill are skipped automatically — no need to delete them. Add
your own query rows freely; delete queries you can't label confidently.

Rules that keep the metric honest:

- Label **every** matching moment for each query you use — an unlabeled true
  moment counts as a miss and silently corrupts recall. If you can't
  enumerate all instances of a query, drop that query.
- Time ranges tight but generous: start just before the action, end just
  after. Scoring is overlap-based, so ±1–2 s doesn't matter. That padding is
  deliberate and it makes these labels **edit points, not action extents** —
  they measure whether search found the moment, and cannot measure whether a
  proposed cut is tight. Judging `mm locate` needs a separate file of
  boundaries drawn to the action itself.
- Mix moment types: specific actions, outcomes (fails/crashes), context
  (talking to camera, crowd reactions), and 2–3 paraphrases of one thing.
- Queries with only 1–2 true moments across many files test discrimination
  best; 15–20 filled queries over 30+ minutes of footage give stable numbers.
- Videos are matched by filename, so keep filenames unique across folders.

Available templates (`moment_miner/templates/`, also listed by
`mm annotate --template nope`): parkour, wedding, concert, conference,
travel (planned: skate/BMX, team sports, climbing, birthdays/family).

### Captioning with a VLM

Hand labels are ground truth for `mm eval`. Captions are the other half:
`mm index --caption claude-haiku-4-5` writes one sentence per segment into the
searchable text, so silent footage is findable by word and not only by pixel
similarity. Captions land in `<folder>/captions.csv` beside the footage and are
reused on re-index. `mm caption <folder>` does the same for an index that
already exists, without re-embedding it. `mm caption-compare a.csv b.csv --videos <folder>` renders several caption sets
as one blind judging page, marking where the models disagree. Method, both
routes, credentials and cost per hour:
[docs/captions.md](docs/captions.md).

## Tests

```bash
pytest            # ffmpeg-dependent tests auto-skip if ffmpeg is missing
```
