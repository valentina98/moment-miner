# Captioning segments

`mm index --caption <model>` writes one sentence per segment into the searchable text. This is what makes a silent clip findable by word. `mm caption FOLDER` does the same for segments already indexed, without re-embedding them.

## Why

`search()` fuses two rankings with RRF: vector similarity over the segment embeddings, and full-text search over the `text` column (`moment_miner/search.py:38-42`). That column is filled from the Whisper transcript and nothing else (`moment_miner/indexer.py`), and it carries the FTS index (`moment_miner/store.py:52`).

Footage without speech leaves that column empty, so the text half of the fusion contributes nothing and search is effectively vision-only. Captions populate it — "pull-up" or "foam pit" become findable by word rather than only by pixel similarity. Captions are appended to the same column: no schema change, no change to the fusion.

## Where captions live

An archive is self-contained: everything derived from a folder of footage is written back into that folder, so a drive carries its own captions and can be unplugged and read elsewhere. `--caption-dir` and `-o` override that when the archive has to stay read-only.

`<folder>/captions.csv`, beside the footage — the same convention as the `labels.csv` that `mm eval` reads.

They are separate files on purpose. **Captions are machine-written indexed content; labels are human-checked ground truth.** Scoring captions against labels written by the same model measures agreement, not accuracy, so a caption is never also the label it is scored against.

Columns: `id,t0,t1,caption,model,written`.

Captions are read back on the next pass. Re-indexing a folder — including `--reindex` — reuses the captions already on disk and only spends on segments that don't have one. This is why they live next to the footage rather than in `mm_data`: a caption costs session quota (money only on the API-key route), and a deleted volume means a re-index, not a second caption pass.

## The method

**Segment geometry.** 8-second window, 4-second stride (`moment_miner/indexer.py`, overridable with `--window` / `--stride`). Segments start at 0 and step forward a stride at a time; a leftover under half a stride stretches the last segment to the video's end, and a larger one gets its own full-length segment ending there. So every segment is full length and the video's final frame is the final frame of a segment — which is exactly where the frame sampler looks.

**Frames.** How many stills a segment gets follows from its span and the stride, not from a fixed number. Under one stride: one frame, in the middle. Under two strides: two, at a third and two thirds. Otherwise: anchored half a stride from the end and stepping back a whole stride at a time, as many as fit. That is one frame per stride-length block of video, at the block's midpoint — and since windows overlap by half, neighbouring windows agree on the frame they share. At the defaults — 8 s window, 4 s stride — an ordinary segment yields two frames, at 2 s and 6 s. The anchor is at the end because that is where the action resolves: the trick lands and recording stops.

These tiers were reasoned about at stride 4 and do not adapt to another one — at stride 2 a full window would yield four frames instead of two, and frames are the dominant cost of a pass. `--caption-frames` forces a count and overrides the rule; `--caption-frames-short` / `--caption-frames-long` do the same per video length.

**Prompt.** The `PROMPT` constant in `moment_miner/captions.py` is the single source of truth. It asks for a **label of 5 to 12 words, not a description**: the action first, then the setting, in the plain words someone would type into a search box — "kong vault over a rail, concrete plaza". Sentences, framing ("a video showing"), and commentary on the shot are all banned, and a subject too small to identify must be called unclear rather than guessed.

The first version asked for 15-30 words and got paragraphs: *"Graffiti-covered concrete river embankment under a road bridge, people leaning on the railing above, green water and wooded hillside; the static shot barely changes."* Nobody types that, and against a BM25 index the extra words dilute the terms that matter.

## Two routes, one method

| | `mm index --caption` | A Claude Code session |
| --- | --- | --- |
| Pays with | `ANTHROPIC_API_KEY`, or your Claude subscription when that is unset (see Credentials) | Your Claude subscription |
| Frames | the stills the rule above picks (two at the defaults), at the 456x256 extraction raster | identical |
| Prompt | `PROMPT` in `captions.py` | paste the same text |
| Output | `captions.csv` + the index | a CSV you place at `<folder>/captions.csv` |
| Also loads | nothing | your `CLAUDE.md` and project memory |

The two are the same pass as long as the session gets the same frames and the same prompt. The one difference that does not go away: a Claude Code session loads `CLAUDE.md` and memory into its context, so it measures the model *plus* that harness. `--bare` would strip them but authenticates strictly by `ANTHROPIC_API_KEY`, never OAuth, so it is not available on a subscription. Identical contamination across sessions does not bias a comparison between models; it does make those captions incomparable with API-produced ones.

**The session route needs watching.** It ends by writing a file, and not every model runs with auto-accept available — where it is not, the session stops at a permission prompt and waits indefinitely. Nothing has failed at that point and nothing is lost; accept the write and it finishes. Check for the output file before concluding a run died, and do not start three sessions and walk away.

Use the session route to compare models or spot-check a handful of segments. Use `--caption` for anything bulk.

## Credentials

Resolved in this order, once, when the first caption is requested. **Reordered 2026-09-18: the free route wins, and the paid one is refused unless it was asked for and capped.**

1. Claude Code credentials at `$CLAUDE_CONFIG_DIR/.credentials.json` (default `~/.claude`), mounted into the container if you run in Docker — spends session quota, no money
2. `ANTHROPIC_API_KEY` — spends money, so it is used **only** with `--allow-paid` and `--paid-budget-usd` above zero. A key on its own, even with no subscription credentials present, raises rather than bills
3. Neither — one error naming both, raised before any frame is sent

The budget is a gate, not a meter: it is checked before the first request and nothing counts dollars as the pass runs. Metering is a next step, not a feature this claims.

Captioning is the only stage that spends anything — money on an API key, session quota on a subscription. `mm index` without `--caption` needs no credentials at all.

## What it costs

Segments per hour of footage is `3600 / stride` — 900 at the default stride 4, and higher on an archive of short clips, whose truncated windows do not overlap (a 50-clip corpus measured 1086). Under `mm index --caption`, stills are sent at the 456x256 extraction raster — captioning reuses the frames decoded for the embedding and never upscales them — which is ~170 input tokens each (156 by `w x h / 750`, 170 by counting 28-pixel tiles; confirm with `count_tokens` before a bulk run). `mm caption` at its default 640x360 sends ~307 by the same `w x h / 750` rule, so the table below understates its cost by roughly 1.5x. The prompt adds ~200 text tokens per segment, and ~30 output tokens come back.

At 900 segments/hour and the two frames an ordinary segment yields: `900 x (2 x 170 + 200)` = 0.49 M input tokens, `900 x 30` = 0.027 M output. **Frame count is the main cost lever** — each extra still adds ~170 tokens to a ~540-token request, so forcing 5 frames costs roughly 1.7x what the rule's two do.

| Model | $/MTok in / out | Per hour of footage | Batch API (-50%) |
| --- | --- | --- | --- |
| Opus 5 | $5 / $25 | $3.11 | $1.55 |
| Sonnet 5 | $2 / $10 | $1.24 | $0.62 |
| Haiku 4.5 | $1 / $5 | $0.62 | $0.31 |

Prices read from Anthropic's model pricing 2026-06-24; re-check before a bulk run. Prompt caching does not help — every image is unique. The Batch API's 50% does.

## Running it

```bash
mm index /videos --caption claude-haiku-4-5
mm index /videos --caption claude-opus-5 --caption-frames-long 6
make caption VIDEOS=footage CAPTION_MODEL=claude-haiku-4-5
mm caption /videos --model claude-haiku-4-5 --frame-size 640x360
make recaption VIDEOS=footage/src CAPTION_MODEL=claude-haiku-4-5 ARGS="--force"
```

**Two routes.** `mm index --caption` captions while it indexes, from the frames it already decoded for the embedding, so the model never sees more than 456x256. `mm caption` is a separate pass over an index that already exists: it reads each segment's span from the store, seeks to the same instants `index --caption` would pick, decodes only those stills at `--frame-size` (default 640x360; anything wider than 640 px is downscaled before sending, `FRAME_WIDTH` in `captions.py`), then rewrites the row's `text` as transcript plus caption and rebuilds the FTS index. Vectors, schema and fusion are untouched, and the embedding model is never loaded. Use it when the caption model, prompt or raster changes and the embeddings have not — that no longer costs a re-index. Segments that already have a sidecar caption are not bought again, but the cached caption is still written into the row; `--force` buys a fresh one. Stride and frames-per-window are read from what the manifest recorded for each video.

`make caption` exists because captions are written beside the footage: it mounts the archive read-write, where every other target mounts it `:ro`. Running `mm index --caption` against a read-only archive fails immediately, before the first paid request, rather than after. `--caption-dir` writes the sidecars elsewhere if the archive must stay read-only.

## Comparing caption sets

```bash
mm caption-compare opus.csv sonnet.csv haiku.csv --videos /videos
```

Renders one HTML page: each segment's stills, the captions under them, and a mark where the models disagree. Written to `caption-comparison.html` inside `--videos` — on the archive, with the footage it describes.

**Blind by default.** Which file wrote which caption is hidden and the column order is shuffled, because you necessarily know which model you launched; the blinding has to happen at judging time. The order is *balanced*, not merely random — every ordering used equally often, so each set sits in each column the same number of times. A free shuffle put one set in the middle column on 10 of 18 segments, which is exactly the position bias the shuffle exists to remove. The decode map is written beside the page as `<name>-key.csv`; leave it shut until every segment is judged. `--no-blind` labels the columns by filename instead.

**Disagreement is marked where the captions share no content word at all.** That is a fact about the text rather than a tuned threshold — and on the first real bake-off it marked 9 of 18 segments, where a graded score marked 17 and so said nothing. Words appearing in only one caption are highlighted inline: that is where one model saw a dam wall and another a bridge.

The test is lexical, so it has two known failure directions: a paraphrase ("bails a trick" / "fails a landing") reads as disagreement, and a confident error all the models share reads as agreement. It is a reading aid for a human judging captions, not a metric — nothing in the index depends on it.

`--frames-dir` uses stills already extracted elsewhere, named `<segment id>_*.jpg`, instead of decoding the videos. `--segments` supplies timings when the caption files carry only `seg,caption`.

## What captions get wrong

- **Stills lose motion and sound.** A swing versus a hang, a failed landing, someone talking — three frames per window is where that error enters.
- **Small subjects.** In wide 4K shots a subject can be 2% of frame height. The prompt tells the model to say so; a caption that confidently names an action at that scale is the failure to look for.
- **Repetition.** Many clips from one session share scenery. A model that writes near-identical captions for different moments is useless for retrieval even when each caption reads well.
