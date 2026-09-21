# Probing a caption set with your own prompts

`mm eval` scores the label set: twelve rows with ground truth, recall@10 and MRR. It answers whether search finds the moments someone already marked.

`mm probe` answers the other question — whether the words you would actually type find the moment. There is no ground truth for that, so it prints hits to read rather than a number to quote.

```bash
mm probe prompts.txt --captions /videos/captions.csv --axis action --ranker laya -k 3
```

```
Pull-ups
  0.85  MVI_8459.MP4 96-104s   pull-up on bar
  0.84  2M4A6901.MP4 4-12s     pull-ups on horizontal bar
```

## Why it scores one axis

A caption carries four axes and a query constrains only the ones it names. Two of the four — `scene` and `light` — are clip-level, so every window of a clip repeats them; scoring the joined caption drowns the one part that varies. `--axis action` compares like with like. Measured on a 131-segment corpus: mean word overlap between windows of the same clip is 0.44 for the whole caption and 0.24 for the action alone.

## The rankers

| ranker | what it is | cost |
| --- | --- | --- |
| `mock` | word overlap | none; a baseline that must be beaten |
| `laya` | a 421M decision model, Apache-2.0, run locally on CPU | CPU time, about a second per segment |
| `jev` | a hosted decision model, same typed questions | metered; needs `TYPESAFE_API_KEY` |

Both classifiers answer a typed yes/no question per segment and return a calibrated probability, which is why a query can match a caption that shares none of its words. Word overlap cannot: on this corpus `Descent` and `Swing gainer` score 0.000 against every segment, because no caption uses those words.

**Neither classifier is wired into `search`.** They are instruments here. Offering one in the product is a separate decision, and for the hosted one it is also a decision about a paid dependency.

## Writing prompts worth probing

Write what you would type, before looking at any caption. A prompt fitted to the captions measures agreement rather than retrieval. Keep some that name only an action, some that name a place, and some about absence — "an empty shot with no people in it" is a `who` query, and it fails against text search, which ranks captions containing *people* at the top.
