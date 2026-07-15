# sage-bioclip2

BioCLIP2 species classifier for Sage/Waggle, built on the **sage-yolo2 v2
cache-consumer pattern**. It reads self-describing `-v2-` frames from the shared
on-node `/local-cache`, runs BioCLIP-2.5 (TreeOfLife-200M) taxonomy
classification, and publishes species predictions **frame-anchored** (observation
time = the moment the photo was taken).

## 1. The switch: full frames OR crops, one CLI parameter

The cache directory bioclip2 reads is a single `--input` argument. Because
image-sampler2 frames and sage-yolo2 crops are the **same v2 format**, the same
plugin classifies either — with no code change:

```bash
# Classify FULL image-sampler2 frames
python3 app.py --source cache --input /local-cache/hummingcam/top --rank Species

# Classify only sage-yolo2 BIRD CROPS (the detect→classify cascade) — change ONE arg
python3 app.py --source cache --input /local-cache/hummingcam-crops/top-crop-0 --rank Species
```

That's the intended production pattern: **image-sampler2 → sage-yolo2 (crop
producer) → sage-bioclip2 (crop consumer)**, all mediated by the shared cache,
with no plugin calling another.

```
image-sampler2      sage-yolo2 (detect + crop)          sage-bioclip2 (classify)
 camera → cache  →   YOLO, crop birds → crop stream  →   read crops, BioCLIP2 species
 hummingcam/top      hummingcam-crops/top-crop-<idx>      env.species.* + provenance
```

## 2. Quick start

```bash
# Production: classify bird crops every 10 min, backlog mode
python3 app.py --source cache --input /local-cache/hummingcam-crops/top-crop-0 \
  --every 10m --all-unseen --max-frames 0 --rank Species --min-confidence 0.1

# Same, but full frames
python3 app.py --source cache --input /local-cache/hummingcam/top --every 10m --all-unseen --rank Species

# Local testing on a folder of images (no node/cache)
python3 app.py --source image-dir --input ./tests/test-images --rank Species
```

## 3. CLI reference

`--source` and `--input` are **required**.

| Flag | Meaning |
|---|---|
| `--source {cache,image-dir}` | **(required)** `cache` = consume v2 frames from the shared cache (production); `image-dir` = local test folder. |
| `--input <value>` | **(required)** cache dir `<root>/<cache-name>/<camera>` — raw OR crop stream — or a directory of test images. **This is the full-frame↔crop switch.** |
| `--every <dur>` | Wake cadence. `0` = single-shot. `s`/`m`/`h`. Default `0`. |
| `--select-every <dur>` | Sampling stride (one frame per this much capture-time). `0` = newest unseen. Default `0`. |
| `--max-frames <int>` | Cap frames per wake (`0` = unlimited). Default `1`. |
| `--all-unseen` | Backlog mode: classify every not-yet-seen frame (capped by `--max-frames`). |
| `--max-runtime <sec>` | Wall-clock bound (`0` = forever). Default `0`. |
| `--consumer-id <id>` | Override seen-store consumer-id (default `WAGGLE_JOB_NAME`+`WAGGLE_TASK_NAME`). |
| `--seen-store <path>` | Override seen-store path. |
| `--reprocess` | Ignore the seen-store (still records what it processes). |
| `--rank <Rank>` | Kingdom/Phylum/Class/Order/Family/Genus/Species. Default `Species`. |
| `--model <str>` | BioCLIP model string. Default `hf-hub:imageomics/bioclip-2.5-vith14`. |
| `--top-k <int>` | Top-k predictions. Default `5`. |
| `--min-confidence <float>` | Below this, treat as no-confident-prediction. Default `0.1`. |
| `--upload-image {Y,N}` | Allow annotated uploads. Governed by `--save-match` when set. |
| `--save-match <rules>` | OR-list of `Taxon:confidence` rules — matches the SCIENTIFIC or COMMON name of any prediction (e.g. `Ruby-throated Hummingbird:0.5`), or `*:0.5`. |

## 4. Published data

| Topic | Value | Notes |
|---|---|---|
| `env.species.<rank>` | string | Top predicted taxon at the chosen rank (only when confident). |
| `env.species.<rank>.confidence` | float | Its confidence 0–1. |
| `env.species.top5` | JSON | Top-k `[{name,common_name,confidence},…]`. |
| `env.species.summary` | int | 1 = confident prediction, 0 = none / heartbeat. |

**Meta on every record:** `camera`, `model`, `rank`; in cache mode `vsn`/`node_id`/
`lat`/`lon` when the frame carries them.

**Crop provenance (when classifying a sage-yolo2 crop):** the crop's `source{}`
block is surfaced as `source_class`, `source_confidence`, `source_unique_id` on
the species record — so a species result traces back to the YOLO detection AND the
parent full frame. Plain image-sampler2 frames have no `source` (full-frame mode);
these keys are simply absent.

**Frame-anchored.** In cache mode the record timestamp is the frame's CAPTURE
time (read from its metadata), not when BioCLIP ran.

## 5. Architecture / reuse

The cache-consumer machinery (`consumer.py`, `selection.py`, `seenstore.py`,
`node_info.py`, `save_match.py`) is **vendored byte-identical from sage-yolo2** —
it is the shared v2 read contract. See `VENDORED.md`. The BioCLIP2 brains
(`BioCLIP2Classifier`, `patch_pybioclip.py`) are grafted from `sage-bioclip` v1.

## 6. Testing

`make test` runs the offline unit suite (pure-stdlib cache-consumer logic + the
bioclip wiring with a stubbed classifier — no GPU, no model download). The real
BioCLIP inference is verified on-node.

## 7. Deployment

ECR portal build fails for this plugin (NVIDIA CUDA base + QEMU cross-build,
Infra #3). Working path = native Thor build + k3s side-load + `pluginctl run`:

```bash
# on the Thor node, from the repo root
scripts/deploy-sideload.sh --skip-register     # build (arm64) → import to k3s

sudo pluginctl run --name sage-bioclip2-consumer --selector zone=core \
  --resource limit.memory=16Gi,request.memory=4Gi \
  -v /media/plugin-data/local-cache:/local-cache \
  registry.sagecontinuum.org/beckman/sage-bioclip2:2.0.0 -- \
  --source cache --input /local-cache/hummingcam-crops/top-crop-0 \
  --every 10m --all-unseen --rank Species --min-confidence 0.1
```

`--selector zone=core` is required with `-v`; `--resource limit.memory=16Gi`
avoids OOMKill. Switch to full frames by changing one arg to
`--input /local-cache/hummingcam/top`.

## Contact

Pete Beckman — pete.beckman@northwestern.edu
