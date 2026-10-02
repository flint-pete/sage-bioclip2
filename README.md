# sage-bioclip2

BioCLIP2 species classifier for Sage/Waggle, built on the **sage-yolo2 v2
cache-consumer pattern**. It reads self-describing `-v2-` frames from the shared
on-node `/local-cache`, runs BioCLIP-2.5 (TreeOfLife-200M) taxonomy
classification, and publishes species predictions **frame-anchored** (observation
time = the moment the photo was taken).

## Where this fits

sage-bioclip2 is the second of three **test consumers** in the media-sampler3 stack
(the third, sage-birdnet2, handles audio). It is the last stage of the image cascade:

```
camera ─▶ media-sampler3 ─▶ /local-cache/camera/top/ ─▶ sage-yolo2 ─▶ /local-cache/camera-crops/top-crop-N/ ─▶ sage-bioclip2 ─▶ env.species.* ─▶ Beehive
```

- **Prerequisites:** a running [media-sampler3](https://github.com/flint-pete/media-sampler3)
  producer, and [sage-yolo2](https://github.com/flint-pete/sage-yolo2) running with
  `--crop-match` (e.g. `bird:0.4`) and `--crop-cache-name camera-crops`. Without
  crops there is nothing for bioclip2 to classify.
- **Install, restart and the big picture** live in the hub repo:
  - [install guide](https://github.com/flint-pete/media-sampler3/blob/master/INSTALLING-MEDIA-SAMPLER3.md)
    (Step 5 to build; Steps 6b–6g to run and test)
  - [REBOOT-RECOVERY.md](https://github.com/flint-pete/media-sampler3/blob/master/REBOOT-RECOVERY.md)
  - [HOW-IT-WORKS.md](https://github.com/flint-pete/media-sampler3/blob/master/docs/HOW-IT-WORKS.md)
- `/local-cache` is provided and bounded by
  [wes-local-cache-manager](https://github.com/flint-pete/wes-local-cache-manager).
  If it isn't mounted, cache mode **fails fast** at startup.

**The model.** [BioCLIP-2.5](https://huggingface.co/imageomics/bioclip-2.5-vith14)
(ViT-H/14) is a CLIP-style vision–language model trained on the
TreeOfLife-200M dataset. bioclip2 compares an image against precomputed text
embeddings for every taxon in the tree of life and returns the best matches at
the chosen rank (`--rank Species` by default).

- **The model is baked into the image.** The Dockerfile instantiates the
  classifier at build time, which downloads the model and the TreeOfLife
  embeddings. That's why the image is about 17 GB.

### Sage adjustments to upstream BioCLIP / pybioclip

> **These are changes we made for Sage. They aren't in upstream BioCLIP or
> pybioclip.** If you compare against upstream examples, or upgrade pybioclip,
> start here.

1. **Offline model loading: `ENV HF_HUB_OFFLINE=1` in the `Dockerfile`** (added
   Oct 2026).
   - **What upstream does.** pybioclip loads its model through open_clip and
     huggingface_hub. Even when the files are already cached, those libraries
     contact huggingface.co at every start-up. The old image did this: on H039 it
     made 5 HEAD requests per start, each asking for the **latest** (`main`)
     version of the model and the TreeOfLife embeddings.
   - **Why that's wrong for a Sage node:**
     - **The model could change without anyone noticing.** If the upstream model
       repository is updated, a pod with internet access would download the new
       files (about 5 GB) when it next starts. It would then run a different
       model from the one this image was built and tested with.
     - **Start-up can stall** on a network that silently drops traffic, waiting on
       each request's timeout.
     - **It depends on huggingface.co** being reachable and not rate-limiting
       unauthenticated requests.
   - **What we changed.** `HF_HUB_OFFLINE=1` makes huggingface_hub use only the
     files baked into the image.
   - **Verified on H039:**
     - the new image makes **0** requests at start-up;
     - it classified the seeded cardinal (*Cardinalis cardinalis*, 100%);
     - it works with networking switched off entirely (`docker run --network none`).
   - **To turn the online checks back on** (for example, when testing a newer
     model): `-e HF_HUB_OFFLINE=0`. To actually change the model version,
     rebuild the image.
2. **BioCLIP-2.5 support: `patch_pybioclip.py`.** pybioclip 2.1.5 only knows
   BioCLIP 1 and 2. At image-build time this script adds the 2.5 model string
   and its embedding filenames to pybioclip's internals. The Dockerfile applies it
   once, and running it a second time isn't supported. **Remove it when upstream
   pybioclip supports 2.5.**
3. **Pinned library versions: `requirements.txt`.** pybioclip, plus the libraries
   it uses to load the model (`open_clip_torch`, `huggingface_hub`, `timm`), are
   pinned to the versions that built and ran on H039. The offline loading in (1)
   depends on how those libraries behave, so change them together, rebuild, and
   re-run the install guide's seeded test.

**Code map**

| File | What it does |
|---|---|
| `app.py` | CLI, wake loop, `BioCLIP2Classifier` (pybioclip `TreeOfLifeClassifier` wrapper), publishing and provenance |
| `consumer.py`, `selection.py`, `seenstore.py`, `node_info.py`, `save_match.py` | Cache-consumer machinery copied (vendored) unchanged from sage-yolo2 (see `VENDORED.md`) |
| `patch_pybioclip.py` | Build-time patch that enables BioCLIP-2.5 in pybioclip |
| `scripts/deploy-sideload.sh` | Native Thor build plus k3s import (identical to sage-yolo2's) |

## 1. The switch: full frames OR crops, one CLI parameter

The cache directory bioclip2 reads is a single `--input` argument. Because
media-sampler3 frames and sage-yolo2 crops are the **same v2 format**, the same
plugin classifies either — with no code change:

```bash
# Classify FULL media-sampler3 frames
python3 app.py --source cache --input /local-cache/camera/top --rank Species

# Classify only sage-yolo2 crops (the detect→classify cascade) — change ONE arg
python3 app.py --source cache --input /local-cache/camera-crops/top-crop-0 --rank Species
```

That's the intended production pattern: **media-sampler3 (THE producer) →
sage-yolo2 (crop producer) → sage-bioclip2 (crop consumer)**, all mediated by the
shared cache, with no plugin calling another.

```
media-sampler3      sage-yolo2 (detect + crop)          sage-bioclip2 (classify)
 camera → cache  →   YOLO detect + crop → crop stream →   read crops, BioCLIP2 species
 camera/top          camera-crops/top-crop-<idx>          env.species.* + provenance
```

## 2. Quick start

```bash
# Production: classify crops every 10 min, backlog mode
python3 app.py --source cache --input /local-cache/camera-crops/top-crop-0 \
  --every 10m --all-unseen --max-frames 0 --rank Species --min-confidence 0.1

# Same, but full frames
python3 app.py --source cache --input /local-cache/camera/top --every 10m --all-unseen --rank Species

# Local testing on a folder of images (no node/cache); sage-yolo2 ships a bird fixture
python3 app.py --source image-dir --input ../sage-yolo2/tests/test-images --rank Species
```

**One instance reads one crop directory.** sage-yolo2 writes the first matching
detection of each frame to `top-crop-0`, the second to `top-crop-1`, and so on.
The standard setup reads only `top-crop-0`, so a second bird in the same frame is
not classified. To cover more, run one bioclip2 instance per `top-crop-N`
directory, each with a different `--name` and `WAGGLE_TASK_NAME`. Reading every
crop directory from one instance is an open improvement.

**Thresholds work together.** yolo2's `--crop-match bird:0.4` decides what gets
cropped. bioclip2's `--min-confidence` decides whether a top species result is
reported as confident (`env.species.<rank>`). A loose crop threshold sends more
non-birds and partial birds to the classifier. BioCLIP will still name *some*
organism for them (a rabbit at about 70% has been seen), so tighten `--crop-match`
if that matters.

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
| `--save-match <rules>` | OR-list of `Taxon:confidence` rules — matches the SCIENTIFIC or COMMON name of any prediction (e.g. `Northern Cardinal:0.5` as an example), or `*:0.5`. |

## 4. Published data

| Topic | Value | Notes |
|---|---|---|
| `env.species.<rank>` | string | Top predicted taxon at the chosen rank (only when confident). |
| `env.species.<rank>.confidence` | float | Its confidence 0–1. |
| `env.species.top5` | JSON | Top-k `[{name,common_name,confidence},…]`. |
| `env.species.summary` | int | 1 = confident prediction, 0 = none / heartbeat. |

**Meta on every record:** `camera`, `model`, `rank`. In cache mode, also
`vsn`/`node_id`/`lat`/`lon`/`location_source` when the frame carries them.
Identity comes from the frame's EXIF first, with the pod's `WAGGLE_NODE_*` env
as fallback.

**Crop provenance (when classifying a sage-yolo2 crop):** the crop's `source{}`
block is surfaced as `source_class`, `source_confidence` and `source_unique_id`
on the species record. `source_unique_id` is the SHA-256 of the **parent full
frame**, the media-sampler3 JPEG the crop was cut from. So a species result
traces back to the YOLO detection and the exact producer frame. Plain
media-sampler3 frames have no `source` block (full-frame mode), so these keys are
simply absent.

Example (abridged) of what a species record carries:

```
name: env.species.species    value: "Cardinalis cardinalis"
timestamp: <capture time of the parent frame>
meta: camera=top-crop-0, rank=Species, vsn=<VSN>,
      source_class=bird, source_confidence=0.91, source_unique_id=<sha256 of parent frame>
```

**Frame-anchored.** In cache mode the record timestamp is the frame's CAPTURE
time (read from its metadata), not when BioCLIP ran.

## 5. Architecture / reuse

The cache-consumer machinery (`consumer.py`, `selection.py`, `seenstore.py`,
`node_info.py`, `save_match.py`) is **copied unchanged from sage-yolo2**. It is the
shared v2 read contract; see `VENDORED.md`. The BioCLIP2-specific parts
(`BioCLIP2Classifier`, `patch_pybioclip.py`) were carried over from
`sage-bioclip` v1.

**Seen-store location (known quirk).** The copied `seenstore.py` names its plugin
directory `sage-yolo2`, so bioclip2's memory lives at
`/local-cache/.state/sage-yolo2/<consumer-id>/camera-crops/top-crop-0/seen`.
With the launch flags below, `<consumer-id>` is `camera-sage-bioclip2`, which
keeps it separate from yolo2's own store. Keep `WAGGLE_JOB_NAME` and
`WAGGLE_TASK_NAME` the same across relaunches, or the memory is lost and the
backlog is reclassified.

## 6. Testing

`make test` runs the offline unit suite (pure-stdlib cache-consumer logic + the
bioclip wiring with a stubbed classifier — no GPU, no model download). The real
BioCLIP inference is verified on-node.

## 7. Deployment

The image hasn't been published to the registry yet, so the verified path is a
native Thor build + k3s side-load + `pluginctl run` (the ECR portal can also build
Thor images now):

```bash
# on the Thor node, from the repo root
scripts/deploy-sideload.sh --skip-register     # build (arm64) → import to k3s

sudo pluginctl-nodeinfo run --name sage-bioclip2-consumer --selector zone=core \
  --resource limit.memory=16Gi,request.memory=4Gi \
  -v /media/plugin-data/local-cache:/local-cache \
  -e WAGGLE_JOB_NAME=camera -e WAGGLE_TASK_NAME=sage-bioclip2 \
  registry.sagecontinuum.org/beckman/sage-bioclip2:2.0.0 -- \
  --source cache --input /local-cache/camera-crops/top-crop-0 \
  --every 10m --all-unseen --max-frames 0 --rank Species --min-confidence 0.1
```

> **Pod identity.** `pluginctl-nodeinfo` is the patched `pluginctl` from install
> Step 3 (same flags). Its pods get the node's VSN, id and GPS, which the consumer
> uses as a cross-check and a GPS fallback, so records carry lat/lon even when
> the frame has none (`location_source: node`). With the stock `pluginctl`, the
> pod has no identity env and only the frame's EXIF counts. Verified on H039, Oct 2026.

This is the same command as the install guide's Step 6d.

- `--selector zone=core` is required whenever you use `-v`.
- `--resource limit.memory=16Gi` prevents an OOMKill.
- The `-e` variables name the seen-store, so it survives relaunches.
- `--max-frames 0` means "drain all unseen crops each wake". The default is 1,
  which would fall behind.

To classify full frames instead, change one argument to
`--input /local-cache/camera/top`.

**Runtime notes (from the H039 fresh install, Oct 2026):**

- **It always runs on the CPU.** `app.py` creates `TreeOfLifeClassifier(model_str=...)`
  without a `device`, and pybioclip's default is `device='cpu'`, so the GPU is never
  used, even on a node where the pod can see one. Classifying one crop took about 1 s
  on H039. Passing `device="cuda" if torch.cuda.is_available() else "cpu"` (as
  sage-yolo2 does) is an open improvement. Separately, on nodes like H039 the pod
  couldn't see the GPU anyway (see the hub guide, Step 6c, "Is it using the GPU?").
- **No internet access needed.** The image loads the model offline (Sage
  adjustment 1 above), with zero requests to huggingface.co. Before that change,
  each start made 5 requests to check for a newer model.

## Docs in this repo

- `VENDORED.md`: which files are copied from sage-yolo2, and what must stay in sync.
- `CHANGELOG.md`: release notes.
- [docs/history/](docs/history/): the July 2026 handoff/status notes (not maintained).

## Contact

Pete Beckman — pete.beckman@northwestern.edu
