# HANDOFF — sage-bioclip2 2.0.0

Records the plugin state, the verified deploy path, and the live H00F node state
(not git-tracked) after the on-node cascade was proven end-to-end.

## Plugin state
- Version **2.0.0**, branch `master`, tree clean. `make test` → 91 passed
  (offline: vendored consumer/selection/seenstore/identity units + bioclip wiring
  with a stubbed classifier — no GPU).
- Architecture: **pywaggle2 cache consumer** (the sage-yolo2 v2 pattern). Reads
  `-v2-` frames from the shared `/local-cache`; the `--input` dir is the
  full-frame↔crop switch. See `README.md`, `VENDORED.md`, `ecr-meta/`.

## Deploy path (ECR portal build does NOT work — same as sage-yolo2)
- Base image `nvcr.io/nvidia/pytorch` (CUDA); ECR/Jenkins cross-build under QEMU
  crashes (Infra #3, open). Working path = **native Thor build + k3s side-load +
  `pluginctl run`**: `scripts/deploy-sideload.sh --skip-register`.
- The node has **no GitHub credentials for a fresh clone of a NEW private repo**
  (already-cloned repos can fetch). Transfer a new repo via git bundle:
  `git bundle create /tmp/x.bundle --all` → scp → `git clone x.bundle` on node →
  `git remote set-url origin https://github.com/flint-pete/sage-bioclip2.git`.
- `pluginctl run` bypasses the ECR catalog — no registration needed for dev/test.

## Verified on a Thor/arm64 node, 2026-07-15 — the live cascade
media-sampler3 (THE producer) → sage-yolo2 (crop producer) → **sage-bioclip2 (crop
consumer)**. Data API confirmed `env.species.species = "Cardinalis cardinalis"`
(Northern Cardinal — an example species, 100% on confident crops), `plugin=.../beckman/
sage-bioclip2:2.0.0`, `camera=top-crop-0`, frame-anchored, with crop provenance
attached: `source_class=bird`, `source_confidence≈0.75`, `source_unique_id=<parent
crop sha256>`. Species results trace back through the YOLO detection to the parent
frame.

## Changes made OUTSIDE this repo (live node state — not git-tracked)
1. **Image side-loaded into the node's k3s containerd:**
   `registry.sagecontinuum.org/beckman/sage-bioclip2:2.0.0` (~17.5 GB, arm64,
   BioCLIP-2.5 + TreeOfLife-200M embeddings baked in). Persists on the node.
2. **Long-running pod launched:** `sage-bioclip2-consumer` via `pluginctl run`
   (`--selector zone=core`, `--resource limit.memory=16Gi,request.memory=4Gi`,
   `-v /media/plugin-data/local-cache:/local-cache`, `WAGGLE_JOB_NAME=camera`
   `WAGGLE_TASK_NAME=sage-bioclip2`), args:
   `--source cache --input /local-cache/camera-crops/top-crop-0 --every 10m
   --all-unseen --max-frames 0 --rank Species --min-confidence 0.1`.
   Switch to full frames by changing one arg: `--input /local-cache/camera/top`.
3. Running alongside (untouched): the media-sampler3 producer and
   `sage-yolo2-consumer` (2.1.0 count + crop-produce).

## Rollback
`sudo pluginctl rm sage-bioclip2-consumer` — removes only this consumer; the
producer + yolo2 keep running. The image stays imported for a re-launch.

## Known tuning follow-up (left as-is per Pete, 2026-07-15)
A few low-confidence crops classify as non-birds (e.g. a rabbit at ~70%), likely
partial/edge crops. Tighten later by raising `--min-confidence` here or the
`--crop-match` threshold on the sage-yolo2 producer.
