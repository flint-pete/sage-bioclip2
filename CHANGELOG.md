# Changelog

All notable changes to the `sage-bioclip2` Sage plugin.

## Unreleased (docs only; image stays 2.0.0)

### Documented
- **GPU now available to pods** (fleet-wide k3s `default-runtime: nvidia`, Oct 2026),
  but bioclip2 still runs on the CPU, because `app.py` passes no `device` to
  pybioclip. Measured on H039 with the same crop: 1.86 s on the CPU vs 0.12 s on
  CUDA, same species and score. The README now says so.

### Changed (Sage adjustment; see README "Sage adjustments")
- **Offline model loading:** the image now sets `ENV HF_HUB_OFFLINE=1`.
  - Before: 5 HEAD requests to huggingface.co at every start, resolving `main`, so
    the model could change without notice.
  - Now: 0 requests.
  - Verified on H039, including with networking switched off (`--network none`).
    Override with `-e HF_HUB_OFFLINE=0`.
- **Pinned dependencies** to the versions verified on H039: `pywaggle[all]==0.56.3`,
  `open_clip_torch==3.3.0`, `huggingface_hub==2.1.1`, `timm==1.0.30`,
  `opencv-python-headless==4.11.0.86`, `pillow==11.3.0`, `piexif==1.1.3`,
  `numpy==1.26.4`.
- **Dockerfile:** the opencv reinstall used an unquoted `opencv-python-headless>=4.8.0`.
  The shell read `>=4.8.0` as an output redirect, so pip installed the newest
  opencv. It's now quoted and pinned.
- **README:** removed the incorrect claim that the pod needs outbound network.

### Changed
- README: three test consumers; step references follow the install guide.
- README: pod-identity note. Launching with the patched `pluginctl-nodeinfo`
  (wes-nodeinfo-injection Tier 1b) gives the pod `WAGGLE_NODE_*`, enabling the
  cross-check and the node-GPS fallback.
- README runtime notes (from the H039 fresh install): bioclip2 always runs on the
  CPU (pybioclip's default `device='cpu'`; passing a device is an open improvement),
  and it contacts huggingface.co at start-up.
- ECR note: the cyberinfrastructure team fixed the ECR build for Thor (arm64,
  including CUDA bases), so current docs no longer say ECR can't build this
  image. The image just hasn't been published yet; side-load stays the verified
  dev path. The QEMU notes in older entries below are historical.
- **Student-readiness README pass:**
  - added "Where this fits", prerequisites, what BioCLIP-2.5 is, why
    `patch_pybioclip.py` exists, and a code map
  - a species-record example; `source_unique_id` is the **parent frame's**
    SHA-256 (corrects the earlier "parent crop" wording)
  - the seen-store path quirk, the one-instance-per-crop-dir limitation, and the
    threshold trade-off
  - the deploy command now matches the install guide (adds `-e WAGGLE_*`,
    `--max-frames 0`)
- Fixed the image-dir test path (`../sage-yolo2/tests/test-images`) in the README
  and the `--help` epilog.
- `consumer.py` was re-synced from sage-yolo2 (comment lines only), so all
  vendored modules are byte-identical again. `node_info.py` was re-synced to
  pywaggle2-nodeinfo v0.1.1 (a doc-link line only).
- `HANDOFF.md` moved to `docs/history/`. `VENDORED.md` documents the seen-store
  quirk and drops a private-file link.
- `scripts/deploy-sideload.sh` was synced with sage-yolo2's: the drift check is
  per-plugin and the final message is accurate.

## 2.0.0 — 2026-07-15

First release. BioCLIP2 species classifier re-architected onto the sage-yolo2 v2
**cache-consumer** pattern (the "2" mirrors sage-yolo2's re-architecture jump from
its v1). Reads `-v2-` frames from the shared `/local-cache`, classifies each with
BioCLIP-2.5 (TreeOfLife-200M), publishes `env.species.*` frame-anchored.

### The feature
- **Full-frame ↔ crop switch is a single `--input` parameter.** Because
  media-sampler3 frames and sage-yolo2 crops share the v2 format, the same plugin
  classifies either the raw stream (`.../camera/top`) or a crop stream
  (`.../camera-crops/top-crop-0`) with no code change — the detect→classify
  cascade, mediated by the shared cache.

### Added
- `app.py` — v2 cache-consumer wake loop (scan → select → read metadata + identity
  + provenance → classify → publish frame-anchored → mark seen), ported from
  sage-yolo2 with `BioCLIP2Classifier` (pybioclip `TreeOfLifeClassifier`) swapped
  in for the detector.
- **Crop provenance passthrough**: when a frame carries a `source{}` block (a
  sage-yolo2 crop), `source_class`/`source_confidence`/`source_unique_id` are
  attached to the species record — a species result traces to the YOLO detection
  and the parent frame. No-op on plain frames.
- `env.species.<rank>`, `.confidence`, `env.species.top5`, `env.species.summary`
  (ported from sage-bioclip v1); `--save-match` matches scientific OR common name.
- Vendored byte-identical from sage-yolo2 (`VENDORED.md`): `consumer.py`,
  `selection.py`, `seenstore.py`, `node_info.py`, `save_match.py`.
- `patch_pybioclip.py` (BioCLIP-2.5 ViT-H/14 enablement) + `Dockerfile` applying
  it once at build time (the patch is not idempotent — NOT re-run at runtime).
- `scripts/deploy-sideload.sh` (native Thor build → k3s import), reads sage.yaml.

### Tests
`make test` → **91 passed** (offline, no GPU): 85 carried-over
consumer/selection/seenstore/identity + 6 bioclip-specific (publish shape,
cache-wake full-frame, cache-wake crop+provenance, `--input`-switch-is-config-only,
save-match on common name).

### Deploy
ECR portal build fails (CUDA base + QEMU, Infra #3) → native Thor build + k3s
side-load + `pluginctl run`. Runs alongside the sage-yolo2 crop producer on H00F.
