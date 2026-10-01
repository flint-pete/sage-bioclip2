# Changelog

All notable changes to the `sage-bioclip2` Sage plugin.

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
