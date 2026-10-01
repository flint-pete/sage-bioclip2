# Vendored code

Modules vendored **byte-identical** from `sage-yolo2` (check with the loop below) — this is the v2
cache-consumer pattern, shared across the v2 plugin family. sage-bioclip2 is a
second consumer of the same contract, so it reuses the same read-side machinery.

| Module | Source | Role |
|---|---|---|
| `consumer.py` | sage-yolo2 (byte-identical) | v2 cache read: scan/parse/newest, EXIF+UserComment metadata, identity resolution |
| `selection.py` | sage-yolo2 (byte-identical) | frame selection (stride / all-unseen / newest), duration parsing |
| `seenstore.py` | sage-yolo2 (byte-identical) | dedup memory (seen unique_ids), bounded, restart-durable |
| `node_info.py` | sage-yolo2 (byte-identical) | pod self-identity from WAGGLE_NODE_* env (itself vendored from pywaggle2-nodeinfo) |
| `save_match.py` | sage-yolo2 (byte-identical; also identical in sage-bioclip v1) | `Class:confidence` OR-rule parsing + matching |

## Verify byte-identity

```sh
Y=../sage-yolo2
for m in consumer.py selection.py seenstore.py node_info.py save_match.py; do
  diff "$m" "$Y/$m" && echo "$m: identical" || echo "$m: DRIFT"
done
```

## Sync obligation

These modules are the **v2 read contract**. sage-bioclip2 reads exactly the v2
frames sage-yolo2 (and media-sampler3) write — including sage-yolo2 crops. If the
v2 format changes (media-sampler3 `metadata.py` / sage-yolo2 `crop_writer.py`),
re-vendor these from sage-yolo2 and re-run `make test`. The carried-over
`tests/test_consumer*.py` / `test_selection.py` / `test_seenstore.py` /
`test_identity.py` are the contract guard.

**Known quirk:** the vendored `seenstore.py` hard-codes `PLUGIN_NAME = "sage-yolo2"`,
so sage-bioclip2's seen-store lives under `/local-cache/.state/sage-yolo2/<consumer-id>/...`
(its consumer-id, e.g. `camera-sage-bioclip2`, keeps it separate from yolo2's own
store). Changing it would orphan existing seen-memory on deployed nodes, so it is
documented rather than changed.

## bioclip-specific (NOT vendored — grafted from sage-bioclip v1)

- `patch_pybioclip.py` — enables BioCLIP-2.5 ViT-H/14 in pybioclip 2.1.5 (patches
  library internals at image-build time). Copied from sage-bioclip v1.
- `app.py::BioCLIP2Classifier` — the pybioclip `TreeOfLifeClassifier` wrapper +
  `env.species.*` publish logic, ported from sage-bioclip v1 onto the v2 wake loop.

## Future: extract a shared package

Vendoring was chosen (matching sage-yolo2's `crop_writer` precedent) to keep this
a single-repo build. With TWO consumers now on the same contract, extracting a
shared `sage-cache-consumer` package is the cleaner long-term move — tracked as a
follow-up (sage-bioclip2 is now proven on-node; the extraction has not been done).
