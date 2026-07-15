# BioCLIP2 Species Classifier — Cache-Consumer Architecture

## Science

Automated species identification is critical for biodiversity monitoring,
ecological surveys, and conservation management. Traditional approaches require
expert taxonomists to manually review images — slow, expensive, and unable to
scale to the millions of images collected by distributed camera networks. This
plugin brings state-of-the-art vision-language classification to the edge, doing
real-time species identification at the point of data collection.

`sage-bioclip2` is re-architected as a **pywaggle2 cache consumer**: instead of
opening a camera itself, it reads self-describing frames a *producer* already
wrote to the shared on-node cache, classifies each, and publishes taxonomy
predictions **frame-anchored** (observation time = the moment the photo was
taken, not when inference ran). This decouples acquisition from analysis and lets
multiple analytics share one image stream without re-capturing.

## The detect→classify cascade

The headline capability is a **two-stage vision pipeline mediated entirely by the
shared cache**, with no plugin ever calling another:

```
image-sampler2      sage-yolo2 (detect + crop)          sage-bioclip2 (classify)
 camera → cache  →   YOLO detects birds, crops each  →   read crops, BioCLIP2 species
 hummingcam/top      → hummingcam-crops/top-crop-<i>      → env.species.* + provenance
```

`sage-yolo2` detects objects and writes a cropped image of each detection into a
crop cache stream; `sage-bioclip2` consumes those crops and identifies the
species. Because both the raw frames and the crops are the identical self-
describing v2 format, **the cache directory to classify is a single `--input`
parameter** — the same plugin classifies whole frames (`.../hummingcam/top`) or
just the crops (`.../hummingcam-crops/top-crop-0`) with no code change.

When classifying a crop, the crop's detection provenance (the YOLO class,
confidence, and the parent frame's identifier) rides along on the species
record — so a species result is fully traceable back through the detection to the
exact source frame and bounding box.

## About BioCLIP 2.5 Huge

**BioCLIP 2.5 Huge** is a contrastive vision-language model purpose-built for
biological image classification, developed by the
[Imageomics Institute](https://imageomics.org). It uses a **ViT-H/14** backbone
trained on **TreeOfLife-200M** — over 219 million biological images spanning
450,000+ species across the full tree of life.

BioCLIP performs **zero-shot classification**: it needs no retraining for new
species. The model compares an image against pre-computed text embeddings for
every taxon in the TreeOfLife taxonomy, returning ranked predictions with
confidence scores — immediately deployable at any field site without custom
training data.

Key capabilities:
- **450,000+ species** recognized out of the box
- **Any taxonomic rank** — Kingdom down to Species (`--rank`)
- **Zero-shot** — no per-site training or fine-tuning
- **ViT-H/14 backbone** — larger and more accurate than BioCLIP 2
- BioCLIP 2 (ViT-L/14) still available via `--model hf-hub:imageomics/bioclip-2`

## Published data

- `env.species.<rank>` — top predicted taxon at the chosen rank (when confident)
- `env.species.<rank>.confidence` — its confidence (0–1)
- `env.species.top5` — JSON of the top-k predictions (scientific + common name)
- `env.species.summary` — 1 = confident prediction, 0 = none / heartbeat

Each record is frame-anchored and carries `camera`, `model`, `rank`, and — when
the frame is a crop — `source_class` / `source_confidence` / `source_unique_id`
for cascade traceability.

## Reporting vs. saving — two independent decisions

Publishing a measurement is cheap (a few bytes); saving an image to Beehive is
expensive (bandwidth + storage). They are controlled separately:

- **`--min-confidence`** (default 0.1) is the reporting floor for whether the top
  prediction is *published*. It does not control image saving.
- **`--save-match`** is the only thing that saves (uploads) an annotated image —
  an OR-list of `Taxon:confidence` rules matching the scientific OR common name.

Because BioCLIP is zero-shot it has no reject class and can score confidently on
empty frames. Feeding it detector *crops* (the cascade) rather than full frames
sharply improves signal: it only classifies regions a detector already flagged.

## Deployment

Runs on ARM64 Sage Thor / DGX Spark nodes (128 GB unified memory). The NVIDIA
CUDA base image cannot be cross-built by the ECR portal (QEMU crash) — the plugin
is built natively on the node and side-loaded into k3s. It runs alongside the
image-sampler2 producer and the sage-yolo2 crop producer to form the live
cascade.
