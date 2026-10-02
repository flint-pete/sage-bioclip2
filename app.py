"""
sage-bioclip2 — BioCLIP2 Species Classifier, v2 cache-consumer architecture.

Reads self-describing `-v2-` frames a producer (media-sampler3, THE producer for
both images and audio) wrote to the shared on-node `/local-cache`, runs BioCLIP2
(TreeOfLife) species classification on each, and publishes taxonomy predictions
frame-anchored (observation time = capture time).

THE SWITCH: the cache dir is a single `--input` parameter, so the same plugin
classifies either full media-sampler3 frames OR sage-yolo2 crops with no code
change:
  --input /local-cache/camera/top             # full frames
  --input /local-cache/camera-crops/top-crop-0 # yolo2 crops (the cascade)
Both are v2 caches; only the path differs.

Architecture mirrors sage-yolo2: the cache-consumer machinery
(consumer/selection/seenstore/node_info) is vendored byte-identical from
sage-yolo2 — see VENDORED.md. This file is the bioclip-specific brains.

Measurement topics:
  env.species.<rank>            — top predicted taxon at the chosen rank
  env.species.<rank>.confidence — its confidence (0-1)
  env.species.top5              — JSON of top-k predictions
  env.species.summary           — heartbeat / per-frame summary
  upload                        — annotated JPEG (when --save-match matches)
"""
import argparse
import json
import logging
import os
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from waggle.plugin import Plugin

from save_match import parse_save_match, should_save, SaveMatchError
import consumer
import selection
import seenstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sage-bioclip2")

RANK_NAMES = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


# ── BioCLIP2 classifier (grafted from sage-bioclip v1) ────────────────────────

def resolve_device(request="auto"):
    """'auto' -> 'cuda' when the pod can see a GPU, else 'cpu'. 'cuda'/'cpu' are honoured
    as given ('cuda' with no GPU fails loudly at model load rather than silently
    falling back). torch is imported lazily so offline unit tests don't need it."""
    if request != "auto":
        return request
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken -> CPU
        return "cpu"


class BioCLIP2Classifier:
    """BioCLIP2 species classifier. Uses pybioclip's TreeOfLifeClassifier.

    Heavy model construction is deferred to load() so it can be timed inside the
    Plugin context. bioclip is imported lazily in load() so the module (and its
    offline unit tests) import without the GPU stack present.
    """

    def __init__(self, rank="Species", model_str="hf-hub:imageomics/bioclip-2.5-vith14",
                 device="auto"):
        if rank not in RANK_NAMES:
            raise ValueError(f"Invalid rank '{rank}'. Must be one of: {RANK_NAMES}")
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"Invalid device '{device}'. Must be auto, cuda or cpu")
        self.rank = rank
        self.model_str = model_str
        self.device_request = device
        self.device = None
        self.classifier = None
        self._rank_enum = None

    def load(self):
        # BioCLIP-2.5 (ViT-H/14) needs the pybioclip patch (see patch_pybioclip.py),
        # which is applied ONCE at image-build time (Dockerfile) — NOT here: the
        # patch is not idempotent (it asserts the pristine source), so it must not
        # run twice. Import bioclip lazily to keep module import light for tests.
        from bioclip import Rank
        from bioclip.predict import TreeOfLifeClassifier
        self._rank_enum = getattr(Rank, self.rank.upper())
        self.device = resolve_device(self.device_request)
        logger.info("Loading BioCLIP2 classifier (model=%s, rank=%s) on %s...",
                    self.model_str, self.rank, self.device)
        # pybioclip's own default is device='cpu', so pass it explicitly (as sage-yolo2
        # does for YOLO): the GPU is ~15x faster per crop on a Thor (H039, Oct 2026).
        self.classifier = TreeOfLifeClassifier(model_str=self.model_str, device=self.device)
        logger.info("BioCLIP2 classifier loaded on %s", self.device)

    def classify(self, image, top_k=5):
        """Classify a PIL image at the configured rank.
        Returns [{name, common_name, confidence}, ...] sorted descending."""
        results = self.classifier.predict(images=[image], rank=self._rank_enum, k=top_k)
        rank_key = self.rank.lower()
        key = "species" if rank_key == "species" else rank_key
        preds = [
            {"name": r.get(key, r.get("genus", "Unknown")),
             "common_name": r.get("common_name", ""),
             "confidence": float(r["score"])}
            for r in results
        ]
        return preds[:top_k]


# ── annotation (adapted from v1) ──────────────────────────────────────────────

def annotate(frame, predictions, *, min_confidence):
    """Draw top predictions (orange) on a copy of the BGR frame."""
    out = frame.copy()
    if predictions and predictions[0]["confidence"] >= min_confidence:
        for i, p in enumerate(predictions[:3]):
            label = "%s %.1f%%" % (p["name"], p["confidence"] * 100)
            if p.get("common_name"):
                label = "%s (%s) %.1f%%" % (p["name"], p["common_name"],
                                            p["confidence"] * 100)
            cv2.putText(out, label, (8, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    else:
        best = predictions[0]["confidence"] if predictions else 0.0
        cv2.putText(out, "No confident prediction (best %.1f%%)" % (best * 100),
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    return out


# ── image sources ─────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def iter_image_dir(directory):
    """Yield (path, bgr_frame, capture_ts_ns) for images in a dir (offline test)."""
    if os.path.isfile(directory):
        files = [directory]
    else:
        files = sorted(os.path.join(directory, f) for f in os.listdir(directory)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
    for path in files:
        img = cv2.imread(path)
        if img is None:
            logger.warning("could not read image: %s", path)
            continue
        yield path, img, time.time_ns()


# ── consumer-id (from sage-yolo2) ─────────────────────────────────────────────

def resolve_consumer_id(override):
    if override:
        return override
    job = os.environ.get("WAGGLE_JOB_NAME", "").strip()
    task = os.environ.get("WAGGLE_TASK_NAME", "").strip()
    if job and task:
        return "%s-%s" % (job, task)
    app_id = os.environ.get("WAGGLE_APP_ID", "").strip()
    if app_id:
        logger.warning("no WAGGLE_JOB_NAME/TASK_NAME; using WAGGLE_APP_ID as consumer-id")
        return app_id
    logger.warning("no consumer identity in env; using 'default' consumer-id")
    return "default"


def parse_cache_input(input_path, cache_root):
    """Split <root>/<cache-name>/<camera> → (cache_name, camera)."""
    rel = os.path.relpath(os.path.abspath(input_path), os.path.abspath(cache_root))
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return (parts[0] if parts else "cache"), os.path.basename(input_path.rstrip("/"))


def main():
    parser = argparse.ArgumentParser(
        description="BioCLIP2 species classifier — v2 cache consumer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Production: classify sage-yolo2 crops (the detect->classify cascade)
  python3 app.py --source cache --input /local-cache/camera-crops/top-crop-0 \\
      --every 10m --all-unseen --rank Species

  # Same plugin, full frames — change ONE arg:
  python3 app.py --source cache --input /local-cache/camera/top --rank Species

  # Local testing on a folder of images
  python3 app.py --source image-dir --input ../sage-yolo2/tests/test-images --rank Species
""")
    parser.add_argument("--source", required=True,
                        choices=["cache", "image-dir"],
                        help="cache = consume v2 frames from the shared cache "
                             "(production); image-dir = local test folder.")
    parser.add_argument("--input", required=True,
                        help="cache dir <root>/<cache-name>/<camera> (raw OR crop "
                             "stream) | directory of test images.")
    parser.add_argument("--every", default="0",
                        help="Wake cadence. 0 = single-shot. Accepts s/m/h.")
    parser.add_argument("--select-every", default="0",
                        help="Sampling stride (one frame per this much capture-time). "
                             "0 = the newest unseen frame.")
    parser.add_argument("--max-frames", type=int, default=1,
                        help="Cap frames per wake (0 = unlimited).")
    parser.add_argument("--all-unseen", action="store_true",
                        help="Backlog mode: classify every not-yet-seen frame.")
    parser.add_argument("--max-runtime", type=int, default=0,
                        help="Wall-clock bound in seconds (0 = forever).")
    parser.add_argument("--consumer-id", default=None,
                        help="Override the seen-store consumer-id segment.")
    parser.add_argument("--seen-store", default=None, help="Override seen-store path.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Ignore the seen-store (still records what it processes).")

    parser.add_argument("--rank", default="Species", choices=RANK_NAMES,
                        help="Taxonomic rank to predict. Default Species.")
    parser.add_argument("--model", default="hf-hub:imageomics/bioclip-2.5-vith14",
                        help="BioCLIP model string.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Where to run the model: auto = GPU if the pod can see one, else CPU.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k predictions.")
    parser.add_argument("--min-confidence", type=float, default=0.1,
                        help="Below this, treat as no-confident-prediction.")
    parser.add_argument("--upload-image", default="Y",
                        help="Y = allow annotated uploads, N = never. Governed by "
                             "--save-match when set.")
    parser.add_argument("--save-match", default="",
                        help="OR-list of 'Taxon:confidence' rules (matches scientific "
                             "OR common name), or '*:0.5'. Upload the annotated frame "
                             "when the top prediction matches.")
    args = parser.parse_args()

    try:
        save_rules = parse_save_match(args.save_match)
    except SaveMatchError as e:
        logger.error("Invalid --save-match: %s", e)
        raise SystemExit(2)
    try:
        every_s = selection.parse_duration(args.every)
        select_every_s = selection.parse_duration(args.select_every)
    except ValueError as e:
        logger.error("Invalid duration: %s", e)
        raise SystemExit(2)

    classifier = BioCLIP2Classifier(rank=args.rank, model_str=args.model, device=args.device)

    is_cache = args.source == "cache"
    is_image_dir = args.source == "image-dir"

    seen = None
    if is_cache:
        try:
            consumer.assert_cache_available(args.input)
        except consumer.CacheError as e:
            logger.error("%s", e)
            raise SystemExit(2)
        cache_root = consumer.resolve_cache_root()
        cache_name, camera = parse_cache_input(args.input, cache_root)
        consumer_id = resolve_consumer_id(args.consumer_id)
        store_path = args.seen_store or seenstore.seen_store_path(
            cache_root, consumer_id, cache_name, camera)
        seen = seenstore.SeenStore(store_path, reprocess=args.reprocess)
        logger.info("cache consumer: input=%s consumer-id=%s seen-store=%s (%d known)",
                    args.input, consumer_id, store_path, len(seen))

    with Plugin() as plugin:
        logger.info("sage-bioclip2 started — source=%s input=%s model=%s rank=%s every=%ds",
                    args.source, args.input, args.model, args.rank, every_s)
        with plugin.timeit("plugin.duration.loadmodel"):
            classifier.load()

        deadline = None
        if every_s > 0 and args.max_runtime > 0:
            deadline = time.monotonic() + args.max_runtime

        last_wake_ts_ns = 0
        img_iter = iter_image_dir(args.input) if is_image_dir else None

        while True:
            try:
                if is_cache:
                    _process_cache_wake(plugin, classifier, args, save_rules, seen,
                                        last_wake_ts_ns, select_every_s)
                    last_wake_ts_ns = time.time_ns()
                else:
                    if not _process_image_dir(plugin, classifier, args, save_rules,
                                              img_iter):
                        break
            except Exception:
                logger.exception("wake error")

            if every_s == 0 and not is_image_dir:
                break
            if deadline is not None and time.monotonic() + every_s >= deadline:
                logger.info("Max runtime reached — self-exiting to free the GPU")
                break
            if not is_image_dir:
                time.sleep(every_s)


def _publish_species(plugin, args, predictions, *, timestamp, camera, identity=None,
                     source=None):
    """Publish env.species.* frame-anchored. `source` = the crop's provenance blob
    (present only when classifying a yolo2 crop); attached to meta for traceability."""
    rank_lower = args.rank.lower()
    base_meta = {"camera": camera, "model": args.model, "rank": args.rank}
    if identity is not None:
        if identity.vsn:
            base_meta["vsn"] = identity.vsn
        if identity.node_id:
            base_meta["node_id"] = identity.node_id
        if identity.has_location:
            base_meta["lat"] = str(identity.lat)
            base_meta["lon"] = str(identity.lon)
            base_meta["location_source"] = identity.location_source
    if source:                              # crop provenance → species traceability
        if source.get("source_class"):
            base_meta["source_class"] = str(source["source_class"])
        if source.get("source_confidence") is not None:
            base_meta["source_confidence"] = str(source["source_confidence"])
        if source.get("source_unique_id"):
            base_meta["source_unique_id"] = str(source["source_unique_id"])

    if not predictions:
        plugin.publish("env.species.summary", 0, timestamp=timestamp,
                       meta=dict(base_meta, note="no_predictions"))
        return

    top = predictions[0]
    confident = top["confidence"] >= args.min_confidence
    if confident:
        m = dict(base_meta)
        if top.get("common_name"):
            m["common_name"] = top["common_name"]
        plugin.publish("env.species.%s" % rank_lower, top["name"],
                       timestamp=timestamp, meta=m)
        plugin.publish("env.species.%s.confidence" % rank_lower, top["confidence"],
                       timestamp=timestamp, meta=dict(base_meta))
        logger.info("Published env.species.%s = %s (%.1f%%)", rank_lower,
                    top["name"], top["confidence"] * 100)
    plugin.publish("env.species.top5",
                   json.dumps(predictions[:args.top_k], separators=(",", ":")),
                   timestamp=timestamp,
                   meta=dict(base_meta, confident=str(confident)))
    plugin.publish("env.species.summary", 1 if confident else 0, timestamp=timestamp,
                   meta=dict(base_meta, top=str(top["name"])))


def _maybe_upload(plugin, args, predictions, frame, *, timestamp, camera, save_rules):
    """Upload the annotated frame per --save-match (matches name OR common_name)."""
    if not predictions:
        return
    if save_rules:
        # match on either scientific or common name of any prediction
        cand = [{"class": p["name"], "confidence": p["confidence"]} for p in predictions]
        cand += [{"class": p["common_name"], "confidence": p["confidence"]}
                 for p in predictions if p.get("common_name")]
        do_upload = should_save(save_rules, cand, name_keys=["class"])
    else:
        do_upload = args.upload_image == "Y"
    if not do_upload:
        return
    annotated = annotate(frame, predictions, min_confidence=args.min_confidence)
    tmp = os.path.join(tempfile.gettempdir(), "%s-species.jpg" % camera)
    cv2.imwrite(tmp, annotated)
    top = predictions[0]
    plugin.upload_file(tmp, timestamp=timestamp,
                       meta={"camera": camera, "top": str(top["name"]),
                             "confidence": str(top["confidence"])})
    if os.path.exists(tmp):
        os.unlink(tmp)
    logger.info("Uploaded annotated species image (top=%s)", top["name"])


def _bgr_to_pil(img):
    return Image.fromarray(img[:, :, ::-1])   # BGR→RGB


def _process_cache_wake(plugin, classifier, args, save_rules, seen,
                        last_wake_ts_ns, select_every_s):
    """One cache wake: scan → select → per frame read meta+identity+provenance,
    classify, publish frame-anchored, mark seen."""
    frames = consumer.scan_frames(args.input)
    selected = selection.select_frames(
        frames, last_wake_ts_ns=last_wake_ts_ns,
        select_every_ns=select_every_s * 1_000_000_000,
        all_unseen=args.all_unseen, max_frames=args.max_frames,
        seen=seen, reprocess=args.reprocess, uid_of=_frame_uid)
    if not selected:
        logger.info("cache wake: 0 frames to classify")
        return
    node_info = consumer.get_node_info()
    for frame in selected:
        meta = consumer.read_frame_metadata(frame)
        identity = consumer.resolve_identity(meta, node_info=node_info)
        source = _read_source_provenance(frame)     # crop provenance or None
        img = cv2.imread(frame.path)
        if img is None:
            logger.warning("frame vanished before read: %s", frame.name)
            continue
        with plugin.timeit("plugin.duration.inference"):
            predictions = classifier.classify(_bgr_to_pil(img), top_k=args.top_k)
        ts = meta.capture_ts_ns
        cam = meta.camera or frame.camera
        _publish_species(plugin, args, predictions, timestamp=ts, camera=cam,
                         identity=identity, source=source)
        _maybe_upload(plugin, args, predictions, img, timestamp=ts, camera=cam,
                      save_rules=save_rules)
        if meta.unique_id:
            seen.mark(meta.unique_id)


def _read_source_provenance(frame):
    """Return the crop's nested `source` dict from its UserComment JSON, or None
    (a plain media-sampler3 frame has no `source` → full-frame mode)."""
    try:
        payload = consumer._extract_usercomment_json(frame.path)
        return (payload or {}).get("source") or None
    except Exception:
        return None


def _frame_uid(frame):
    meta = consumer.read_frame_metadata(frame)
    return meta.unique_id or frame.name


def _process_image_dir(plugin, classifier, args, save_rules, img_iter):
    """One image from the local test dir. Returns False when exhausted."""
    with plugin.timeit("plugin.duration.input"):
        try:
            img_path, frame, timestamp = next(img_iter)
        except StopIteration:
            logger.info("All test images processed")
            return False
    camera = os.path.splitext(os.path.basename(img_path))[0]
    with plugin.timeit("plugin.duration.inference"):
        predictions = classifier.classify(_bgr_to_pil(frame), top_k=args.top_k)
    _publish_species(plugin, args, predictions, timestamp=timestamp, camera=camera)
    _maybe_upload(plugin, args, predictions, frame, timestamp=timestamp,
                  camera=camera, save_rules=save_rules)
    return True


if __name__ == "__main__":
    main()
