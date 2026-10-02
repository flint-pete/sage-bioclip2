#!/usr/bin/env python3
"""Offline unit tests for sage-bioclip2 app.py (no GPU, no bioclip model).

app.py imports the runtime stack (cv2, numpy, PIL, waggle) at module top; we
inject light stubs before import (same pattern as sage-yolo2's test_app_cache).
The BioCLIP2Classifier is replaced with a stub returning fixed predictions, so we
test the WIRING: cache wake → read frame → classify → publish env.species.* frame-
anchored → mark seen; the full-frame↔crop --input switch; and crop `source{}`
provenance attachment to species records.
"""
import io
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

piexif = pytest.importorskip("piexif")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _install_stubs():
    cv2 = types.ModuleType("cv2")
    cv2.imread = lambda p: (np.full((40, 40, 3), 120, dtype=np.uint8)
                            if os.path.exists(p) else None)
    cv2.imwrite = lambda p, i: True
    cv2.putText = lambda *a, **k: None
    cv2.FONT_HERSHEY_SIMPLEX = 0
    sys.modules["cv2"] = cv2

    waggle = types.ModuleType("waggle")
    wplugin = types.ModuleType("waggle.plugin"); wplugin.Plugin = object
    wdata = types.ModuleType("waggle.data")
    wvision = types.ModuleType("waggle.data.vision"); wvision.Camera = object
    sys.modules.update({"waggle": waggle, "waggle.plugin": wplugin,
                        "waggle.data": wdata, "waggle.data.vision": wvision})


_install_stubs()
import app          # noqa: E402
import consumer     # noqa: E402


class _Timeit:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakePlugin:
    def __init__(self):
        self.published = []
        self.uploads = []
    def timeit(self, name): return _Timeit()
    def publish(self, topic, value, timestamp=None, meta=None):
        self.published.append((topic, value, timestamp, dict(meta or {})))
    def upload_file(self, path, timestamp=None, meta=None):
        self.uploads.append((path, timestamp, dict(meta or {})))


class StubClassifier:
    """Returns a fixed confident prediction for every image."""
    def __init__(self, preds=None):
        self.preds = preds or [
            {"name": "Cardinalis cardinalis", "common_name": "Northern Cardinal",
             "confidence": 0.87},
            {"name": "Cyanocitta cristata", "common_name": "Blue Jay",
             "confidence": 0.05},
        ]
    def load(self): pass
    def classify(self, image, top_k=5): return self.preds[:top_k]


class Args:
    rank = "Species"
    model = "hf-hub:imageomics/bioclip-2.5-vith14"
    top_k = 5
    min_confidence = 0.1
    upload_image = "N"
    save_match = ""
    all_unseen = True
    max_frames = 0
    reprocess = False
    def __init__(self, **kw): self.__dict__.update(kw)


# --- helpers: write a real v2 frame (optionally a crop with source{}) ---------

def _write_v2_frame(dir_path, ts_ns, vsn, camera, *, unique_id, source=None):
    os.makedirs(dir_path, exist_ok=True)
    img = Image.new("RGB", (40, 40), (100, 100, 100))
    payload = {
        "schema_version": "2.0", "vsn": vsn, "node_id": "nodeX",
        "job": "j", "task": "t", "plugin": "sage-yolo2",
        "camera": camera, "capture_timestamp_ns": ts_ns,
        "unique_id": unique_id, "acquisition_path": "test",
    }
    if source is not None:
        payload["source"] = source
    uc = b"ASCII\x00\x00\x00" + json.dumps(payload).encode("ascii")
    exif = {"Exif": {piexif.ExifIFD.UserComment: uc}, "0th": {}, "1st": {},
            "GPS": {}, "Interop": {}}
    name = "%d-v2-%s-%s.jpg" % (ts_ns, vsn, camera)
    path = os.path.join(dir_path, name)
    b = io.BytesIO(); img.save(b, "JPEG")
    piexif.insert(piexif.dump(exif), b.getvalue(), path)
    return path


# --- publish shape ------------------------------------------------------------

def test_publish_species_confident():
    p = FakePlugin()
    app._publish_species(p, Args(), StubClassifier().preds, timestamp=123,
                         camera="top")
    topics = {t for t, *_ in p.published}
    assert "env.species.species" in topics
    assert "env.species.species.confidence" in topics
    assert "env.species.top5" in topics
    sp = [x for x in p.published if x[0] == "env.species.species"][0]
    assert sp[1] == "Cardinalis cardinalis"
    assert sp[2] == 123                       # frame-anchored
    assert sp[3]["common_name"] == "Northern Cardinal"


def test_publish_species_below_confidence_no_rank_topic():
    p = FakePlugin()
    low = [{"name": "X", "common_name": "", "confidence": 0.02}]
    app._publish_species(p, Args(), low, timestamp=1, camera="top")
    topics = {t for t, *_ in p.published}
    assert "env.species.species" not in topics   # not confident
    assert "env.species.top5" in topics          # still publishes top5 + summary


# --- cache wake end-to-end (full frame) ---------------------------------------

def test_cache_wake_full_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(consumer, "resolve_cache_root", lambda explicit=None: str(tmp_path))
    monkeypatch.setattr(consumer, "get_node_info", lambda: None)
    cam_dir = os.path.join(str(tmp_path), "camera", "top")
    _write_v2_frame(cam_dir, 1700000000000000000, "H00F", "top", unique_id="uidfull")

    import seenstore
    seen = seenstore.SeenStore(os.path.join(str(tmp_path), "seen"), reprocess=False)
    p = FakePlugin()
    args = Args()
    args.input = cam_dir
    app._process_cache_wake(p, StubClassifier(), args, [], seen, 0, 0)

    sp = [x for x in p.published if x[0] == "env.species.species"]
    assert len(sp) == 1 and sp[0][1] == "Cardinalis cardinalis"
    assert sp[0][2] == 1700000000000000000       # frame-anchored to capture ts
    assert "source_class" not in sp[0][3]         # full frame → no provenance
    assert "uidfull" in seen._set                  # marked seen


# --- cache wake end-to-end (CROP, with provenance) ----------------------------

def test_cache_wake_crop_attaches_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(consumer, "resolve_cache_root", lambda explicit=None: str(tmp_path))
    monkeypatch.setattr(consumer, "get_node_info", lambda: None)
    crop_dir = os.path.join(str(tmp_path), "camera-crops", "top-crop-0")
    src = {"source_class": "bird", "source_confidence": 0.91,
           "source_bbox": [10, 10, 30, 30], "source_unique_id": "parentuid",
           "detection_index": 0}
    _write_v2_frame(crop_dir, 1700000000000000000, "H00F", "top-crop-0",
                    unique_id="cropuid", source=src)

    import seenstore
    seen = seenstore.SeenStore(os.path.join(str(tmp_path), "seen2"), reprocess=False)
    p = FakePlugin()
    args = Args()
    args.input = crop_dir              # THE SWITCH: crop dir instead of full-frame dir
    app._process_cache_wake(p, StubClassifier(), args, [], seen, 0, 0)

    sp = [x for x in p.published if x[0] == "env.species.species"][0]
    # provenance from the crop's source{} is attached to the species record
    assert sp[3]["source_class"] == "bird"
    assert sp[3]["source_unique_id"] == "parentuid"
    assert sp[2] == 1700000000000000000           # still frame-anchored


# --- the --input switch is the ONLY difference (same code path both ways) ------

def test_input_switch_is_config_only(tmp_path, monkeypatch):
    """Full-frame and crop dirs run the identical wake code — only --input differs.
    Proven by both producing an env.species.species record with no branching."""
    monkeypatch.setattr(consumer, "resolve_cache_root", lambda explicit=None: str(tmp_path))
    monkeypatch.setattr(consumer, "get_node_info", lambda: None)
    import seenstore
    for i, (sub, source) in enumerate([
            (("camera", "top"), None),
            (("camera-crops", "top-crop-0"),
             {"source_class": "bird", "source_confidence": 0.9,
              "source_bbox": [1, 1, 9, 9], "source_unique_id": "pu", "detection_index": 0}),
    ]):
        d = os.path.join(str(tmp_path), *sub)
        _write_v2_frame(d, 1700000000000000000 + i, "H00F", sub[-1],
                        unique_id="u%d" % i, source=source)
        seen = seenstore.SeenStore(os.path.join(str(tmp_path), "s%d" % i), reprocess=False)
        p = FakePlugin()
        args = Args(); args.input = d
        app._process_cache_wake(p, StubClassifier(), args, [], seen, 0, 0)
        assert any(x[0] == "env.species.species" for x in p.published), \
            "input=%s produced no species record" % d


# --- save-match matches on common name ----------------------------------------

def test_save_match_on_common_name(tmp_path):
    from save_match import parse_save_match
    p = FakePlugin()
    frame = np.full((40, 40, 3), 120, dtype=np.uint8)
    # rule targets the common name "Northern Cardinal" (an example bird species)
    rules = parse_save_match("Northern Cardinal:0.5")
    app._maybe_upload(p, Args(upload_image="N"), StubClassifier().preds, frame,
                      timestamp=1, camera="top", save_rules=rules)
    assert len(p.uploads) == 1                     # matched on common name


# ── device selection (GPU when the pod can see one) ───────────────────────────
def _fake_torch(monkeypatch, cuda):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_resolve_device_auto_picks_cuda_when_available(monkeypatch):
    _fake_torch(monkeypatch, True)
    assert app.resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    _fake_torch(monkeypatch, False)
    assert app.resolve_device("auto") == "cpu"


def test_resolve_device_explicit_is_honoured(monkeypatch):
    _fake_torch(monkeypatch, False)
    assert app.resolve_device("cuda") == "cuda"
    assert app.resolve_device("cpu") == "cpu"


def test_classifier_passes_device_to_pybioclip(monkeypatch):
    _fake_torch(monkeypatch, True)
    seen = {}

    class FakeTOL:
        def __init__(self, model_str, device="cpu"):
            seen["model_str"], seen["device"] = model_str, device

    bioclip = types.ModuleType("bioclip")
    bioclip.Rank = types.SimpleNamespace(SPECIES="SPECIES")
    predict = types.ModuleType("bioclip.predict")
    predict.TreeOfLifeClassifier = FakeTOL
    monkeypatch.setitem(sys.modules, "bioclip", bioclip)
    monkeypatch.setitem(sys.modules, "bioclip.predict", predict)
    c = app.BioCLIP2Classifier(rank="Species")
    c.load()
    assert seen["device"] == "cuda" and c.device == "cuda"


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        app.BioCLIP2Classifier(device="tpu")
