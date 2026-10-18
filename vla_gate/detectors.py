"""Open-vocabulary detectors and the grounding step.

A detector is any callable  detector(image_pil, phrase, box_threshold, text_threshold) -> list of
{"bbox": [x1, y1, x2, y2], "confidence": float, "label": str}. Three are provided:
  GroundingDINO   - the detector of the first gate (Study 1)
  OWLv2           - better at the bowls, ramekin and plate, misses the cabinet and the stove
  HybridDetector  - OWLv2 for bowls / ramekin / plate, Grounding DINO for cabinet, stove and cookie box (Study 2)
torch / transformers are imported only when a model is loaded.
"""
from .config import (ENGINE_CFG, IMG_H, IMG_W, OWL_RAMP, GD_PHRASES, GD_FALLBACK, GDINO_MODEL_ID, OWLV2_MODEL_ID)
from .geometry import box_iou

# ---------- phrase handling (used by the one re-grounding pass) ----------
SYNONYMS = {
    "black bowl":     ["black bowl", "dark bowl", "bowl"],
    "ramekin":        ["ramekin", "small white bowl", "white cup"],
    "plate":          ["plate", "white plate", "dish"],
    "cookie box":     ["cookie box", "box", "carton"],
    "wooden cabinet": ["wooden cabinet", "cabinet", "drawer"],
    "stove":          ["stove", "cooktop", "burner"],
}


def phrase_variants(phrase, attempt):
    """attempt 0: the phrase itself; attempt 1 (re-grounding) adds known synonyms."""
    base = phrase.strip().lower()
    variants = SYNONYMS.get(base, [base])   # unknown phrases are NOT loosened (avoids hallucinations)
    out = []
    for v in variants[: 1 + attempt]:
        if v not in out:
            out.append(v)
    return out


def clean_detections(dets, min_conf, allow_large=False, cfg=None):
    """Clip boxes, drop low-confidence / giant / arm-shaped boxes, NMS, sort by confidence."""
    cfg = cfg or ENGINE_CFG
    out = []
    for d in dets:
        x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(IMG_W), x2), min(float(IMG_H), y2)
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2 or float(d["confidence"]) < min_conf:
            continue
        if not allow_large:
            if w * h > cfg["max_area_frac"] * IMG_W * IMG_H:
                continue
            if max(w / h, h / w) > cfg["max_aspect"]:
                continue
        out.append({**d, "bbox": [x1, y1, x2, y2], "confidence": float(d["confidence"])})
    out.sort(key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in out:
        if all(box_iou(d["bbox"], k["bbox"]) < cfg["nms_iou"] for k in kept):
            kept.append(d)
    return kept[: cfg["max_candidates"]]


def ground_phrase(detector, image_pil, phrase, attempt=0, cfg=None, min_conf=None):
    """the detector over all phrase variants of this attempt -> cleaned candidate list."""
    cfg = cfg or ENGINE_CFG
    if min_conf is None:
        min_conf = cfg["min_conf"][min(attempt, len(cfg["min_conf"]) - 1)]
    dets = []
    for p in phrase_variants(phrase, attempt):
        for d in detector(image_pil, p, box_threshold=min_conf, text_threshold=cfg["text_threshold"]):
            dets.append({**d, "phrase_used": p})
    return clean_detections(dets, min_conf, allow_large=False, cfg=cfg)


# ---------- detectors ----------
def _default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class GroundingDINO:
    """IDEA-Research/grounding-dino-base through Hugging Face transformers."""
    name = "gdino"

    def __init__(self, model_id=GDINO_MODEL_ID, device=None):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = device or _default_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, image_pil, phrase, box_threshold=0.35, text_threshold=0.25):
        import torch
        text_prompt = phrase.lower().strip() + "."
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],  # (height, width)
        )[0]
        return [{"bbox": [round(v, 1) for v in box.tolist()], "confidence": round(score.item(), 3), "label": label}
                for box, score, label in zip(results["boxes"], results["scores"], results["labels"])]


class OWLv2:
    """google/owlv2-base-patch16-ensemble; same output format as GroundingDINO."""
    name = "owlv2"

    def __init__(self, model_id=OWLV2_MODEL_ID, device=None):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.device = device or _default_device()
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, image_pil, phrase, box_threshold=0.10, text_threshold=None):
        import torch
        inp = self.processor(text=[["a photo of a " + phrase.lower().strip()]], images=image_pil, return_tensors="pt",
                             truncation=True, max_length=16).to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        ts = torch.tensor([[max(image_pil.size), max(image_pil.size)]], device=self.device)
        try:
            r = self.processor.post_process_grounded_object_detection(outputs=out, target_sizes=ts, threshold=box_threshold)[0]
        except Exception:
            r = self.processor.post_process_object_detection(outputs=out, target_sizes=ts, threshold=box_threshold)[0]
        return [{"bbox": [round(v, 1) for v in b.tolist()], "confidence": round(float(s), 3), "label": phrase}
                for b, s in zip(r["boxes"], r["scores"])]


class HybridDetector:
    """OWLv2 for bowls / ramekin / plate; Grounding DINO for cabinet, stove, cookie box (and the plate if OWLv2 finds none).
    Grounding DINO's confidences are moved onto OWLv2's scale (0.22-0.40 -> 0.10-0.25) so one ramp serves both."""
    name = "hybrid"

    def __init__(self, owl, gdino, owl_ramp=OWL_RAMP, gd_phrases=GD_PHRASES, gd_fallback=GD_FALLBACK):
        self.owl, self.gdino = owl, gdino
        self.owl_ramp, self.gd_phrases, self.gd_fallback = tuple(owl_ramp), set(gd_phrases), set(gd_fallback)

    def _gd_rescaled(self, image_pil, phrase, box_threshold):
        out = []
        lo, hi = self.owl_ramp
        for d in self.gdino(image_pil, phrase, box_threshold=0.22, text_threshold=0.20):
            c = lo + (float(d["confidence"]) - 0.22) * (hi - lo) / (0.40 - 0.22)
            if c >= box_threshold:
                out.append({**d, "confidence": round(float(c), 3)})
        return out

    def __call__(self, image_pil, phrase, box_threshold=0.10, text_threshold=None):
        p = phrase.lower().strip()
        if p in self.gd_phrases:
            return self._gd_rescaled(image_pil, phrase, box_threshold)
        dets = self.owl(image_pil, phrase, box_threshold)
        if not dets and p in self.gd_fallback:
            dets = self._gd_rescaled(image_pil, phrase, box_threshold)
        return dets


def load_detector(name="hybrid", device=None):
    """'hybrid' (Study 2, recommended), 'gdino' (first gate, Study 1) or 'owlv2'"""
    if name == "gdino":
        return GroundingDINO(device=device)
    if name == "owlv2":
        return OWLv2(device=device)
    if name == "hybrid":
        return HybridDetector(OWLv2(device=device), GroundingDINO(device=device))
    raise ValueError(f"unknown detector '{name}'")
