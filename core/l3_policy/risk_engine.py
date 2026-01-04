import time
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from core.l1_shield.frame_processor import FrameProcessor
from core.l2_brain.llm_client import ContentReviewer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("L3-RiskEngine")

LABEL_MAP = {
    "risky_content": "违规引流/手写内容",
    "qr_code": "二维码",
    "wechat_id": "微信号",
    "wechat_qr": "微信二维码",
    "knife": "刀具",
    "scissors": "剪刀",
    "gun": "枪支",
    "fire": "火焰/火",
}

HARD_BLOCK_LABELS = {"risky_content", "qr_code", "wechat_id", "wechat_qr", "knife", "scissors", "gun", "fire"}
MAX_SCORE_WITHOUT_EVIDENCE = 30


class RiskEngine:
    def __init__(self, api_key: Optional[str] = None, l2_cooldown: float = 2.0):
        logger.info("初始化风控引擎...")
        self.l1 = FrameProcessor()
        self.l2 = ContentReviewer(api_key=api_key)
        self.policy_rules = self._load_policy()

        self._l2_cache: Dict[str, Any] = {}
        self._l2_cooldown = float(l2_cooldown)

    def _load_policy(self) -> Dict[str, Any]:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "rules.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return {r["category"]: r for r in (config.get("rules", []) or []) if isinstance(r, dict) and r.get("category")}
        except Exception:
            return {"Normal": {"risk_threshold": 999, "action": "pass", "label_cn": "合规"}}

    @staticmethod
    def _normalize_label(label: str) -> str:
        return (label or "").strip().lower()

    def _extract_objects_labels(self, l1_objects):
        labels, objects = [], []
        if not l1_objects:
            return labels, objects

        if isinstance(l1_objects, list) and l1_objects and isinstance(l1_objects[0], dict):
            for o in l1_objects:
                lab = self._normalize_label(str(o.get("label", "")))
                if not lab:
                    continue
                objects.append({"label": lab, "conf": o.get("conf", None), "bbox": o.get("bbox", None)})
                labels.append(lab)
            return labels, objects

        if isinstance(l1_objects, list) and l1_objects and isinstance(l1_objects[0], str):
            for lab in l1_objects:
                lab = self._normalize_label(lab)
                labels.append(lab)
                objects.append({"label": lab, "conf": None, "bbox": None})
            return labels, objects

        return labels, objects

    def _objects_cn(self, labels):
        return [LABEL_MAP.get(l, l) for l in labels]

    @staticmethod
    def _frame_md5(path: str) -> Optional[str]:
        try:
            if not path:
                return None
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _get_cached(self, key: str):
        now = time.time()
        item = self._l2_cache.get(key)
        if not item:
            return None
        ts, res = item
        if now - ts < self._l2_cooldown:
            return res
        return None

    def _set_cached(self, key: str, res: dict):
        self._l2_cache[key] = (time.time(), res)

    def detect_and_judge(self, frame_path: str, l2_threshold_override: int = None, l1_hard_block: bool = True):
        t0 = time.time()

        # 1) L1
        l1_res = self.l1.process_frame(frame_path)
        labels, objects = self._extract_objects_labels(l1_res.get("objects", []))
        texts = l1_res.get("texts", []) or []
        hs = l1_res.get("hit_summary", {}) or {}

        skin_trigger = bool(hs.get("skin_trigger", False))
        skin_ratio = float(hs.get("skin_ratio", 0.0))

        # 2) L1 强证据
        hard_hits = [o for o in objects if self._normalize_label(o.get("label", "")) in HARD_BLOCK_LABELS]
        l1_has_hard_risk = len(hard_hits) > 0

        if l1_hard_block and l1_has_hard_risk:
            final = {
                "risk_score": 95,
                "action": "拦截",
                "reason": f"L1 强证据命中：{', '.join([LABEL_MAP.get(h.get('label'), h.get('label')) for h in hard_hits[:4]])}",
                "evidence": {
                    "l1": {
                        "hard_hits": hard_hits[:10],
                        "labels": labels[:30],
                        "labels_cn": self._objects_cn(labels[:30]),
                        "hit_summary": hs,
                    },
                    "l2": {},
                    "time_ms": int((time.time() - t0) * 1000),
                },
            }
            l1_res["final_decision"] = final
            return l1_res

        # 3) L2 text
        l2_text_res = None
        if texts or labels:
            md5 = self._frame_md5(frame_path)
            key = f"{md5}:l2_text" if md5 else None
            if key:
                cached = self._get_cached(key)
                if cached is None:
                    logger.info("触发 L2 文本/引流审计...")
                    l2_text_res = self.l2.review_from_l1(l1_res)
                    self._set_cached(key, l2_text_res)
                else:
                    l2_text_res = cached
            else:
                l2_text_res = self.l2.review_from_l1(l1_res)

        # 4) L2 sex (gated by skin_trigger)
        l2_sex_res = None
        if skin_trigger:
            md5 = self._frame_md5(frame_path)
            key = f"{md5}:l2_sex" if md5 else None
            if key:
                cached = self._get_cached(key)
                if cached is None:
                    logger.info(f"触发 L2 擦边视觉审计... (skin_ratio={skin_ratio:.3f})")
                    l2_sex_res = self.l2.review_visual_sexual(frame_path)
                    self._set_cached(key, l2_sex_res)
                else:
                    l2_sex_res = cached
            else:
                l2_sex_res = self.l2.review_visual_sexual(frame_path)

        # 5) Fusion: pick highest risk_score
        candidates = []
        if isinstance(l2_text_res, dict):
            candidates.append(("text", l2_text_res))
        if isinstance(l2_sex_res, dict):
            candidates.append(("sex", l2_sex_res))

        final_action = "通过"
        final_score = 0
        final_reason = "画面极其纯净"
        final_cat = "Normal"
        chosen = None

        if candidates:
            candidates.sort(key=lambda x: int(x[1].get("risk_score", 0) or 0), reverse=True)
            chosen_src, chosen = candidates[0]

            score = int(chosen.get("risk_score", 0) or 0)
            category = str(chosen.get("category", "Normal") or "Normal")
            evidence_strength = int(chosen.get("evidence_strength", 0) or 0)
            evidence_spans = chosen.get("evidence_spans", []) or []
            visual_evidence = str(chosen.get("visual_evidence", "") or "").strip()
            is_whitelist = bool(chosen.get("is_whitelist", False))

            has_any_evidence = (len(evidence_spans) > 0) or (len(visual_evidence) > 0)

            if evidence_strength < 2 or (not has_any_evidence):
                score = min(score, MAX_SCORE_WITHOUT_EVIDENCE)
                final_action = "通过"
                final_score = score
                final_reason = "证据不足（已抑制误报）"
            else:
                if l2_threshold_override is not None:
                    try:
                        threshold = int(l2_threshold_override)
                    except Exception:
                        threshold = 80
                else:
                    rule = self.policy_rules.get(category, None)
                    threshold = int(rule.get("risk_threshold", 80)) if isinstance(rule, dict) else 80

                if is_whitelist and evidence_strength >= 2:
                    is_whitelist = False

                if score >= threshold and (not is_whitelist):
                    final_action = "拦截"
                    final_score = score
                    ev = ", ".join([str(x) for x in evidence_spans[:3]]) if evidence_spans else ""
                    if not ev and visual_evidence:
                        ev = visual_evidence[:80]
                    final_reason = f"L2({chosen_src}) 证据命中拦截（{category}）：{ev}" if ev else f"L2({chosen_src}) 证据命中拦截（{category}）"
                else:
                    final_action = "通过"
                    final_score = score
                    final_reason = "有风险证据，但未达拦截阈值"

            final_cat = category

        l1_res["final_decision"] = {
            "risk_score": int(final_score),
            "action": final_action,
            "reason": final_reason[:220],
            "evidence": {
                "l1": {
                    "labels": labels[:30],
                    "labels_cn": self._objects_cn(labels[:30]),
                    "hard_hits": hard_hits[:10] if l1_has_hard_risk else [],
                    "hit_summary": hs,
                    "l1_hard_block": bool(l1_hard_block),
                },
                "l2_text": l2_text_res or {},
                "l2_sex": l2_sex_res or {},
                "chosen_category": final_cat,
                "time_ms": int((time.time() - t0) * 1000),
            },
        }
        return l1_res
