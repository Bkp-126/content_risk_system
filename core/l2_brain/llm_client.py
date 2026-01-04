import os
import re
import json
import time
import base64
import hashlib
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger("L2-Brain")


class ContentReviewer:
    SAFE_PHRASES = [
        "欢迎来到直播间", "点点关注", "点个关注", "谢谢大哥", "谢谢老板",
        "宝宝", "家人们", "下单", "拍下", "福利", "上链接", "直播间", "点赞", "关注"
    ]

    RE_WECHAT_HINT = re.compile(
        r"(vx|v信|威信|微信|微x|微X|w(e)?chat|weixin|加v|加V|加微|加薇|加威|w:|W:)",
        re.IGNORECASE
    )
    RE_QQ = re.compile(r"(?:qq|Qq|QQ)\s*[:：]?\s*\d{5,12}")
    RE_PHONE_STRICT = re.compile(r"(?:\b|[^0-9])1[3-9]\d{9}(?:\b|[^0-9])")
    RE_LONG_DIGITS_LOOSE = re.compile(r"\d(?:[\s\.\-\_]*\d){7,}")
    RE_MONEY_GAMBLE = re.compile(r"(下注|返利|中奖|博彩|百家乐|彩票|代充|充值|刷单|返现|收益)")

    SEX_TERMS = ["月泡", "外围", "包夜", "上门", "裸", "没穿", "露", "约", "xxoo", "换衣服", "坐气球"]
    AD_TERMS = ["私聊", "带走", "进群", "暗号", "口令", "加威", "加微", "加v", "vx", "微信", "w:"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus",
        vl_enabled: bool = True,
        vl_cooldown_sec: float = 2.0,
        vl_max_calls_per_min: int = 18,
        vl_cache_ttl_sec: float = 30.0,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.model = str(model)
        self.vl_enabled = bool(vl_enabled) and bool(self.api_key)
        self.vl_cooldown_sec = float(vl_cooldown_sec)
        self.vl_max_calls_per_min = int(vl_max_calls_per_min)
        self.vl_cache_ttl_sec = float(vl_cache_ttl_sec)

        self.client = None
        if self.vl_enabled:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            logger.info("L2：VL 已禁用（缺少 API Key 或手动关闭）")

        self._vl_cache: Dict[str, Any] = {}       # key -> (ts, res)
        self._vl_call_times: List[float] = []     # timestamps in last 60s

        logger.info("L2 初始化完成")

    @staticmethod
    def _normalize_text(s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        trans_map = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "丨": "1"})
        return s.translate(trans_map)

    def _is_phrase_whitelist(self, full_text_raw: str) -> bool:
        if not full_text_raw:
            return False
        return any(p in full_text_raw for p in self.SAFE_PHRASES)

    def _extract_rule(self, texts: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_lines, norm_lines = [], []
        for it in texts or []:
            raw = str(it.get("text", "")).strip()
            if not raw:
                continue
            raw_lines.append(raw)
            norm_lines.append(self._normalize_text(raw))

        full_raw = " ".join(raw_lines)[:800]
        full_norm = " ".join(norm_lines)[:800]

        evidence: List[str] = []

        mg = self.RE_MONEY_GAMBLE.search(full_raw) or self.RE_MONEY_GAMBLE.search(full_norm)
        if mg:
            evidence.append(mg.group(0))

        mw = self.RE_WECHAT_HINT.search(full_norm)
        if mw:
            evidence.append(mw.group(0))

        mq = self.RE_QQ.search(full_raw) or self.RE_QQ.search(full_norm)
        if mq:
            evidence.append(mq.group(0))

        mp = self.RE_PHONE_STRICT.search(full_norm)
        if mp:
            evidence.append(mp.group(0).strip())

        ml = self.RE_LONG_DIGITS_LOOSE.search(full_norm)
        if ml:
            evidence.append(ml.group(0))

        for t in self.SEX_TERMS + self.AD_TERMS:
            if t and (t in full_raw):
                evidence.append(t)

        dedup, seen = [], set()
        for e in evidence:
            e = str(e).strip()
            if e and e not in seen:
                dedup.append(e)
                seen.add(e)

        category, strength, score = "Normal", 0, 0

        has_wechat = self.RE_WECHAT_HINT.search(full_norm) is not None
        has_phone = self.RE_PHONE_STRICT.search(full_norm) is not None
        has_qq = mq is not None
        has_long_digits = ml is not None
        has_gamble = mg is not None
        has_sex = any(t in full_raw for t in self.SEX_TERMS)

        if has_gamble:
            category, strength, score = "Gambling", 3, 90
        elif has_sex:
            category, strength, score = "Sex", 2, 80
        elif has_phone or has_qq or (has_wechat and has_long_digits):
            category, strength, score = "Ad", 3, 92
        elif has_wechat or has_long_digits:
            category, strength, score = "Ad", 2, 70

        return {
            "category": category,
            "risk_score": int(score),
            "evidence_strength": int(strength),
            "evidence_spans": dedup[:10],
            "full_text_raw": full_raw,
            "full_text_norm": full_norm,
        }

    @staticmethod
    def _file_md5(path: str) -> Optional[str]:
        try:
            if not path or (not os.path.exists(path)):
                return None
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _encode_image_to_data_url(image_path: str) -> Optional[str]:
        try:
            if not image_path or not os.path.exists(image_path):
                return None
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    def _vl_rate_limit_ok(self) -> bool:
        now = time.time()
        self._vl_call_times = [t for t in self._vl_call_times if now - t < 60]
        return len(self._vl_call_times) < self.vl_max_calls_per_min

    def _vl_cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        item = self._vl_cache.get(cache_key)
        if not item:
            return None
        ts, res = item
        if now - ts > self.vl_cache_ttl_sec:
            self._vl_cache.pop(cache_key, None)
            return None
        if now - ts < self.vl_cooldown_sec:
            return res
        return None

    def _vl_cache_set(self, cache_key: str, res: Dict[str, Any]) -> None:
        self._vl_cache[cache_key] = (time.time(), res)

    def _vl_cached_call(self, cache_key: str, build_messages_fn, max_tokens: int = 220) -> Dict[str, Any]:
        cached = self._vl_cache_get(cache_key)
        if cached is not None:
            logger.info("L2 VL 命中缓存/冷却，跳过远程调用")
            return cached

        if (not self.vl_enabled) or (self.client is None) or (not self._vl_rate_limit_ok()):
            out = {
                "category": "Normal",
                "risk_score": 0,
                "evidence_strength": 0,
                "evidence_spans": [],
                "visual_evidence": "",
                "is_whitelist": True,
                "reason": "VL禁用或限频",
            }
            self._vl_cache_set(cache_key, out)
            return out

        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages_fn(),
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        t1 = time.time()
        self._vl_call_times.append(time.time())

        content = (resp.choices[0].message.content or "").strip()
        try:
            res = json.loads(content)
        except Exception:
            res = {}

        out = {
            "category": str(res.get("category", "Normal") or "Normal"),
            "risk_score": int(res.get("risk_score", 0) or 0),
            "evidence_strength": int(res.get("evidence_strength", 0) or 0),
            "evidence_spans": [],
            "visual_evidence": str(res.get("visual_evidence", "") or "")[:220],
            "is_whitelist": bool(res.get("is_whitelist", False)),
            "reason": (str(res.get("reason", "") or "")[:180] + f" | vl_ms={int((t1 - t0) * 1000)}"),
        }

        self._vl_cache_set(cache_key, out)
        return out

    def review_visual_sexual(self, image_path: str) -> Dict[str, Any]:
        md5 = self._file_md5(image_path)
        if not md5:
            return {
                "category": "Normal", "risk_score": 0, "evidence_strength": 0,
                "evidence_spans": [], "visual_evidence": "", "is_whitelist": True, "reason": "no_image_md5"
            }

        data_url = self._encode_image_to_data_url(image_path)
        if not data_url:
            return {
                "category": "Normal", "risk_score": 0, "evidence_strength": 0,
                "evidence_spans": [], "visual_evidence": "", "is_whitelist": True, "reason": "image_read_fail"
            }

        cache_key = f"{md5}:vl_sex"

        def build_msgs():
            system_prompt = (
                "Role: Live-stream Safety Moderator.\n"
                "Task: Judge if image contains sexual/suggestive content (borderline / nudity / erotic pose / sexual hint).\n"
                "Output JSON only.\n"
                "JSON:\n"
                "{\n"
                '  "category": "Sex"|"Normal",\n'
                '  "risk_score": 0-100,\n'
                '  "evidence_strength": 0-3,\n'
                '  "visual_evidence": "short evidence",\n'
                '  "is_whitelist": false,\n'
                '  "reason": "short reason"\n'
                "}\n"
            )
            user_content = [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "请判断是否存在擦边/性暗示/裸露/挑逗姿态等风险。若有，给出 visual_evidence（简短描述）。"},
            ]
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

        return self._vl_cached_call(cache_key, build_msgs, max_tokens=200)

    def review_from_l1(self, l1_payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            texts = l1_payload.get("texts", []) or []
            frame_path = l1_payload.get("frame_path", None)

            rule = self._extract_rule(texts)
            phrase_wl = self._is_phrase_whitelist(rule.get("full_text_raw", ""))

            if rule["risk_score"] >= 90:
                return {
                    "category": rule["category"],
                    "risk_score": rule["risk_score"],
                    "evidence_strength": rule["evidence_strength"],
                    "evidence_spans": rule["evidence_spans"],
                    "visual_evidence": "",
                    "is_whitelist": bool(phrase_wl),
                    "reason": f"结构化强证据：{', '.join(rule['evidence_spans'][:3])}" if rule["evidence_spans"] else "结构化强证据",
                }

            enable_vl = self.vl_enabled and (rule["risk_score"] >= 60)
            ocr_too_weak = (len(rule["evidence_spans"]) == 0 and rule["evidence_strength"] == 0)

            md5 = self._file_md5(frame_path) if frame_path else None
            cache_key = f"{md5}:vl_text:{rule['category']}" if md5 else None

            if enable_vl and ocr_too_weak and frame_path and cache_key:
                data_url = self._encode_image_to_data_url(frame_path)
                if not data_url:
                    return {
                        "category": rule["category"],
                        "risk_score": rule["risk_score"],
                        "evidence_strength": rule["evidence_strength"],
                        "evidence_spans": rule["evidence_spans"],
                        "visual_evidence": "",
                        "is_whitelist": bool(phrase_wl),
                        "reason": "图片读取失败（未触发VL）",
                    }

                def build_msgs():
                    system_prompt = (
                        "Role: Live-stream Content Moderation Expert.\n"
                        "Task: Transcribe any handwritten/overlay text that indicates:\n"
                        "- contact guiding (w:/vx/加V/微信/QQ/手机号/群/私聊)\n"
                        "- gambling/sex keywords\n"
                        "Output JSON only.\n"
                        "JSON:\n"
                        "{\n"
                        '  "category": "Sex"|"Ad"|"Gambling"|"Normal",\n'
                        '  "risk_score": 0-100,\n'
                        '  "evidence_strength": 0-3,\n'
                        '  "visual_evidence": "key transcribed text",\n'
                        '  "is_whitelist": true/false,\n'
                        '  "reason": "short reason"\n'
                        "}\n"
                    )
                    user_content = [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "请抄写画面中关键文字，重点关注引流联系方式和赌博/色情。"},
                    ]
                    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

                vl = self._vl_cached_call(cache_key, build_msgs, max_tokens=220)
                vl["is_whitelist"] = bool(vl.get("is_whitelist", False)) or bool(phrase_wl)
                return vl

            return {
                "category": rule["category"],
                "risk_score": rule["risk_score"],
                "evidence_strength": rule["evidence_strength"],
                "evidence_spans": rule["evidence_spans"],
                "visual_evidence": "",
                "is_whitelist": bool(phrase_wl),
                "reason": "规则审计（未触发VL）" if rule["risk_score"] > 0 else "证据不足/常规话术",
            }

        except Exception as e:
            logger.error(f"L2 Error: {e}")
            return {
                "category": "Normal",
                "risk_score": 0,
                "evidence_strength": 0,
                "evidence_spans": [],
                "visual_evidence": "",
                "is_whitelist": True,
                "reason": "error",
            }
