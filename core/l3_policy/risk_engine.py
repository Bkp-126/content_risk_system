import sys
import os
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.l1_shield.frame_processor import FrameProcessor
from core.l2_brain.llm_client import ContentReviewer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("L3-RiskEngine")

LABEL_MAP = {"person": "人物", "cell phone": "手机", "knife": "刀具", "scissors": "剪刀", "gun": "枪支"}


class RiskEngine:
    def __init__(self, api_key=None):
        logger.info("初始化风控引擎...")
        self.l1 = FrameProcessor()
        self.l2 = ContentReviewer(api_key=api_key)
        self.policy_rules = self._load_policy()

    def _load_policy(self):
        config_path = project_root / "configs" / "rules.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return {r['category']: r for r in config.get('rules', [])}
        except:
            return {"Normal": {"risk_threshold": 60, "action": "pass", "label_cn": "合规"}}

    def detect_and_judge(self, frame):
        l1_res = self.l1.process_frame(frame)

        raw_objs = [obj['label'] for obj in l1_res.get('objects', [])]
        objs_cn = [LABEL_MAP.get(o, o) for o in raw_objs]
        texts = l1_res.get('texts', [])

        # --- 性能优化：精准触发逻辑 ---
        # 1. 发现明确危险品 (risk_flag)
        # 2. 发现文字 (疑似引流)
        # 3. 发现关键敏感物体 (如刀具、枪支等)
        is_sensitive_obj = any(o in ['knife', 'scissors', 'gun', 'baseball bat'] for o in raw_objs)

        if l1_res.get('risk_flag') or texts or is_sensitive_obj:
            logger.info("触发深度审计...")
            context = f"物体: {objs_cn} | 文字: {' / '.join(texts)}"
            decision = self.l2.review_text_only(context)

            category = decision.get("category", "Normal")
            score = decision.get("risk_score", 0)
            rule = self.policy_rules.get(category, self.policy_rules.get("Normal"))

            final_action = "通过"
            if score >= rule.get('risk_threshold', 30):
                action_map = {"block": "拦截", "warn": "警告", "拦截": "拦截", "警告": "警告"}
                final_action = action_map.get(rule.get('action'), "拦截")

            l1_res["最终裁决"] = {
                "risk_score": score,
                "action": final_action,
                "reason": decision.get("reason", "无")
            }
        else:
            l1_res["最终裁决"] = {"risk_score": 0, "action": "通过", "reason": "L1安全过滤"}

        return l1_res