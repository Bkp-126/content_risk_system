import sys
import os
import json
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

# 1. 加载环境变量 (确保能读到 .env 中的 API KEY)
load_dotenv()

# 2. 自动化项目路径定位
current_file = Path(__file__).resolve()
# 向上跳三级到达 content_risk_system 根目录
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 3. 模块导入
try:
    from core.l1_shield.frame_processor import FrameProcessor
    from core.l2_brain.llm_client import ContentReviewer
except ImportError as e:
    print(f"导入核心模块失败，请检查项目结构: {e}")

# 4. 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("L3-RiskEngine")

# 物体中文对照表
LABEL_MAP = {
    "person": "人物",
    "cell phone": "手机",
    "knife": "刀具",
    "scissors": "剪刀",
    "gun": "枪支"
}


class RiskEngine:
    def __init__(self, api_key=None):
        logger.info("正在初始化风控策略引擎...")

        # 初始化 L1 视觉过滤层
        self.l1 = FrameProcessor()

        # 初始化 L2 语义决策层 (api_key 会优先从环境变量读取)
        self.l2 = ContentReviewer(api_key=api_key)

        # 加载 rules.yaml 策略配置
        self.policy_rules = self._load_policy()
        logger.info(f"成功加载策略，当前共有 {len(self.policy_rules)} 条活跃规则。")

    def _load_policy(self):
        """加载外部 YAML 配置文件"""
        config_path = project_root / "configs" / "rules.yaml"
        if not config_path.exists():
            logger.warning(f"⚠️ 配置文件不存在: {config_path}，使用系统默认策略。")
            return {"Normal": {"risk_threshold": 60, "action": "pass", "label_cn": "合规"}}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # 将列表转换为字典，方便通过 category 快速查询
            return {r['category']: r for r in config.get('rules', [])}
        except Exception as e:
            logger.error(f"解析配置文件出错: {e}")
            return {"Normal": {"risk_threshold": 60, "action": "pass", "label_cn": "合规"}}

    def detect_and_judge(self, frame):
        """执行全链路检测与判决"""
        # A. 执行 L1 视觉检测
        l1_res = self.l1.process_frame(frame)

        # B. 汉化处理
        raw_objs = [obj['label'] for obj in l1_res.get('objects', [])]
        objs_cn = [LABEL_MAP.get(o, o) for o in raw_objs]
        texts = l1_res.get('texts', [])

        # C. 逻辑判定：是否触发 L2 深度审计
        if l1_res.get('risk_flag') or texts:
            logger.info("L1 触发预警，请求 L2 语义裁决...")
            context = f"物体: {objs_cn} | 文字: {' / '.join(texts)}"

            # 调用云端大脑 (已配置为输出中文)
            decision = self.l2.review_text_only(context)

            # D. 基于 YAML 策略进行动态判决
            category = decision.get("category", "Normal")
            score = decision.get("risk_score", 0)

            # 匹配规则，匹配不到则用 Normal 兜底
            rule = self.policy_rules.get(category, self.policy_rules.get("Normal"))

            # 动态执行 Action
            final_action = "pass"
            if score >= rule.get('risk_threshold', 60):
                final_action = rule.get('action', 'warn')

            l1_res["最终裁决"] = {
                "risk_score": score,
                "category_cn": rule.get('label_cn', '未知'),
                "action": final_action,
                "reason": decision.get("reason", "无详细理由"),
                "rule_id": rule.get('rule_id', "Default")
            }
            l1_res["审计级别"] = "深度语义审计"
        else:
            l1_res["最终裁决"] = {
                "risk_score": 0,
                "category_cn": "合规",
                "action": "pass",
                "reason": "安全"
            }
            l1_res["审计级别"] = "基础过滤"

        return l1_res