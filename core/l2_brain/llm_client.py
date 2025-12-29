import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("L2-Brain")


class ContentReviewer:
    """
    L2 语义决策引擎：利用大语言模型对视觉特征进行深度合规性判定
    """

    def __init__(self, api_key=None):
        # 优先级：手动传入 > 环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not self.api_key:
            logger.error("❌ 未找到 DASHSCOPE_API_KEY，请检查 .env 文件")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.model_name = "qwen-vl-plus"
            logger.info(f"L2 客户端初始化成功。")

    def review_text_only(self, context_string):
        if not self.client:
            return {"risk_score": 50, "category": "Error", "reason": "API Key 缺失"}

        system_prompt = "你是一名资深内容安全审计员。请根据提供的元数据进行风险评估，并严格以 JSON 格式输出。"
        user_prompt = f"""
        请分析以下从监控画面中提取出的特征信息：
        "{context_string}"

        请严格按照以下 JSON 格式输出中文判决结果：
        {{
            "risk_score": (整数 0-100),
            "category": (字符串：例如 "Normal", "Ad", "Porn", "Violence"),
            "reason": (字符串：请用简洁的中文说明判决理由)
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"推理出错: {e}")
            return {"risk_score": 50, "category": "Error", "reason": str(e)}