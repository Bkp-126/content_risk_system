import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger("L2-Brain")

class ContentReviewer:
    def __init__(self, api_key=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        logger.info("L2 Decision Engine: High-speed audit mode initialized.")

    def review_text_only(self, context_str):
        """
        Industrial-grade semantic audit:
        1. Context Truncation: Prevents token overflow.
        2. Evasive Language Detection: Identifies homophones and onomatopoeia.
        """
        try:
            # 截断输入，防止长文本导致延迟
            sanitized_context = str(context_str)[:500]

            # 工业级 Prompt：强化对隐喻、拟声词（如月炮等变体）的识别
            system_prompt = """
            Role: Content Moderation Expert.
            Task: Audit the provided text for risks (Sex, Gambling, Illegal Ads, or Evasive Language).
            Rules:
            1. Response must be in JSON format.
            2. 'reason' must be concise (under 12 words) and professional.
            3. Strictly identify evasive language, homophones, or onomatopoeia (e.g., phonetic symbols for solicitation).
            4. If the content involves pornographic solicitation or gambling, risk_score must be > 85.
            Output format: {"category": "Ad", "risk_score": 90, "reason": "Solicitation via onomatopoeia detected"}
            """

            response = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Audit Target: {sanitized_context}"}
                ],
                max_tokens=100,
                response_format={"type": "json_object"},
                temperature=0.1  # 降低随机性，使判定更果断
            )

            res_content = response.choices[0].message.content
            return json.loads(res_content)

        except Exception as e:
            logger.error(f"L2 Audit Error: {e}")
            return {"category": "Normal", "risk_score": 0, "reason": "Audit timeout or error"}