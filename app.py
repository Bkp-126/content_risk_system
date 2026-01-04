import gradio as gr
import cv2
import os
import logging
from core.l3_policy.risk_engine import RiskEngine

# 规范化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RiskControlApp")

engine = RiskEngine()

# 自定义 CSS 样式
custom_css = """
#header { text-align: center; padding: 20px; background: #f0f2f5; border-radius: 10px; margin-bottom: 20px; }
#audit-report { background: #ffffff; border-left: 5px solid #2d5cf7; padding: 15px; border-radius: 5px; }
"""


def process_image(input_img):
    if input_img is None:
        return None, "Error: Please upload an image first."

    frame = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    temp_path = "temp_web_upload.jpg"
    cv2.imwrite(temp_path, frame)

    result = engine.detect_and_judge(temp_path)
    decision = result.get("最终裁决", {})

    action = decision.get("action", "Pass")
    score = int(decision.get("risk_score", 0))
    reason = decision.get("reason", "Normal content.")

    color = "red" if action in ["拦截", "Block", "【拦截】"] else "green"
    display_category = "Violative Content" if action in ["拦截", "Block", "【拦截】"] else "Normal Content"

    report = f"""
<div id="audit-report">
    <p style="font-size: 18px; color: {color};"><b>Action: {action}</b></p>
    <p><b>Category:</b> {display_category}</p>
    <p><b>Risk Score:</b> <span style="font-size: 20px; color: {color};">{score}</span></p>
    <p><b>Rationale:</b> {reason}</p>
    <hr>
    <p style="font-size: 12px; color: #666;"><b>Metadata (L1):</b><br>
    Objects: {[obj['label'] for obj in result.get('objects', [])]}<br>
    Text: {str(result.get('texts', []))[:100]}...</p>
</div>
"""
    return input_img, report


# 修正 Blocks 构造函数及 Div 报错
with gr.Blocks(title="Content Risk Management System") as demo:
    # 修正：将 gr.Div 替换为 gr.Column 并设置 elem_id 模拟标题区
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🛡️ Multi-modal Content Risk Management System")
        gr.Markdown("Enterprise-grade live stream moderation solution based on YOLOv11 & Qwen-VL")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📥 Input Source")
                input_image = gr.Image(label=None, type="numpy", interactive=True)
                run_button = gr.Button("🚀 Start Multi-Layer Audit", variant="primary")

            with gr.Accordion("⚙️ System Configurations", open=False):
                gr.Slider(0, 100, value=55, label="Ad Threshold")
                gr.Slider(0, 100, value=35, label="Violence Threshold")

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 👁️ Visual Analysis")
                    output_image = gr.Image(label=None)
                with gr.Column():
                    gr.Markdown("### 📄 Audit Summary")
                    output_text = gr.HTML()

    run_button.click(fn=process_image, inputs=input_image, outputs=[output_image, output_text])

if __name__ == "__main__":
    # 修正：根据 Gradio 6.0 规范，将 css 参数移动到 launch 中
    demo.launch(inbrowser=True, css=custom_css)