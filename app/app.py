import os
import cv2
import uuid
import time
import tempfile
import logging

import gradio as gr

from core.l3_policy.risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RiskControlApp")

custom_css = """
body { background-color: #f4f7f9; }
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
#header {
    text-align: center;
    padding: 20px;
    background: #2563eb;
    color: white;
    border-radius: 12px;
    margin-bottom: 18px;
}
#audit-report {
    background: #ffffff;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 6px 10px rgba(0,0,0,0.08);
    border-top: 6px solid #2563eb;
}
.status-bar { height: 10px; border-radius: 5px; background: #e5e7eb; margin: 10px 0; overflow: hidden; }
.status-fill { height: 100%; transition: width 0.2s ease-in-out; }
.small { font-size: 12px; color: #64748b; }
.kv { margin: 4px 0; }
"""


_engine = None


def get_engine() -> RiskEngine:
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine


def _safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _get_final_decision(result: dict):
    if not isinstance(result, dict):
        return {}
    if isinstance(result.get("final_decision"), dict):
        return result["final_decision"]
    if isinstance(result.get("最终裁决"), dict):
        return result["最终裁决"]
    return {}


def _format_objects(objects_preview):
    if isinstance(objects_preview, list) and objects_preview and isinstance(objects_preview[0], dict):
        show = []
        for o in objects_preview[:10]:
            lab = o.get("label", "")
            conf = o.get("conf", None)
            if conf is None:
                show.append(str(lab))
            else:
                try:
                    show.append(f"{lab}({float(conf):.2f})")
                except Exception:
                    show.append(str(lab))
        return ", ".join(show) if show else "无"
    return str(objects_preview) if objects_preview else "无"


def _format_texts(texts_preview):
    if isinstance(texts_preview, list) and texts_preview and isinstance(texts_preview[0], dict):
        show = []
        for t in texts_preview[:10]:
            txt = str(t.get("text", "")).strip()
            if not txt:
                continue
            conf = t.get("conf", None)
            if conf is None:
                show.append(txt)
            else:
                try:
                    show.append(f"{txt}(conf={float(conf):.2f})")
                except Exception:
                    show.append(txt)
        return " / ".join(show) if show else "无"
    if isinstance(texts_preview, list) and texts_preview:
        return " / ".join([str(x) for x in texts_preview[:10]])
    return "无"


def _extract_evidence(decision: dict):
    if not isinstance(decision, dict):
        return {}, {}, {}, {}
    evidence = decision.get("evidence", {})
    if not isinstance(evidence, dict):
        return {}, {}, {}, {}

    l1_ev = evidence.get("l1", {}) or {}
    l2_text = evidence.get("l2_text", {}) or {}
    l2_sex = evidence.get("l2_sex", {}) or {}
    meta = {"chosen_category": evidence.get("chosen_category", ""), "time_ms": evidence.get("time_ms", None)}
    return l1_ev, l2_text, l2_sex, meta


def process_image(input_img, l2_threshold, l1_veto, skin_trigger):
    if input_img is None:
        return None, "请先上传需要审计的图片。"

    engine = get_engine()

    try:
        engine.l1.skin_ratio_trigger = float(skin_trigger)
    except Exception:
        pass

    frame = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    tmp_dir = tempfile.gettempdir()
    temp_path = os.path.join(tmp_dir, f"risk_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg")

    t0 = time.time()
    try:
        ok = cv2.imwrite(temp_path, frame)
        if not ok:
            return input_img, "图片写入失败，请重试。"

        try:
            result = engine.detect_and_judge(
                temp_path,
                l2_threshold_override=int(l2_threshold),
                l1_hard_block=bool(l1_veto),
            )
        except TypeError:
            result = engine.detect_and_judge(temp_path)

        decision = _get_final_decision(result)

        action = decision.get("action", "通过")
        score = int(decision.get("risk_score", 0) or 0)
        reason = str(decision.get("reason", "未发现违规迹象。"))

        is_blocked = action in ["拦截", "Block", "【拦截】"]
        color = "#dc2626" if is_blocked else "#16a34a"
        display_category = "高危风险内容" if is_blocked else "合规正常内容"

        objects_preview_str = _format_objects(result.get("objects", []))
        texts_preview_str = _format_texts(result.get("texts", []))

        l1_ev, l2_text, l2_sex, meta = _extract_evidence(decision)

        def spans_str(d):
            if not isinstance(d, dict):
                return "无"
            spans = d.get("evidence_spans", []) or []
            if isinstance(spans, list) and spans:
                return ", ".join([str(x) for x in spans[:6]])
            ve = str(d.get("visual_evidence", "") or "").strip()
            return ve[:120] if ve else "无"

        l2_text_spans = spans_str(l2_text)
        l2_sex_spans = spans_str(l2_sex)

        hs = result.get("hit_summary", {}) if isinstance(result, dict) else {}
        l1_times = (hs.get("time_ms", {}) if isinstance(hs, dict) else {}) or {}
        total_ms = int((time.time() - t0) * 1000)

        skin_ratio = None
        if isinstance(hs, dict):
            try:
                skin_ratio = float(hs.get("skin_ratio", 0.0))
            except Exception:
                skin_ratio = None

        report = f"""
<div id="audit-report">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:22px;color:{color};"><b>最终裁决: {action}</b></span>
    <span style="background:{color}22;color:{color};padding:4px 12px;border-radius:16px;font-weight:bold;">
      {display_category}
    </span>
  </div>

  <div style="margin-top:12px;">
    <p style="margin-bottom:6px;"><b>风险评分: {score}/100</b></p>
    <div class="status-bar">
      <div class="status-fill" style="width:{score}%;background:{color};"></div>
    </div>
  </div>

  <p style="margin-top:12px;font-size:15px;line-height:1.6;">
    <b>判定结论:</b> {reason}
  </p>

  <div style="margin-top:14px;padding:10px;background:#f8fafc;border-radius:8px;font-size:13px;color:#475569;">
    <p style="margin:0;"><b>L1 元数据</b></p>
    <p class="kv">物体识别: {objects_preview_str}</p>
    <p class="kv">OCR提取: {texts_preview_str}</p>
    <p class="kv">skin_ratio: {f"{skin_ratio:.3f}" if isinstance(skin_ratio, float) else "N/A"} | trigger阈值: {float(skin_trigger):.2f}</p>

    <p style="margin:10px 0 0 0;"><b>L2 证据（文本/引流）</b></p>
    <p class="kv">{l2_text_spans}</p>

    <p style="margin:10px 0 0 0;"><b>L2 证据（擦边/性暗示）</b></p>
    <p class="kv">{l2_sex_spans}</p>
  </div>

  <div class="small" style="margin-top:10px;">
    L2阈值: {int(l2_threshold)} | L1一票否决: {"开" if bool(l1_veto) else "关"} | 总耗时: {total_ms}ms
    <br/>
    L1耗时(ms): qr={l1_times.get("qr","-")}, skin={l1_times.get("skin","-")}, yolo={l1_times.get("yolo","-")},
    ocr={l1_times.get("ocr","-")}, total={l1_times.get("total","-")}
  </div>
</div>
"""
        return input_img, report

    finally:
        _safe_remove(temp_path)


with gr.Blocks(title="多模态风控系统", css=custom_css) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🛡️ 多模态内容风险管理系统")
        gr.Markdown("YOLOv11 + Qwen-VL（低延迟证据融合版）")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📥 素材提交")
            input_image = gr.Image(label=None, type="numpy", interactive=True)
            run_button = gr.Button("提交审计", variant="primary")

            with gr.Accordion("⚙️ 配置", open=False):
                l2_threshold = gr.Slider(0, 100, value=80, step=1, label="L2 拦截阈值（覆盖 rules.yaml）")
                l1_veto = gr.Checkbox(label="开启 L1 一票否决（强证据必拦）", value=True)
                skin_trigger = gr.Slider(0.05, 0.40, value=0.18, step=0.01, label="擦边门控：skin_ratio_trigger")

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 👁️ 输入预览")
                    output_image = gr.Image(label=None)
                with gr.Column():
                    gr.Markdown("### 📄 审计报告")
                    output_text = gr.HTML()

    run_button.click(
        fn=process_image,
        inputs=[input_image, l2_threshold, l1_veto, skin_trigger],
        outputs=[output_image, output_text],
    )

try:
    demo.queue(concurrency_count=1, max_size=30)
except TypeError:
    try:
        demo.queue(concurrency_limit=1, max_size=30)
    except TypeError:
        demo.queue(max_size=30)

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_port=7860, show_error=True)
