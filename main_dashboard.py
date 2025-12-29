import sys
import streamlit as st
import cv2
import numpy as np
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 路径修复
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from core.l3_policy.risk_engine import RiskEngine

st.set_page_config(page_title="AI 内容风控大屏", layout="wide")
st.title("🛡️ RTX 5070 Ti 实时内容安全监控后台")

# 侧边栏
st.sidebar.header("系统配置")
env_key = os.getenv("DASHSCOPE_API_KEY", "")
api_key = st.sidebar.text_input("DashScope API Key", value=env_key, type="password")

if 'engine' not in st.session_state and api_key:
    try:
        with st.spinner("AI 引擎初始化中..."):
            st.session_state.engine = RiskEngine(api_key=api_key)
            st.sidebar.success("✅ 引擎就绪")
    except Exception as e:
        st.sidebar.error(f"初始化失败: {e}")

col_video, col_log = st.columns([2, 1])

with col_video:
    st.subheader("📷 实时画面")
    video_placeholder = st.empty()

with col_log:
    st.subheader("🚨 风险日志")
    log_area = st.container()

if st.sidebar.button("启动监控"):
    if 'engine' not in st.session_state:
        st.error("请先激活引擎")
    else:
        cap = cv2.VideoCapture(0)
        # 记录上一次检测时间，用于控制频率
        last_check_time = 0
        last_results = {"objects": [], "最终裁决": {}}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            curr_time = time.time()
            if curr_time - last_check_time > 0.5:
                last_results = st.session_state.engine.detect_and_judge(frame)
                last_check_time = curr_time

            # 渲染
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for obj in last_results.get("objects", []):
                b = [int(i) for i in obj["bbox"]]
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

            video_placeholder.image(display_frame, channels="RGB")

            # 日志更新
            decision = last_results.get("最终裁决", {})
            if decision.get("action") in ["warn", "block"]:
                with log_area:
                    t = time.strftime("%H:%M:%S", time.localtime())
                    st.error(f"🔴 [{t}] {decision.get('category_cn')} | 动作: {decision.get('action')}")
                    st.write(f"**理由**: {decision.get('reason')} (Rule: {decision.get('rule_id')})")
                    st.divider()

            time.sleep(0.01)
        cap.release()