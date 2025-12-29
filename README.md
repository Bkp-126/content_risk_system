# 🛡️ 多模态实时内容风控与策略决策系统 (Content-Risk-System)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA-RTX%205070%20Ti-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
</p>

> **面向短视频/直播高并发场景的端云协同风控方案** > 运行环境：**Windows 11(开发验证) / Ubuntu 22.04(推荐生产环境）** | **Python 3.10** | **RTX 5070 Ti**

本项目针对内容审核场景中“全量大模型审计成本极高、纯本地规则误报严重”的痛点，设计并实现了 **Funnel (漏斗) 分层架构**。系统利用边缘端 GPU 资源拦截 95% 以上的安全流量，仅针对疑难帧调用云端 VLM 进行深度语义决策，实现性能、成本与精度的最优平衡。

---

## 📖 目录
- [🏗️ 系统架构](#️-系统架构)
- [✨ 核心功能](#-核心功能)
- [🛠️ 技术深度: 工程难点与挑战](#️-技术深度-工程难点与挑战)
- [📈 性能表现](#-性能表现)
- [🚀 快速部署](#-快速部署)
- [📜 策略配置示例](#-策略配置示例)

---

## 🏗️ 系统架构

系统遵循 **感知(L1) -> 决策(L2) -> 执行(L3)** 的设计哲学：

[Image of a multi-layer hierarchical software architecture diagram]

```mermaid
graph TD
    subgraph "L1 边缘感知层 (Local GPU)"
        A[视频流捕获] --> B[YOLOv11 违规物体检测]
        B --> C[EasyOCR 敏感文本抓取]
    end

    subgraph "L2 云端裁决层 (VLM Cloud)"
        B -- 命中疑似目标 -- D{Qwen-VL-Plus 深度语义推理}
        C -- 提取敏感变体 -- D
    end

    subgraph "L3 策略执行引擎 (Policy Engine)"
        D --> E{Policy Matching}
        E -- 读取 rules.yaml -- F[最终裁决 Action]
    end

    F --> G[结果输出: 封禁/警告/放行]