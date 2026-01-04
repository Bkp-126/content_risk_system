import cv2
import logging
from ultralytics import YOLO
import easyocr

logger = logging.getLogger("L1-Processor")


class FrameProcessor:
    def __init__(self):
        # 锁定使用性价比最高的 YOLO11s
        self.model = YOLO('yolo11s.pt')

        # 初始化 EasyOCR (简体中文/英文)
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        logger.info("风控视觉引擎初始化完成 (当前基准模型: YOLO11s)")

    def process_frame(self, frame_path):
        """处理单张图片"""
        img = cv2.imread(frame_path)
        if img is None:
            return {"risk_flag": False, "objects": [], "texts": []}

        # --- 核心策略：极低阈值策略提升召回 ---
        # conf=0.15 保证哪怕是模糊的风险物体也能被捕获并送审
        results = self.model(img, conf=0.15, verbose=False)[0]

        detected_objs = []
        risk_flag = False
        for box in results.boxes:
            label = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detected_objs.append({"label": label, "conf": conf})

            # 基础风险监控池
            if label in ['knife', 'scissors', 'gun', 'baseball bat', 'fire']:
                risk_flag = True

        try:
            # 极速灰度化处理，规避 EasyOCR 维度报错并加速
            gray_for_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ocr_res = self.ocr_reader.readtext(gray_for_ocr, detail=0)
        except Exception as e:
            logger.error(f"OCR 处理流程异常: {e}")
            ocr_res = []

        return {
            "risk_flag": risk_flag,
            "objects": detected_objs,
            "texts": ocr_res
        }