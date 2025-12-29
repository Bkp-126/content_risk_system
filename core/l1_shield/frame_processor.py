import cv2
import logging
from ultralytics import YOLO
import easyocr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FrameProcessor")

class FrameProcessor:
    """
    L1 视觉过滤层：负责实时物体检测 (YOLO) 与 文字识别 (OCR)
    """
    def __init__(self):
        logger.info("正在初始化视觉引擎...")
        try:
            self.yolo_model = YOLO("yolo11n.pt")
            self.yolo_model.to('cuda:0')
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
            logger.info("YOLO 与 EasyOCR 初始化完成 (GPU 加速已启动)")
        except Exception as e:
            logger.error(f"引擎加载失败: {e}")
            raise e

    def process_frame(self, frame):
        if frame is None:
            return {}

        results = {"objects": [], "texts": [], "risk_flag": False}

        # 1. 目标检测
        yolo_res = self.yolo_model(frame, conf=0.25, verbose=False)
        for r in yolo_res:
            for box in r.boxes:
                label = self.yolo_model.names[int(box.cls[0])]
                results["objects"].append({
                    "label": label,
                    "bbox": box.xyxy[0].tolist()
                })
                if label in ['knife', 'gun']: results["risk_flag"] = True

        # 2. 文字识别
        ocr_res = self.ocr_reader.readtext(frame, detail=0)
        if ocr_res:
            valid_texts = [t for t in ocr_res if len(t) > 1]
            results["texts"] = valid_texts
            keywords = ["加V", "微信", "WeChat", "vx", "福利"]
            if any(k.lower() in " ".join(valid_texts).lower() for k in keywords):
                results["risk_flag"] = True

        return results