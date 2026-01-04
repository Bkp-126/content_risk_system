import cv2
import time
import logging
from pathlib import Path

from ultralytics import YOLO
import easyocr

logger = logging.getLogger("L1-Processor")


class FrameProcessor:
    """
    L1 Fast Path:
    - YOLO: risky objects / cues
    - QR fast check: OpenCV QRCodeDetector
    - OCR: ROI first, optional full-frame fallback
    - skin_ratio: lightweight gate for borderline visual cases
    """

    def __init__(
        self,
        yolo_conf: float = 0.15,
        yolo_iou: float = 0.45,
        max_det: int = 80,
        ocr_gpu: bool = True,
        ocr_fullframe_fallback: bool = True,
        ocr_min_conf_roi: float = 0.25,
        ocr_min_conf_full: float = 0.12,
        roi_pad: int = 8,
        qr_fastcheck: bool = True,
        skin_check: bool = True,
        skin_ratio_trigger: float = 0.18,
    ):
        self.yolo_conf = float(yolo_conf)
        self.yolo_iou = float(yolo_iou)
        self.max_det = int(max_det)

        self.ocr_fullframe_fallback = bool(ocr_fullframe_fallback)
        self.ocr_min_conf_roi = float(ocr_min_conf_roi)
        self.ocr_min_conf_full = float(ocr_min_conf_full)
        self.roi_pad = int(roi_pad)

        self.qr_fastcheck = bool(qr_fastcheck)
        self.skin_check = bool(skin_check)
        self.skin_ratio_trigger = float(skin_ratio_trigger)

        project_root = Path(__file__).resolve().parents[2]
        models_dir = project_root / "models"
        best_path = models_dir / "best.pt"
        yolo_fallback = models_dir / "yolo11s.pt"

        self.model = None
        if best_path.exists() and best_path.stat().st_size > 1024 * 1024:
            try:
                self.model = YOLO(str(best_path))
                logger.info(f"已加载自定义微调模型: {best_path}")
            except Exception as e:
                logger.warning(f"best.pt 加载失败，将回退官方模型: err={e}")
        else:
            logger.warning(f"best.pt 不存在或文件过小(疑似损坏): {best_path}")

        if self.model is None:
            try:
                self.model = YOLO(str(yolo_fallback)) if yolo_fallback.exists() else YOLO("yolo11s.pt")
                logger.info("已加载官方模型作为回退")
            except Exception as e:
                raise RuntimeError(f"无法加载任何 YOLO 权重: {e}")

        try:
            self.ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=bool(ocr_gpu))
            logger.info(f"EasyOCR 初始化完成 | gpu={bool(ocr_gpu)}")
        except Exception as e:
            logger.warning(f"EasyOCR GPU 初始化失败，降级 CPU: {e}")
            self.ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)

        self.qr_detector = cv2.QRCodeDetector() if self.qr_fastcheck else None

        logger.info(
            f"L1 初始化完成 | yolo_conf={self.yolo_conf} iou={self.yolo_iou} max_det={self.max_det} "
            f"ocr_roi_min={self.ocr_min_conf_roi} ocr_full_min={self.ocr_min_conf_full} "
            f"qr_fastcheck={self.qr_fastcheck} skin_check={self.skin_check}"
        )

    @staticmethod
    def _clip_box(x1, y1, x2, y2, w, h):
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    def _ocr_one(self, bgr_img, min_conf: float, max_items: int = 10):
        try:
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            res = self.ocr_reader.readtext(th, detail=1)
            out = []
            for bbox, text, conf in res:
                if not text:
                    continue
                conf = float(conf or 0.0)
                if conf < float(min_conf):
                    continue
                t = str(text).strip()
                if not t:
                    continue
                out.append({"text": t, "conf": conf, "bbox": bbox})
                if len(out) >= max_items:
                    break
            return out
        except Exception as e:
            logger.error(f"OCR 异常: {e}")
            return []

    def _qr_fast_detect(self, img_bgr):
        if not self.qr_detector:
            return {"found": False, "data": [], "points": []}

        try:
            h, w = img_bgr.shape[:2]
            scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
            small = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr

            ok, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(small)
            if not ok or not decoded_info:
                return {"found": False, "data": [], "points": []}

            data = [d for d in decoded_info if d]
            pts = points if points is not None else []
            return {"found": len(data) > 0, "data": data[:3], "points": pts}
        except Exception:
            return {"found": False, "data": [], "points": []}

    def _skin_ratio(self, img_bgr):
        try:
            h, w = img_bgr.shape[:2]
            scale = 320.0 / max(h, w) if max(h, w) > 320 else 1.0
            img = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr

            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            _, cr, cb = cv2.split(ycrcb)

            skin_mask = (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)
            return float(skin_mask.mean())
        except Exception:
            return 0.0

    def process_frame(self, frame_path: str):
        t0 = time.time()
        img = cv2.imread(frame_path)
        if img is None:
            return {"frame_path": frame_path, "objects": [], "texts": [], "hit_summary": {}, "risk_flag": False}

        h, w = img.shape[:2]

        t_q0 = time.time()
        qr_res = self._qr_fast_detect(img) if self.qr_fastcheck else {"found": False, "data": [], "points": []}
        t_q1 = time.time()

        t_s0 = time.time()
        skin_ratio = self._skin_ratio(img) if self.skin_check else 0.0
        t_s1 = time.time()

        t_y0 = time.time()
        try:
            result = self.model(
                img,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                verbose=False,
                max_det=self.max_det,
            )[0]
        except Exception as e:
            logger.error(f"YOLO 推理失败: {e}")
            result = None
        t_y1 = time.time()

        objects = []
        risk_flag = False
        yolo_risk_hits = []
        roi_boxes = []

        if result is not None and getattr(result, "boxes", None) is not None:
            for b in result.boxes:
                try:
                    cls_id = int(b.cls[0])
                    label = str(self.model.names.get(cls_id, cls_id)).strip()
                    conf = float(b.conf[0]) if b.conf is not None else None
                    xyxy = b.xyxy[0].tolist() if b.xyxy is not None else None

                    bbox = None
                    if xyxy:
                        x1, y1, x2, y2 = xyxy
                        x1 -= self.roi_pad
                        y1 -= self.roi_pad
                        x2 += self.roi_pad
                        y2 += self.roi_pad
                        x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, w, h)
                        bbox = [x1, y1, x2, y2]
                        roi_boxes.append((x1, y1, x2, y2))

                    objects.append({"label": label, "conf": conf, "bbox": bbox})

                    if label in [
                        "risky_content",
                        "qr_code",
                        "wechat_id",
                        "wechat_qr",
                        "knife",
                        "scissors",
                        "gun",
                        "fire",
                    ]:
                        risk_flag = True
                        yolo_risk_hits.append(label)
                except Exception:
                    continue

        if qr_res.get("found", False):
            risk_flag = True
            objects.append({"label": "qr_code", "conf": 0.99, "bbox": None})
            yolo_risk_hits.append("qr_code")

        t_o0 = time.time()
        texts = []

        if roi_boxes:
            roi_boxes = sorted(roi_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:3]
            for (x1, y1, x2, y2) in roi_boxes:
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                ocr_out = self._ocr_one(roi, min_conf=self.ocr_min_conf_roi, max_items=8)
                for item in ocr_out:
                    item["roi_xyxy"] = [x1, y1, x2, y2]
                texts.extend(ocr_out)

        if (not texts) and self.ocr_fullframe_fallback:
            texts = self._ocr_one(img, min_conf=self.ocr_min_conf_full, max_items=10)

        t_o1 = time.time()

        hit_summary = {
            "yolo_total": len(objects),
            "yolo_risk_hits": list(set(yolo_risk_hits))[:10],
            "ocr_total": len(texts),
            "qr_found": bool(qr_res.get("found", False)),
            "skin_ratio": float(skin_ratio),
            "skin_trigger": bool(skin_ratio >= self.skin_ratio_trigger),
            "time_ms": {
                "qr": int((t_q1 - t_q0) * 1000),
                "skin": int((t_s1 - t_s0) * 1000),
                "yolo": int((t_y1 - t_y0) * 1000),
                "ocr": int((t_o1 - t_o0) * 1000),
                "total": int((time.time() - t0) * 1000),
            },
        }

        return {
            "frame_path": frame_path,
            "risk_flag": bool(risk_flag),
            "hit_summary": hit_summary,
            "objects": objects,
            "texts": texts,
        }
