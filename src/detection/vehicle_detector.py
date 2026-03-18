"""
Vehicle Detection Module using YOLO

Loads pre-trained YOLO model and performs vehicle detection.
Returns bounding boxes with confidence scores.

Offline / HPC notes:
    - The model file (yolov8n.pt) must be present locally.
    - YOLO_OFFLINE=1 environment variable disables any auto-download attempt.
    - ultralytics telemetry/update checks are suppressed at import time.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Offline / HPC guard ───────────────────────────────────────────────────
# These must be set before ultralytics is imported for the first time.
# They prevent any network access (model downloads, telemetry, update checks).
os.environ.setdefault("YOLO_OFFLINE", "1")        # ultralytics offline mode
os.environ.setdefault("YOLO_VERBOSE", "0")         # suppress ultralytics console spam
os.environ.setdefault("WANDB_DISABLED", "true")    # disable Weights & Biases
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("YOLO_TELEMETRY", "0")       # disable ultralytics telemetry
os.environ.setdefault("HUB_OFFLINE", "1")          # disable Ultralytics HUB
# Torch Hub cache – point to local dir so torch.hub.load() doesn't fetch
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[3]))


@dataclass
class Detection:
    """Container for a single vehicle detection."""
    bbox: List[float]  # [x_min, y_min, x_max, y_max]
    confidence: float  # Detection confidence [0, 1]
    class_id: int
    class_name: str


class YOLOVehicleDetector:
    """
    Vehicle detector using YOLO.
    
    Supports YOLOv7 or YOLOv8 (configurable).
    """

    def __init__(self, model_path: str, model_type: str = "yolov8", 
                 device: str = "cpu", confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to pretrained model weights
            model_type: Type of YOLO model ("yolov7" or "yolov8")
            device: Device to run inference on ("cpu" or "cuda")
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
        
        # Auto-load model on init
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load model; log failures instead of silently swallowing them."""
        try:
            self.load_model()
        except Exception as exc:
            logger.error("Model loading failed: %s", exc)

    def load_model(self) -> None:
        """
        Load pre-trained YOLO model.
        
        Note: Actual implementation requires ultralytics or custom YOLO library.
        This is a template structure.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}\n"
                "Copy yolov8n.pt to the project root before running on HPC."
            )

        try:
            if self.model_type == "yolov8":
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                self.model.to(self.device)
            elif self.model_type == "yolov7":
                import torch
                self.model = torch.hub.load(
                    "WongKinYiu/yolov7", "custom",
                    path=str(self.model_path), force_reload=False,
                    trust_repo=True,
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.is_loaded = True
            logger.info("%s model loaded from %s  (device=%s)",
                        self.model_type.upper(), self.model_path, self.device)
        except ImportError as exc:
            logger.error(
                "Could not import model library: %s\n"
                "Ensure ultralytics, torch, and torchvision are installed.", exc
            )

    def detect(self, image_input, return_tensors: bool = False) -> List[Detection]:
        """
        Perform vehicle detection on an image.
        
        Args:
            image_input: Path to input image (str) or numpy array (BGR image)
            return_tensors: Whether to return raw tensors
            
        Returns:
            List[Detection]: List of detected vehicles with bboxes and confidences
        """
        if not self.is_loaded:
            # Return empty list if model not loaded (graceful degradation)
            return []

        detections = []
        
        try:
            if self.model_type == "yolov8":
                # image_input can be path (str) or numpy array
                results = self.model(image_input, conf=self.confidence_threshold)
                
                for result in results:
                    if result.boxes is None:
                        continue
                    for detection in result.boxes:
                        bbox = detection.xyxy[0].cpu().numpy().tolist()  # [x_min, y_min, x_max, y_max]
                        conf = float(detection.conf[0].cpu().numpy())
                        cls_id = int(detection.cls[0].cpu().numpy())
                        
                        # Filter: typically car class (id varies by model)
                        # For COCO: car=2, person=0, truck=8, bus=5
                        if conf >= self.confidence_threshold and cls_id in [2, 5, 7, 8]:  # car, bus, truck variants
                            detections.append(Detection(
                                bbox=bbox,
                                confidence=conf,
                                class_id=cls_id,
                                class_name=self._get_class_name(cls_id)
                            ))
            
            elif self.model_type == "yolov7":
                # YOLOv7 inference - also accepts path or array
                results = self.model(image_input)
                detections_raw = results.xyxy[0].cpu().numpy()
                
                for det in detections_raw:
                    x_min, y_min, x_max, y_max, conf, cls_id = det
                    cls_id = int(cls_id)
                    
                    if conf >= self.confidence_threshold and cls_id in [2, 5, 7, 8]:
                        detections.append(Detection(
                            bbox=[float(x_min), float(y_min), float(x_max), float(y_max)],
                            confidence=float(conf),
                            class_id=cls_id,
                            class_name=self._get_class_name(cls_id)
                        ))
        
        except Exception as exc:
            logger.error("Detection error on %s: %s", image_input, exc)
            return []

        return detections

    def detect_batch(self, image_paths: List[str]) -> List[List[Detection]]:
        """
        Perform batch detection on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List[List[Detection]]: Detections for each image
        """
        batch_results = []
        for image_path in image_paths:
            detections = self.detect(image_path)
            batch_results.append(detections)
        return batch_results

    @staticmethod
    def _get_class_name(class_id: int) -> str:
        """Map COCO class IDs to names."""
        coco_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            8: "boat", 9: "traffic light"
        }
        return coco_names.get(class_id, "unknown")

    def visualize_detections(self, image_path: str, detections: List[Detection], 
                            output_path: Optional[str] = None) -> None:
        """
        Visualize detections on image (optional).
        
        Args:
            image_path: Input image path
            detections: List of Detection objects
            output_path: Optional path to save visualization
        """
        try:
            import cv2
            
            image = cv2.imread(image_path)
            
            for det in detections:
                x_min, y_min, x_max, y_max = [int(x) for x in det.bbox]
                
                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw label
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(image, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if output_path:
                cv2.imwrite(output_path, image)
            
            # Display (comment out if running headless)
            # cv2.imshow("Detections", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        except ImportError:
            logger.warning("OpenCV not installed – skipping visualisation.")
