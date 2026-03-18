"""
Slot Alignment Module

Computes the alignment of detected vehicles with parking slot boundaries.
This is the novelty: combining IoU-based overlap with center deviation.

Core formula:
    A_s = IoU × (1 - D)
    
Where:
    - IoU = Area(B_v ∩ P_s) / Area(B_v ∪ P_s)
    - D = ||c_v - c_s|| / Diagonal(P_s)
    - B_v: vehicle bounding box
    - P_s: slot polygon
    - c_v: vehicle center, c_s: slot center
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from shapely.geometry import Polygon
from src.utils.geometry import GeometryCalculator


@dataclass
class AlignmentResult:
    """Container for alignment computation results."""
    iou: float
    center_deviation: float
    alignment_score: float
    is_improperly_parked: bool
    severity_score: float  # 1 - alignment_score
    overlap_percentage: float


class SlotAlignmentModule:
    """
    Analyzes how well a vehicle fits within its assigned parking slot.
    
    Used for:
    - Detecting improper parking (boundary violations)
    - Computing parking severity scores
    - Allocation decision making
    - Confidence filtering in penalization
    """

    def __init__(self, delta: float = 0.5, epsilon: float = 0.3):
        """
        Initialize alignment module with thresholds.
        
        Args:
            delta: Alignment score threshold for improper parking (default: 0.5)
            epsilon: IoU threshold for improper parking (default: 0.3)
        """
        self.delta = delta
        self.epsilon = epsilon
        self.geometry = GeometryCalculator()

    def analyze_vehicle_slot_alignment(self, 
                                      bbox: List[float],
                                      slot_polygon: Polygon,
                                      detector_confidence: float = 1.0) -> AlignmentResult:
        """
        Analyze alignment between vehicle bounding box and parking slot.
        
        Args:
            bbox: [x_min, y_min, x_max, y_max] vehicle bounding box
            slot_polygon: Shapely Polygon representing parking slot
            detector_confidence: YOLO detector confidence score
            
        Returns:
            AlignmentResult: Comprehensive alignment metrics
        """
        # Compute base metrics
        iou = self.geometry.compute_iou(bbox, slot_polygon)
        center_deviation = self.geometry.compute_center_deviation(bbox, slot_polygon)
        overlap_percentage = self.geometry.bbox_overlap_percentage(bbox, slot_polygon)
        
        # Alignment score (main novelty metric)
        alignment_score = self.geometry.compute_alignment_score(iou, center_deviation)
        
        # Severity: inverse of alignment
        severity_score = 1.0 - alignment_score
        
        # Check if improperly parked
        is_improper = self.geometry.is_improperly_parked(
            alignment_score, iou, self.delta, self.epsilon
        )
        
        return AlignmentResult(
            iou=iou,
            center_deviation=center_deviation,
            alignment_score=alignment_score,
            is_improperly_parked=is_improper,
            severity_score=severity_score,
            overlap_percentage=overlap_percentage
        )

    def batch_analyze_vehicles(self,
                              bboxes: List[List[float]],
                              slot_polygons: List[Polygon],
                              detector_confidences: List[float] = None) -> List[AlignmentResult]:
        """
        Analyze multiple vehicles against multiple slots.
        
        Args:
            bboxes: List of vehicle bounding boxes
            slot_polygons: List of slot polygons
            detector_confidences: Optional list of detector confidence scores
            
        Returns:
            List[AlignmentResult]: Alignment results for each vehicle
        """
        if detector_confidences is None:
            detector_confidences = [1.0] * len(bboxes)
        
        results = []
        for bbox, slot, conf in zip(bboxes, slot_polygons, detector_confidences):
            result = self.analyze_vehicle_slot_alignment(bbox, slot, conf)
            results.append(result)
        
        return results

    def compute_slot_confidence(self,
                               alignment_result: AlignmentResult,
                               detector_confidence: float) -> float:
        """
        Compute slot confidence combining detection and alignment.
        
        C_s = C_YOLO × A_s
        
        Args:
            alignment_result: AlignmentResult from analyze_vehicle_slot_alignment
            detector_confidence: YOLO confidence score
            
        Returns:
            float: Slot confidence [0, 1]
        """
        return self.geometry.compute_slot_confidence(
            detector_confidence, alignment_result.alignment_score
        )

    def filter_by_confidence(self,
                            results: List[AlignmentResult],
                            detector_confidences: List[float],
                            min_confidence: float = 0.5) -> Tuple[List[AlignmentResult], 
                                                                   List[int]]:
        """
        Filter results by slot confidence threshold.
        
        Args:
            results: List of alignment results
            detector_confidences: Corresponding detector confidences
            min_confidence: Minimum required slot confidence
            
        Returns:
            Tuple of (filtered_results, kept_indices)
        """
        filtered_results = []
        kept_indices = []
        
        for idx, (result, conf) in enumerate(zip(results, detector_confidences)):
            slot_conf = self.compute_slot_confidence(result, conf)
            if slot_conf >= min_confidence:
                filtered_results.append(result)
                kept_indices.append(idx)
        
        return filtered_results, kept_indices

    def get_alignment_report(self, result: AlignmentResult) -> Dict:
        """
        Generate human-readable alignment report.
        
        Args:
            result: AlignmentResult from analyze_vehicle_slot_alignment
            
        Returns:
            Dict: Formatted alignment report
        """
        status = "IMPROPER" if result.is_improperly_parked else "PROPER"
        
        return {
            "status": status,
            "alignment_score": round(result.alignment_score, 3),
            "severity_score": round(result.severity_score, 3),
            "iou": round(result.iou, 3),
            "center_deviation": round(result.center_deviation, 3),
            "overlap_percentage": f"{result.overlap_percentage * 100:.1f}%",
            "thresholds": {
                "delta": self.delta,
                "epsilon": self.epsilon
            }
        }
