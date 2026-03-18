"""
Occupancy Classifier

Determines parking slot occupancy status:
- Empty
- Occupied (properly)
- Occupied (improperly)

Uses detection + alignment results.
"""

from typing import Dict, Optional
from enum import Enum


class OccupancyStatus(Enum):
    """Parking slot occupancy status."""
    EMPTY = "empty"
    OCCUPIED_PROPER = "occupied"
    OCCUPIED_IMPROPER = "improper"
    UNKNOWN = "unknown"


class OccupancyClassifier:
    """Classifies slot occupancy status."""

    def __init__(self,
                 alignment_threshold: float = 0.5,
                 iou_threshold: float = 0.3,
                 confidence_threshold: float = 0.5):
        """
        Initialize occupancy classifier.
        
        Args:
            alignment_threshold: Threshold for proper parking
            iou_threshold: Minimum IoU for occupancy detection
            confidence_threshold: Minimum confidence for classification
        """
        self.alignment_threshold = alignment_threshold
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def classify(self,
                has_detection: bool,
                detection_confidence: float = 0.0,
                alignment_score=None,
                iou: float = 0.0,
                temporal_confirmed: bool = False) -> OccupancyStatus:
        """
        Classify slot occupancy.
        
        Args:
            has_detection: Whether vehicle detected in slot
            detection_confidence: Detection confidence
            alignment_score: Alignment score (float or AlignmentResult object)
            iou: Intersection over Union
            temporal_confirmed: Whether violation confirmed over time
            
        Returns:
            OccupancyStatus: Classification result
        """
        # Handle AlignmentResult object or None
        actual_alignment_score = 0.0
        actual_iou = iou
        if alignment_score is not None:
            if hasattr(alignment_score, 'alignment_score'):
                # It's an AlignmentResult object
                actual_alignment_score = alignment_score.alignment_score
                actual_iou = alignment_score.iou if iou == 0.0 else iou
            else:
                actual_alignment_score = float(alignment_score)
        
        if not has_detection or actual_iou < self.iou_threshold:
            return OccupancyStatus.EMPTY

        if detection_confidence < self.confidence_threshold:
            return OccupancyStatus.UNKNOWN

        # Vehicle detected with sufficient confidence
        if actual_alignment_score >= self.alignment_threshold and not temporal_confirmed:
            return OccupancyStatus.OCCUPIED_PROPER
        else:
            return OccupancyStatus.OCCUPIED_IMPROPER

    def batch_classify(self,
                      slot_data: Dict[int, Dict]) -> Dict[int, OccupancyStatus]:
        """
        Classify multiple slots.
        
        Args:
            slot_data: Dict mapping slot_id to classification parameters
            
        Returns:
            Dict[int, OccupancyStatus]: Classification per slot
        """
        results = {}
        for slot_id, data in slot_data.items():
            results[slot_id] = self.classify(**data)
        return results
