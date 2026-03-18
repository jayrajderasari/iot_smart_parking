"""
Geometry utilities for smart parking system.

Implements fundamental geometric calculations for:
- Intersection over Union (IoU) between vehicle bbox and parking slot
- Center deviation between vehicle and slot
- Slot diagonal and distance calculations
"""

import numpy as np
from typing import Tuple, List
from shapely.geometry import Polygon, box


class GeometryCalculator:
    """Handles geometric calculations for parking slot alignment."""

    @staticmethod
    def compute_iou(bbox: List[float], slot_polygon: Polygon) -> float:
        """
        Compute Intersection over Union (IoU) between vehicle bounding box and slot.
        
        IoU = Area(B_v ∩ P_s) / Area(B_v ∪ P_s)
        
        Args:
            bbox: [x_min, y_min, x_max, y_max] vehicle bounding box in pixels
            slot_polygon: Shapely Polygon representing the parking slot
            
        Returns:
            float: IoU score in range [0, 1]
        """
        # Convert bbox to polygon
        x_min, y_min, x_max, y_max = bbox
        vehicle_box = box(x_min, y_min, x_max, y_max)
        
        # Compute intersection and union
        intersection = vehicle_box.intersection(slot_polygon)
        union = vehicle_box.union(slot_polygon)
        
        if union.area == 0:
            return 0.0
        
        iou = intersection.area / union.area
        return float(np.clip(iou, 0, 1))

    @staticmethod
    def compute_center_deviation(bbox: List[float], slot_polygon: Polygon) -> float:
        """
        Compute normalized center deviation between vehicle and slot.
        
        D = ||c_v - c_s|| / Diagonal(P_s)
        
        where:
        - c_v: vehicle center
        - c_s: slot center
        - Diagonal(P_s): diagonal of slot bounding box
        
        Args:
            bbox: [x_min, y_min, x_max, y_max] vehicle bounding box
            slot_polygon: Shapely Polygon representing the parking slot
            
        Returns:
            float: Normalized deviation in range [0, 1]
        """
        # Vehicle center
        x_min, y_min, x_max, y_max = bbox
        c_v = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        
        # Slot center
        c_s = np.array(slot_polygon.centroid.coords[0])
        
        # Euclidean distance between centers
        center_distance = np.linalg.norm(c_v - c_s)
        
        # Slot diagonal (from bounding box of polygon)
        minx, miny, maxx, maxy = slot_polygon.bounds
        diagonal = np.sqrt((maxx - minx)**2 + (maxy - miny)**2)
        
        if diagonal == 0:
            return 1.0
        
        # Normalized deviation
        deviation = center_distance / diagonal
        return float(np.clip(deviation, 0, 1))

    @staticmethod
    def compute_alignment_score(iou: float, deviation: float) -> float:
        """
        Compute slot alignment score combining IoU and center deviation.
        
        A_s = IoU × (1 - D)
        
        Args:
            iou: Intersection over Union score [0, 1]
            deviation: Normalized center deviation [0, 1]
            
        Returns:
            float: Alignment score in range [0, 1]
                  - 1.0: Perfect alignment
                  - 0.0: Poor alignment
        """
        alignment_score = iou * (1 - deviation)
        return float(np.clip(alignment_score, 0, 1))

    @staticmethod
    def compute_slot_confidence(detector_confidence: float, alignment_score: float) -> float:
        """
        Compute overall slot confidence combining detection and alignment.
        
        C_s = C_YOLO × A_s
        
        Args:
            detector_confidence: YOLO detection confidence [0, 1]
            alignment_score: Slot alignment score [0, 1]
            
        Returns:
            float: Slot confidence in range [0, 1]
        """
        confidence = detector_confidence * alignment_score
        return float(np.clip(confidence, 0, 1))

    @staticmethod
    def is_improperly_parked(alignment_score: float, iou: float, 
                            delta: float = 0.5, epsilon: float = 0.3) -> bool:
        """
        Determine if vehicle is improperly parked.
        
        Vehicle is improperly parked if:
        A_s < δ OR IoU < ε
        
        Args:
            alignment_score: Slot alignment score
            iou: Intersection over Union score
            delta: Alignment threshold (default: 0.5)
            epsilon: IoU threshold (default: 0.3)
            
        Returns:
            bool: True if improperly parked, False otherwise
        """
        return alignment_score < delta or iou < epsilon

    @staticmethod
    def polygon_from_coordinates(coordinates: List[Tuple[float, float]]) -> Polygon:
        """
        Create a Shapely Polygon from coordinate list.
        
        Args:
            coordinates: List of (x, y) tuples representing polygon vertices
            
        Returns:
            Polygon: Shapely Polygon object
        """
        if len(coordinates) < 3:
            raise ValueError("Polygon requires at least 3 vertices")
        return Polygon(coordinates)

    @staticmethod
    def bbox_overlap_percentage(bbox: List[float], slot_polygon: Polygon) -> float:
        """
        Compute percentage of bounding box that overlaps with slot.
        
        Args:
            bbox: [x_min, y_min, x_max, y_max] vehicle bounding box
            slot_polygon: Parking slot polygon
            
        Returns:
            float: Percentage of bbox area within slot [0, 1]
        """
        x_min, y_min, x_max, y_max = bbox
        vehicle_box = box(x_min, y_min, x_max, y_max)
        
        bbox_area = vehicle_box.area
        if bbox_area == 0:
            return 0.0
        
        intersection = vehicle_box.intersection(slot_polygon)
        overlap_percentage = intersection.area / bbox_area
        
        return float(np.clip(overlap_percentage, 0, 1))
