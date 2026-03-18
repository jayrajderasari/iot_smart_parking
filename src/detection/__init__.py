"""Smart Parking System - Detection Module"""

from src.detection.vehicle_detector import YOLOVehicleDetector, Detection
from src.detection.slot_alignment import SlotAlignmentModule, AlignmentResult
from src.detection.occupancy_classifier import OccupancyClassifier, OccupancyStatus

__all__ = [
    'YOLOVehicleDetector',
    'Detection',
    'SlotAlignmentModule',
    'AlignmentResult',
    'OccupancyClassifier',
    'OccupancyStatus'
]
