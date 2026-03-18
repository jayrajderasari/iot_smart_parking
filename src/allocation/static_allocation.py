"""
Static Allocation Strategy

Simple baseline allocation:
- Assigns vehicles to slots based on detector confidence only
- No temporal or alignment information
- Used for baseline comparisons
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class AllocationResult:
    """Result of vehicle-to-slot allocation."""
    vehicle_id: int
    slot_id: int
    confidence: float
    allocation_score: float


class StaticAllocationModule:
    """
    Static allocation without temporal verification.
    
    Baseline strategy: assign based on detector confidence.
    """

    def __init__(self):
        """Initialize static allocator."""
        self.allocations: Dict[int, AllocationResult] = {}

    def allocate(self,
                 vehicle_id: int,
                 detector_confidence: float,
                 available_slots: List[int],
                 slot_confidences: Dict[int, float]) -> Optional[AllocationResult]:
        """
        Allocate vehicle to best available slot.
        
        Args:
            vehicle_id: Vehicle ID
            detector_confidence: YOLO confidence
            available_slots: List of available slot IDs
            slot_confidences: Dict mapping slot_id to confidence (YOLO × alignment)
            
        Returns:
            AllocationResult or None if no suitable slot
        """
        if not available_slots:
            return None

        # Find best slot by confidence
        best_slot = None
        best_confidence = 0.0

        for slot_id in available_slots:
            confidence = slot_confidences.get(slot_id, 0.0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_slot = slot_id

        if best_slot is None or best_confidence < 0.3:
            return None

        result = AllocationResult(
            vehicle_id=vehicle_id,
            slot_id=best_slot,
            confidence=detector_confidence,
            allocation_score=best_confidence
        )

        self.allocations[vehicle_id] = result
        return result

    def get_allocation(self, vehicle_id: int) -> Optional[AllocationResult]:
        """Get allocation for vehicle."""
        return self.allocations.get(vehicle_id)

    def clear(self) -> None:
        """Clear all allocations."""
        self.allocations.clear()
