"""
Sorting-Based Allocation Strategy

Allocates vehicles by ranking slots/vehicles:
- Sorts vehicles by confidence
- Sorts slots by alignment quality
- Matches in order
- Greedy approach
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class VehicleSlotPair:
    """Pairing of vehicle to slot with score."""
    vehicle_id: int
    slot_id: int
    score: float  # Combined matching score


class SortingAllocationModule:
    """
    Rank-based allocation using sorting.
    
    Greedy strategy: match highest-confidence vehicles to best slots.
    """

    def __init__(self):
        """Initialize sorting allocator."""
        self.allocations: Dict[int, VehicleSlotPair] = {}

    def allocate_batch(self,
                       vehicle_ids: List[int],
                       vehicle_confidences: List[float],
                       available_slots: List[int],
                       slot_confidences: Dict[int, float],
                       slot_alignment_scores: Dict[int, float]) -> List[VehicleSlotPair]:
        """
        Allocate multiple vehicles using sorting.
        
        Algorithm:
        1. Sort vehicles by confidence (descending)
        2. Sort available slots by alignment (descending)
        3. Greedily match in order
        
        Args:
            vehicle_ids: List of vehicle IDs
            vehicle_confidences: Corresponding YOLO confidences
            available_slots: Available slot IDs
            slot_confidences: Slot confidence scores
            slot_alignment_scores: Alignment scores per slot
            
        Returns:
            List[VehicleSlotPair]: Allocations made
        """
        self.allocations.clear()
        results = []

        # Sort vehicles by confidence (descending)
        vehicle_ranking = sorted(
            zip(vehicle_ids, vehicle_confidences),
            key=lambda x: x[1],
            reverse=True
        )

        # Sort slots by alignment (descending)
        slot_ranking = sorted(
            available_slots,
            key=lambda s: slot_alignment_scores.get(s, 0),
            reverse=True
        )

        allocated_slots = set()

        # Greedy matching
        for vehicle_id, v_conf in vehicle_ranking:
            for slot_id in slot_ranking:
                if slot_id in allocated_slots:
                    continue

                s_conf = slot_confidences.get(slot_id, 0.0)
                alignment = slot_alignment_scores.get(slot_id, 0.0)

                # Combined score
                score = v_conf * s_conf * alignment

                if score >= 0.3:  # Threshold
                    pair = VehicleSlotPair(
                        vehicle_id=vehicle_id,
                        slot_id=slot_id,
                        score=score
                    )
                    results.append(pair)
                    self.allocations[vehicle_id] = pair
                    allocated_slots.add(slot_id)
                    break

        return results

    def get_allocation(self, vehicle_id: int) -> Optional[VehicleSlotPair]:
        """Get allocation for vehicle."""
        return self.allocations.get(vehicle_id)

    def clear(self) -> None:
        """Clear allocations."""
        self.allocations.clear()
