"""Smart Parking System - Allocation Module"""

from src.allocation.static_allocation import StaticAllocationModule, AllocationResult
from src.allocation.sorting_allocation import SortingAllocationModule, VehicleSlotPair
from src.allocation.dynamic_allocation import DynamicAllocationModule, DynamicAllocationDecision

__all__ = [
    'StaticAllocationModule',
    'AllocationResult',
    'SortingAllocationModule',
    'VehicleSlotPair',
    'DynamicAllocationModule',
    'DynamicAllocationDecision'
]
