"""
Test script to verify smart parking system functionality.

Tests all major components with sample data.
Run this before submitting an HPC job to confirm the environment is correct.
"""

import json
import sys
from pathlib import Path
from shapely.geometry import Polygon

from src.utils.logger import setup_logging, get_logger

# Initialise logging so test output is captured alongside print()-based results.
setup_logging()
logger = get_logger(__name__)

from src.detection.slot_alignment import SlotAlignmentModule
from src.detection.occupancy_classifier import OccupancyClassifier, OccupancyStatus
from src.temporal.temporal_verification import TemporalVerificationModule
from src.penalization.penalty_engine import PenaltyEngine
from src.utils.geometry import GeometryCalculator


def test_geometry_calculator():
    """Test geometry calculations."""
    print("\n" + "="*70)
    print("TEST 1: Geometry Calculator")
    print("="*70)
    
    geom = GeometryCalculator()
    
    # Create test data
    bbox = [10, 10, 60, 60]  # Vehicle bbox
    slot = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])  # Slot polygon
    
    # Test IoU
    iou = geom.compute_iou(bbox, slot)
    print(f"\n✓ IoU Calculation: {iou:.4f}")
    assert 0 <= iou <= 1, "IoU should be in [0, 1]"
    
    # Test center deviation
    dev = geom.compute_center_deviation(bbox, slot)
    print(f"✓ Center Deviation: {dev:.4f}")
    assert 0 <= dev <= 1, "Deviation should be in [0, 1]"
    
    # Test alignment score
    alignment = geom.compute_alignment_score(iou, dev)
    print(f"✓ Alignment Score: {alignment:.4f}")
    assert 0 <= alignment <= 1, "Alignment should be in [0, 1]"
    
    # Test confidence
    confidence = geom.compute_slot_confidence(0.9, alignment)
    print(f"✓ Slot Confidence: {confidence:.4f}")
    
    print("\n✅ Geometry Calculator: PASSED")


def test_slot_alignment():
    """Test slot alignment module."""
    print("\n" + "="*70)
    print("TEST 2: Slot Alignment Module")
    print("="*70)
    
    alignment = SlotAlignmentModule(delta=0.5, epsilon=0.3)
    
    # Test case 1: Perfect parking
    bbox1 = [20, 20, 80, 80]
    slot = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    
    result1 = alignment.analyze_vehicle_slot_alignment(bbox1, slot, 0.95)
    print(f"\n✓ Perfect parking:")
    print(f"  Alignment Score: {result1.alignment_score:.4f}")
    print(f"  Is Improper: {result1.is_improperly_parked}")
    print(f"  Severity: {result1.severity_score:.4f}")
    
    # Test case 2: Poor parking (off-center)
    bbox2 = [5, 10, 40, 60]
    result2 = alignment.analyze_vehicle_slot_alignment(bbox2, slot, 0.7)
    print(f"\n✓ Poor parking:")
    print(f"  Alignment Score: {result2.alignment_score:.4f}")
    print(f"  Is Improper: {result2.is_improperly_parked}")
    print(f"  Severity: {result2.severity_score:.4f}")
    
    print("\n✅ Slot Alignment: PASSED")


def test_temporal_verification():
    """Test temporal verification module."""
    print("\n" + "="*70)
    print("TEST 3: Temporal Verification Module")
    print("="*70)
    
    temporal = TemporalVerificationModule(confirmation_threshold=3, window_size=10)
    
    # Simulate frames with violations
    print("\n✓ Processing 5 frames...")
    
    for frame in range(1, 6):
        observations = {
            1: (True, 0.4, 0.6, 0.8),  # Slot 1: improper
            2: (False, 0.8, 0.2, 0.9),  # Slot 2: proper
            3: (True, 0.3, 0.7, 0.75)   # Slot 3: improper
        }
        
        confirmed = temporal.process_frame_observations(frame, observations)
        
        print(f"  Frame {frame}: {sum(confirmed.values())} confirmed violations")
    
    # Check results
    confirmed_violations = temporal.get_all_confirmed_violations()
    print(f"\n✓ Confirmed violations: {confirmed_violations}")
    
    summary = temporal.get_summary(1)
    print(f"✓ Slot 1 temporal summary:")
    print(f"  Violation count: {summary['violation_count']}")
    print(f"  Violation confirmed: {summary['violation_confirmed']}")
    print(f"  Duration: {summary['duration_seconds']:.1f} seconds")
    
    print("\n✅ Temporal Verification: PASSED")


def test_penalty_engine():
    """Test penalty scoring."""
    print("\n" + "="*70)
    print("TEST 4: Penalty Engine")
    print("="*70)
    
    engine = PenaltyEngine(alpha=0.3, beta=0.4, gamma=0.3)
    
    # Test case 1: Short, minor violation
    print("\n✓ Case 1: Short, minor violation")
    breakdown1 = engine.compute_penalty(
        slot_id=1,
        severity_score=0.2,
        duration_seconds=60,
        violation_count=1
    )
    print(f"  Total Penalty: {breakdown1.total_penalty:.2f}")
    print(f"  Duration Component: {breakdown1.duration_component:.4f}")
    print(f"  Severity Component: {breakdown1.severity_component:.4f}")
    print(f"  Repetition Component: {breakdown1.repetition_component:.4f}")
    
    # Test case 2: Long, severe violation
    print("\n✓ Case 2: Long, severe violation")
    breakdown2 = engine.compute_penalty(
        slot_id=2,
        severity_score=0.9,
        duration_seconds=3600,
        violation_count=3
    )
    print(f"  Total Penalty: {breakdown2.total_penalty:.2f}")
    print(f"  Duration Component: {breakdown2.duration_component:.4f}")
    print(f"  Severity Component: {breakdown2.severity_component:.4f}")
    print(f"  Repetition Component: {breakdown2.repetition_component:.4f}")
    
    assert breakdown2.total_penalty > breakdown1.total_penalty, \
        "Worse violation should have higher penalty"
    
    print("\n✅ Penalty Engine: PASSED")


def test_occupancy_classifier():
    """Test occupancy classification."""
    print("\n" + "="*70)
    print("TEST 5: Occupancy Classifier")
    print("="*70)
    
    classifier = OccupancyClassifier()
    
    # Test cases
    cases = [
        {
            "name": "Empty slot",
            "has_detection": False,
            "occupancy": "empty"
        },
        {
            "name": "Properly parked",
            "has_detection": True,
            "detection_confidence": 0.95,
            "alignment_score": 0.8,
            "iou": 0.7,
            "temporal_confirmed": False,
            "occupancy": "occupied"
        },
        {
            "name": "Improperly parked",
            "has_detection": True,
            "detection_confidence": 0.9,
            "alignment_score": 0.4,
            "iou": 0.5,
            "temporal_confirmed": True,
            "occupancy": "improper"
        }
    ]
    
    for case in cases:
        status = classifier.classify(
            has_detection=case["has_detection"],
            detection_confidence=case.get("detection_confidence", 0),
            alignment_score=case.get("alignment_score", 0),
            iou=case.get("iou", 0),
            temporal_confirmed=case.get("temporal_confirmed", False)
        )
        
        print(f"\n✓ {case['name']}: {status.value}")
        assert status.value == case["occupancy"], f"Wrong classification for {case['name']}"
    
    print("\n✅ Occupancy Classifier: PASSED")


def test_end_to_end():
    """End-to-end system test."""
    print("\n" + "="*70)
    print("TEST 6: End-to-End System Test")
    print("="*70)
    
    # Load sample slot coordinates
    slots_file = Path("data/cnrpark/slots_coordinates.json")
    if slots_file.exists():
        with open(slots_file) as f:
            slots_data = json.load(f)
        
        print(f"\n✓ Loaded {len(slots_data)} parking slots")
        
        # Convert to polygons (handle both string and list coordinates)
        slot_polygons = {}
        for slot_id, slot_info in slots_data.items():
            coords = slot_info.get("coordinates", [])
            # Skip if coordinates are not proper lists
            if not coords or not isinstance(coords[0], (list, tuple)):
                continue
            try:
                slot_polygons[int(slot_id)] = Polygon(coords)
            except (ValueError, TypeError):
                continue
        
        if not slot_polygons:
            print("⚠ No valid slot polygons found, generating synthetic ones")
            for i in range(1, min(6, len(slots_data) + 1)):
                slot_polygons[i] = Polygon([
                    [i*200, i*200],
                    [i*200 + 150, i*200],
                    [i*200 + 150, i*200 + 150],
                    [i*200, i*200 + 150]
                ])
        
        # Initialize modules
        alignment = SlotAlignmentModule()
        temporal = TemporalVerificationModule()
        penalty_engine = PenaltyEngine()
        
        # Simulate detection on first slot
        slot_id = min(slot_polygons.keys())
        slot_polygon = slot_polygons[slot_id]
        
        # Vehicle bbox overlapping the slot
        bbox = [
            slot_polygon.bounds[0] + 10,
            slot_polygon.bounds[1] + 10,
            slot_polygon.bounds[2] - 10,
            slot_polygon.bounds[3] - 10
        ]
        
        # Analyze alignment
        result = alignment.analyze_vehicle_slot_alignment(bbox, slot_polygon, 0.9)
        print(f"\n✓ Slot {slot_id} analysis:")
        print(f"  IoU: {result.iou:.4f}")
        print(f"  Alignment Score: {result.alignment_score:.4f}")
        print(f"  Improper: {result.is_improperly_parked}")
        
        # Track temporally
        for frame in range(5):
            observations = {
                slot_id: (
                    result.is_improperly_parked,
                    result.alignment_score,
                    result.severity_score,
                    0.9
                )
            }
            confirmed = temporal.process_frame_observations(frame, observations)
        
        # Compute penalty if confirmed
        if temporal.get_all_confirmed_violations():
            severity = temporal.get_violation_severity(slot_id)
            duration = temporal.get_violation_duration_seconds(slot_id)
            
            breakdown = penalty_engine.compute_penalty(
                slot_id, severity, duration, 1
            )
            print(f"\n✓ Violation penalty:")
            print(f"  Severity: {severity:.4f}")
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Total Penalty: {breakdown.total_penalty:.2f}")
        
        print("\n✅ End-to-End Test: PASSED")
    else:
        print("⚠ Sample slots file not found")


def run_all_tests():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("SMART PARKING SYSTEM - TEST SUITE")
    logger.info("=" * 70)

    try:
        test_geometry_calculator()
        test_slot_alignment()
        test_temporal_verification()
        test_penalty_engine()
        test_occupancy_classifier()
        test_end_to_end()

        logger.info("=" * 70)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 70)
        return True

    except AssertionError as exc:
        logger.error("TEST FAILED: %s", exc)
        return False
    except Exception:
        import traceback
        logger.error("UNEXPECTED ERROR\n%s", traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
