"""Conflict-pattern crash codes (proposal §D5 / Conflict Measures reference).

The engine already derives an encounter type from geometry; this module attaches
the reference document's crash codes to it so each conflict record is traceable
to the taxonomy TTS supplied.

Codes, per the Conflict Measures reference:

    Vehicle-vehicle
      3.1        Head-on
      3.2, 3.3   Opposing - oncoming        (PET)
      3.4        Left / Right turning
      4.1-4.3    Rear-end
      4.4-4.6    Lane change / Side-swipe
      2.2, 2.6   Adjacent direction merging
      2.1        Through - through          (PET)
      2.3        Left turning - through from left (PET)
      4.7, 4.8   Parallel lane turning      -- NOT detected (needs trajectory shape)
      5.6        U-turn                     -- NOT detected (needs trajectory shape)

    Vehicle-pedestrian / bicycle
      1.1        Near-side
      1.3        Far-side
      1.4        Right turn
      1.5        Left turn
      1.6        Through/Right/Left (near-side)
      1.7        Through (far-side)

The two undetected vehicle-vehicle families are listed explicitly rather than
silently omitted: they need full trajectory-shape analysis, and the proposal
scopes pattern outputs to "where observable and feasible".

VRU sub-side (near vs far) needs approach-side geometry relative to the crossing,
so vehicle-VRU encounters carry the family codes they can be resolved to and no
more.
"""

from typing import Dict, List

# encounter_type (as produced by ConflictEngine._encounter) -> crash codes
CRASH_CODES: Dict[str, List[str]] = {
    # --- vehicle-vehicle ---------------------------------------------------
    "rear_end":                     ["4.1", "4.2", "4.3"],
    "lane_change_merge":            ["4.4", "4.5", "4.6"],
    "turning":                      ["3.4"],
    "head_on":                      ["3.1"],
    "crossing":                     ["2.1"],
    "opposing_through":             ["3.2", "3.3"],
    # --- vehicle-VRU (pedestrian / bicycle) --------------------------------
    # Family-level only: near-side vs far-side (1.1 vs 1.3) requires knowing
    # which side of the crossing the vehicle approaches from.
    "vehicle_vru_crossing":         ["1.6"],
    "vehicle_vru_opposing_through": ["1.7"],
    "vehicle_vru_rear_end":         ["1.6"],
    "vehicle_vru_turning":          ["1.4", "1.5"],
    "vehicle_vru_lane_change_merge": ["1.6"],
    "vehicle_vru_head_on":          ["1.7"],
}

# Documented as out of scope rather than quietly missing.
NOT_DETECTED = {
    "parallel_lane_turning": ["4.7", "4.8"],
    "u_turn": ["5.6"],
}


def crash_codes_for(encounter_type: str) -> List[str]:
    """Reference crash codes for an encounter type ([] when unmapped)."""
    return list(CRASH_CODES.get(encounter_type, []))
