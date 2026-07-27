"""Conflict-measure primitives (proposal §4.5), all evaluated in the BEV plane.

Unit note (BEV-only baseline, §4.2 / §3.3)
------------------------------------------
Positions and velocities live in the internally-consistent transformed plane.
TTC (1-D and 2-D) and PET are scale-invariant / purely time-based, so they are
valid in BEV-only mode. DRAC and Delta-V carry physical units and are only
meaningful once metric scale is validated — the engine flags them conditional;
these functions still compute them so the outputs are ready the moment a
validated scale is supplied.

Every function here is pure and side-effect free. Angles are radians.
"""

import math
from typing import Optional, Tuple

import numpy as np

EPS = 1e-9


# ── 1-D longitudinal TTC (§4.5) ────────────────────────────────────────────

def ttc_rear_end(x_lead: float, x_follow: float,
                 v_lead: float, v_follow: float, d_lead: float = 0.0) -> Optional[float]:
    """Rear-end TTC = (X1 − X2 − D_L) / (v2 − v1), valid only while the follower
    is closing on the leader (v_follow > v_lead). Returns None when not closing,
    0.0 when the gap has already collapsed."""
    closing = v_follow - v_lead
    if closing <= EPS:
        return None
    gap = x_lead - x_follow - d_lead
    if gap <= 0.0:
        return 0.0
    return gap / closing


def ttc_head_on(x_lead: float, x_follow: float,
                v_lead: float, v_follow: float) -> Optional[float]:
    """Head-on TTC = (X1 − X2) / (v1 + v2). None if the pair is not approaching
    (non-positive combined speed) or already crossed."""
    denom = v_lead + v_follow
    if denom <= EPS:
        return None
    gap = x_lead - x_follow
    if gap < 0.0:
        return None
    return gap / denom


# ── interaction angle ───────────────────────────────────────────────────────

def interaction_angle(v1, v2) -> float:
    """Angle in radians, [0, π], between two velocity vectors. Returns 0.0 if
    either vector is ~stationary. Used for encounter typing and the Delta-V θ."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < EPS or n2 < EPS:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.arccos(cos))


# ── 2-D TTC via axis-aligned footprints (proposal default, §4.5) ────────────

def _axis_overlap(a_min: float, a_max: float,
                  b_min: float, b_max: float, v: float) -> Optional[Tuple[float, float]]:
    """Entry/exit times over which a moving 1-D interval B (bounds b_min..b_max
    at t=0, velocity v) overlaps the fixed interval A (a_min..a_max). Returns
    (-inf, +inf) when v≈0 and they already overlap, or None when v≈0 and they
    never touch on this axis."""
    if abs(v) < EPS:
        if b_max >= a_min and a_max >= b_min:
            return (-np.inf, np.inf)
        return None
    if v > 0:
        t_enter = (a_min - b_max) / v
        t_exit = (a_max - b_min) / v
    else:
        t_enter = (a_max - b_min) / v
        t_exit = (a_min - b_max) / v
    return (t_enter, t_exit)


def ttc_2d_aabb(p1, v1, half1, p2, v2, half2, tmax: float = np.inf) -> Optional[float]:
    """2-D TTC via axis-aligned BEV footprints.

    Rectangle 1 (centre ``p1``, half-extents ``half1`` = (hx, hy)) is held
    fixed; rectangle 2 is advanced under the relative velocity v_rel = v2 − v1.
    Returns the earliest non-negative time the footprints intersect, or None if
    they never do within [0, tmax]. Matches the entry/exit-time construction in
    proposal §4.5.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    half1 = np.asarray(half1, dtype=np.float64)
    half2 = np.asarray(half2, dtype=np.float64)
    v_rel = np.asarray(v2, dtype=np.float64) - np.asarray(v1, dtype=np.float64)

    a_min, a_max = p1 - half1, p1 + half1
    b_min, b_max = p2 - half2, p2 + half2

    tx = _axis_overlap(a_min[0], a_max[0], b_min[0], b_max[0], v_rel[0])
    if tx is None:
        return None
    ty = _axis_overlap(a_min[1], a_max[1], b_min[1], b_max[1], v_rel[1])
    if ty is None:
        return None

    t_enter = max(tx[0], ty[0])
    t_exit = min(tx[1], ty[1])
    if t_enter <= t_exit and t_exit > 0.0:
        ttc = max(t_enter, 0.0)
        if ttc <= tmax:
            return float(ttc)
    return None


# ── 2-D TTC via oriented footprints (higher fidelity, optional §4.5) ────────

def rect_corners(center, theta: float, length: float, width: float) -> np.ndarray:
    """Four corners of an oriented rectangle: ``length`` along heading θ,
    ``width`` across it."""
    c = np.asarray(center, dtype=np.float64)
    ct, st = np.cos(theta), np.sin(theta)
    fwd = np.array([ct, st])
    left = np.array([-st, ct])
    hl, hw = length / 2.0, width / 2.0
    return np.array([
        c + hl * fwd + hw * left,
        c + hl * fwd - hw * left,
        c - hl * fwd - hw * left,
        c - hl * fwd + hw * left,
    ])


def _obb_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Separating-axis test for two convex quads. True if they overlap."""
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            na = float(np.linalg.norm(axis))
            if na < EPS:
                continue
            axis /= na
            pa = poly_a @ axis
            pb = poly_b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False  # gap found on this axis -> no overlap
    return True


def ttc_2d_obb(p1, v1, theta1, L1, W1, p2, v2, theta2, L2, W2,
               tmax: float = 4.0, dt: float = 0.05) -> Optional[float]:
    """Oriented-footprint TTC. Advances both footprints under constant velocity
    (headings held fixed over the short forecast) and returns the first contact
    time found by a separating-axis test, or None if none within [0, tmax].
    Higher fidelity than the AABB form for turning / angled approaches."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    if _obb_overlap(rect_corners(p1, theta1, L1, W1),
                    rect_corners(p2, theta2, L2, W2)):
        return 0.0

    t = dt
    while t <= tmax:
        if _obb_overlap(rect_corners(p1 + v1 * t, theta1, L1, W1),
                        rect_corners(p2 + v2 * t, theta2, L2, W2)):
            return float(t)
        t += dt
    return None


# ── PET (§4.5) ───────────────────────────────────────────────────────────────

def pet(t_first_leaves_s: float, t_second_arrives_s: float) -> float:
    """Post-encroachment time = t_S − t_F (seconds), where t_F is when the first
    road user clears the conflict zone and t_S is when the second reaches it.
    A non-positive result means the two occupied the zone simultaneously — a
    collision course rather than a clean encroachment gap."""
    return float(t_second_arrives_s - t_first_leaves_s)


# ── DRAC (§4.5) ──────────────────────────────────────────────────────────────

def drac(x_lead: float, x_follow: float,
         v_lead: float, v_follow: float, d_lead: float = 0.0) -> Optional[float]:
    """Deceleration to avoid crash = (v_F − v_L)² / (2 (x_L − x_F − D_L)).
    None when not closing or the effective gap has collapsed. Metric — treat as
    conditional in BEV-only mode."""
    dv = v_follow - v_lead
    if dv <= EPS:
        return None
    gap = x_lead - x_follow - d_lead
    if gap <= EPS:
        return None
    return (dv * dv) / (2.0 * gap)


def drac_2d(p_a, v_a, p_b, v_b, radius, closing_eps=0.1) -> Optional[float]:
    """2-D generalisation of DRAC (RITSMS form), in the BEV plane.

        r        = p_b - p_a           v_rel = v_b - v_a
        closing  = -(r·v_rel)/|r|      (positive while the gap is shrinking)
        gap      = |r| - radius        (radius = r_a + r_b + buffer)
        DRAC     = closing^2 / (2·gap)

    Reduces to the 1-D PDF formula in the collinear case. Returns None when
    the pair is not actively closing (r·v_rel >= -eps) or already inside the
    contact envelope (gap <= 0). Metric — conditional in BEV-only mode."""
    p_a = np.asarray(p_a, dtype=np.float64); p_b = np.asarray(p_b, dtype=np.float64)
    v_a = np.asarray(v_a, dtype=np.float64); v_b = np.asarray(v_b, dtype=np.float64)
    r = p_b - p_a
    v = v_b - v_a
    r2 = float(r @ r)
    if r2 <= 0.0:
        return None
    rv = float(r @ v)
    if rv >= -closing_eps:
        return None
    dist = math.sqrt(r2)
    gap = dist - radius
    if gap <= 0.0:
        return None
    closing = -rv / dist
    return (closing * closing) / (2.0 * gap)


# ── Delta-V (§4.5, Appendix B) ──────────────────────────────────────────────

def delta_v(v_f: float, v_s: float, m_f: float, m_s: float, theta: float) -> float:
    """Post-collision velocity change under the perfectly-inelastic
    momentum-conservation assumption. Returns max(ΔV_F, ΔV_S). θ in radians.

    ΔV_F = m_S·√(V_F² + V_S² − 2 V_F V_S cos θ) / (m_F + m_S)
    ΔV_S = m_F·√(  …same…  ) / (m_F + m_S)

    NOTE: the worked example in the reference deck (m_f=2000, m_s=2500, θ=30°,
    v_f=4.5, v_s=5.5 → ΔV=3.63) does NOT reproduce from this standard formula
    (which gives ≈1.53 m/s); the deck's arithmetic appears internally
    inconsistent. This implements the formula exactly as written in both source
    PDFs — reconcile the example with TTS before treating Delta-V as
    acceptance-grade.
    """
    rel_sq = v_f * v_f + v_s * v_s - 2.0 * v_f * v_s * np.cos(theta)
    rel = float(np.sqrt(max(rel_sq, 0.0)))
    total = m_f + m_s
    if total <= EPS:
        return 0.0
    dvf = m_s * rel / total
    dvs = m_f * rel / total
    return float(max(dvf, dvs))
