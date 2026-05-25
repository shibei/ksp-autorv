"""Orbital mechanics utility library for KSP kRPC autopilot.

Pure math functions using numpy. Zero kRPC dependency.
"""

import numpy as np

# ── Kerbin constants ──────────────────────────────────────────────────
MU_KERBIN = 3.5316e12  # gravitational parameter [m³/s²]
R_KERBIN = 600_000  # radius [m]
G0_KERBIN = 9.81  # standard gravity [m/s²]


# ── Kepler equation solvers ───────────────────────────────────────────


def kepler_epoch(M: float, e: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    """Solve Kepler's equation  M = E - e·sin(E)  for eccentric anomaly E.

    Newton-Raphson iteration starting from E = M.
    """
    E = float(M)
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def eccentric_anomaly_from_true(f: float, e: float) -> float:
    """Convert true anomaly to eccentric anomaly."""
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(f / 2.0), np.sqrt(1.0 + e) * np.cos(f / 2.0))
    return E


def true_anomaly_from_mean(M: float, e: float, tol: float = 1e-10) -> float:
    """Convert mean anomaly to true anomaly via Kepler's equation."""
    E = kepler_epoch(M, e, tol=tol)
    f = np.arctan2(np.sqrt(1.0 - e**2) * np.sin(E), np.cos(E) - e)
    return f


# ── Orbital element helpers ───────────────────────────────────────────


def orbital_period(sma: float, mu: float = MU_KERBIN) -> float:
    """Orbital period  T = 2π√(a³/μ)."""
    return 2.0 * np.pi * np.sqrt(sma**3 / mu)


def mean_motion(sma: float, mu: float = MU_KERBIN) -> float:
    """Mean motion  n = √(μ/a³)."""
    return np.sqrt(mu / sma**3)


def mean_anomaly_at_time(t: float, t0: float, M0: float, n: float) -> float:
    """Mean anomaly at time t:  M(t) = (M₀ + n·(t−t₀)) mod 2π."""
    return (M0 + n * (t - t0)) % (2.0 * np.pi)


# ── Position / velocity from elements ────────────────────────────────


def orbital_position(
    inc: float,
    lan: float,
    arg_pe: float,
    sma: float,
    ecc: float,
    true_anomaly: float,
    mu: float = MU_KERBIN,
):
    """Return (r, v) numpy arrays from classical orbital elements.

    Uses perifocal frame then rotates via 3-1-3 Euler angles
    (Ω, i, ω) → Rz(Ω)·Rx(i)·Rz(ω).
    """
    f = true_anomaly
    p = sma * (1.0 - ecc**2)
    r_mag = p / (1.0 + ecc * np.cos(f))
    h = np.sqrt(mu * p)

    # Perifocal frame
    r_pf = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0.0])
    v_pf = np.array([-mu / h * np.sin(f), mu / h * (ecc + np.cos(f)), 0.0])

    # Rotation matrix: Rz(Ω) · Rx(i) · Rz(ω)
    cos_O, sin_O = np.cos(lan), np.sin(lan)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_w, sin_w = np.cos(arg_pe), np.sin(arg_pe)

    R = np.array(
        [
            [
                cos_O * cos_w - sin_O * sin_w * cos_i,
                -cos_O * sin_w - sin_O * cos_w * cos_i,
                sin_O * sin_i,
            ],
            [
                sin_O * cos_w + cos_O * sin_w * cos_i,
                -sin_O * sin_w + cos_O * cos_w * cos_i,
                -cos_O * sin_i,
            ],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ]
    )

    return R @ r_pf, R @ v_pf


# ── Cartesian → Keplerian elements ───────────────────────────────────


def cartesian_to_kepler(r_vec, v_vec, mu: float = MU_KERBIN) -> dict:
    """Convert (r, v) to classical orbital elements dict.

    Returns dict with keys: inc, lan, arg_pe, sma, ecc, true_anomaly.
    Handles zero eccentricity and zero inclination edge cases.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    ecc = np.linalg.norm(e_vec)

    # Semi-major axis
    energy = v**2 / 2.0 - mu / r
    sma = -mu / (2.0 * energy)

    # Inclination
    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    # LAN (Ω)
    if n > 1e-12:
        lan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            lan = 2.0 * np.pi - lan
    else:
        lan = 0.0

    # Argument of periapsis (ω)
    if n > 1e-12 and ecc > 1e-12:
        arg_pe = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * ecc), -1.0, 1.0))
        if e_vec[2] < 0:
            arg_pe = 2.0 * np.pi - arg_pe
    elif ecc > 1e-12:
        # Zero inclination: measure from x-axis
        arg_pe = np.arctan2(e_vec[1], e_vec[0])
        if arg_pe < 0:
            arg_pe += 2.0 * np.pi
    else:
        arg_pe = 0.0

    # True anomaly (ν)
    if ecc > 1e-12:
        true_anomaly = np.arccos(np.clip(np.dot(e_vec, r_vec) / (ecc * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            true_anomaly = 2.0 * np.pi - true_anomaly
    else:
        # Circular orbit: measure from ascending node or x-axis
        if n > 1e-12:
            true_anomaly = np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1.0, 1.0))
            if r_vec[2] < 0:
                true_anomaly = 2.0 * np.pi - true_anomaly
        else:
            true_anomaly = np.arctan2(r_vec[1], r_vec[0])
            if true_anomaly < 0:
                true_anomaly += 2.0 * np.pi

    return {
        'inc': inc,
        'lan': lan,
        'arg_pe': arg_pe,
        'sma': sma,
        'ecc': ecc,
        'true_anomaly': true_anomaly,
    }


# ── Stumpff functions ────────────────────────────────────────────────


def _stumpff_C(z: float) -> float:
    """Stumpff function C(z)."""
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    else:
        return 0.5 - z / 24.0 + z**2 / 720.0


def _stumpff_S(z: float) -> float:
    """Stumpff function S(z)."""
    if z > 1e-6:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (z * sz)
    elif z < -1e-6:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / ((-z) * sz)
    else:
        return 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0


# ── Lambert solver (universal variable) ──────────────────────────────


def lambert_universal(
    r1_vec, r2_vec, dt: float, mu: float = MU_KERBIN, tol: float = 1e-8, max_iter: int = 200
):
    """Solve Lambert's problem using universal variable formulation.

    Returns (v1, v2) velocity vectors.
    """
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_dnu = np.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

    # Determine transfer angle (short way, < π)
    cross = np.cross(r1_vec, r2_vec)
    if cross[2] >= 0:
        dnu = np.arccos(cos_dnu)
    else:
        dnu = 2.0 * np.pi - np.arccos(cos_dnu)

    sin_dnu = np.sin(dnu)

    A = sin_dnu * np.sqrt(r1 * r2 / (1.0 - cos_dnu))

    # Initial guess for z
    z = 0.0

    for _ in range(max_iter):
        C = _stumpff_C(z)
        S = _stumpff_S(z)

        y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)

        if y < 0:
            # Adjust z upward
            z += 0.1
            continue

        x = np.sqrt(y / C)
        t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)

        # Derivative dt/dz
        if abs(z) > 1e-6:
            dt_dz = x**3 * (
                S
                - 3.0 * S * z / (2.0 * z)
                + 1.0 / (2.0 * z) * (C / (2.0 * z) - 3.0 * S / (2.0 * z * z))
            )
            # Simpler numerical derivative for robustness
            dz = 1e-7 * max(1.0, abs(z))
            C2 = _stumpff_C(z + dz)
            S2 = _stumpff_S(z + dz)
            y2 = r1 + r2 + A * ((z + dz) * S2 - 1.0) / np.sqrt(C2)
            if y2 < 0:
                z += 0.5
                continue
            x2 = np.sqrt(y2 / C2)
            t2 = (x2**3 * S2 + A * np.sqrt(y2)) / np.sqrt(mu)
            dt_dz = (t2 - t) / dz
        else:
            dz = 1e-7
            C2 = _stumpff_C(z + dz)
            S2 = _stumpff_S(z + dz)
            y2 = r1 + r2 + A * ((z + dz) * S2 - 1.0) / np.sqrt(C2)
            if y2 < 0:
                z += 0.5
                continue
            x2 = np.sqrt(y2 / C2)
            t2 = (x2**3 * S2 + A * np.sqrt(y2)) / np.sqrt(mu)
            dt_dz = (t2 - t) / dz

        if abs(dt_dz) < 1e-20:
            break

        z_new = z + (dt - t) / dt_dz

        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new

    # Final computation with converged z
    C = _stumpff_C(z)
    S = _stumpff_S(z)
    y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)

    f = 1.0 - y / r1
    g = A * np.sqrt(y / mu)
    g_dot = 1.0 - y / r2

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g

    return v1, v2


# ── Ascent helpers ────────────────────────────────────────────────────


def gravity_turn_pitch_profile(h: float, h_start: float, h_end: float, pitch_end: float) -> float:
    """Gravity turn pitch profile: vertical (90°) below h_start,
    quadratically interpolating to pitch_end at h_end so the pitch
    stays high during the early phase of the turn.

    Returns pitch angle in radians.
    """
    pitch_start = np.radians(90.0)
    if h <= h_start:
        return pitch_start
    if h >= h_end:
        return pitch_end
    frac = (h - h_start) / (h_end - h_start)
    return pitch_start + (pitch_end - pitch_start) * frac**2


def delta_v_estimate(isp, m0: float, mf: float, g0: float = G0_KERBIN) -> float:
    """Tsiolkovsky rocket equation: Δv = Isp·g₀·ln(m₀/m_f).

    If isp is a list, uses the average.
    """
    if isinstance(isp, (list, tuple)):
        isp = sum(isp) / len(isp)
    return float(isp * g0 * np.log(m0 / mf))


def ascent_duration_estimate(stages: list, target_altitude: float) -> float:
    """Estimate total ascent burn duration.

    stages: list of dicts with keys dry_mass, wet_mass, isp, thrust.
    Returns total burn time in seconds.
    """
    total = 0.0
    for stage in stages:
        m_prop = stage['wet_mass'] - stage['dry_mass']
        mdot = stage['thrust'] / (stage['isp'] * G0_KERBIN)
        total += m_prop / mdot
    return total
