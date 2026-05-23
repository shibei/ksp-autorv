"""Launch window calculator for KSP kRPC direct ascent rendezvous.

Finds optimal launch windows by searching for alignment between KSC
longitude and target orbital position, then estimates total Δv using
Lambert solver between ascent endpoint and target.

Phase 1: equatorial orbits only (inc ≤ 3°).
"""

import argparse
import json
import math
import sys
import logging

import numpy as np

from common.orbit_utils import (
    kepler_epoch,
    orbital_period,
    mean_motion,
    mean_anomaly_at_time,
    true_anomaly_from_mean,
    orbital_position,
    lambert_universal,
    delta_v_estimate,
    ascent_duration_estimate,
    MU_KERBIN,
    R_KERBIN,
    G0_KERBIN,
)
from common.krpc_connection import connect_krpc, get_connection

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

KSC_LON_RAD = 0.0  # KSC at ~0° longitude (radians)
KERBIN_ROTATION_PERIOD = 21600.0  # 6 h sidereal day [s]
KERBIN_ROTATION_RATE = 2.0 * math.pi / KERBIN_ROTATION_PERIOD  # rad/s

# Pitch curve: (altitude_m, pitch_rad)
PITCH_CURVE = [
    (0, math.pi / 2),          # 90° vertical
    (4000, math.radians(85)),   # 85°
    (8000, math.radians(70)),   # 70°
    (20000, math.radians(35)),  # 35°
    (40000, math.radians(15)), # 15°
    (70000, math.radians(5)),  # 5°
]

MAX_INCLINATION_DEG = 3.0
MAX_INCLINATION_RAD = math.radians(MAX_INCLINATION_DEG)
DEFAULT_WINDOW_TOLERANCE_DEG = 2.0
DEFAULT_SKIP_SECONDS = 5 * 3600  # ~5 h
DEFAULT_STEP_SECONDS = 30
DEFAULT_SEARCH_DAYS = 1


# ── Helpers ─────────────────────────────────────────────────────────────

def ksc_longitude(t: float, now: float) -> float:
    """KSC longitude at universal time *t*, given current time *now*.

    KSC is at longitude ≈ 0° at t=now; Kerbin rotates at 2π/21600 rad/s.
    """
    return (KSC_LON_RAD + KERBIN_ROTATION_RATE * (t - now)) % (2.0 * math.pi)


def target_longitude(t: float, now: float, lan: float, arg_pe: float,
                     ecc: float, sma: float, M0: float) -> float:
    """Target vessel longitude at universal time *t*.

    For equatorial orbits: lon = Ω + ω + ν(t).
    """
    n = mean_motion(sma)
    M = mean_anomaly_at_time(t, now, M0, n)
    nu = true_anomaly_from_mean(M, ecc)
    return (lan + arg_pe + nu) % (2.0 * math.pi)


def angular_distance(a: float, b: float) -> float:
    """Shortest angular distance between two angles [0, π]."""
    d = (a - b) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def find_launch_windows(now: float, lan: float, arg_pe: float,
                        ecc: float, sma: float, M0: float,
                        search_days: float = DEFAULT_SEARCH_DAYS,
                        step: float = DEFAULT_STEP_SECONDS,
                        tolerance_deg: float = DEFAULT_WINDOW_TOLERANCE_DEG,
                        skip_seconds: float = DEFAULT_SKIP_SECONDS):
    """Search for launch windows over *search_days* starting from *now*.

    Yields (window_ut, angular_error_rad) for each window found.
    After finding a window, skips *skip_seconds* before searching again.
    """
    tolerance_rad = math.radians(tolerance_deg)
    search_end = now + search_days * 86400.0
    t = now
    windows = []

    while t < search_end:
        ksc_lon = ksc_longitude(t, now)
        tgt_lon = target_longitude(t, now, lan, arg_pe, ecc, sma, M0)
        err = angular_distance(ksc_lon, tgt_lon)

        if err <= tolerance_rad:
            windows.append((t, err))
            t += skip_seconds  # skip to next window opportunity
        else:
            t += step

    return windows


def compute_window_dv(window_ut: float, now: float, ascent_time: float,
                      target_orbit: dict, ascent_altitude: float = 80000.0):
    """Compute total Δv for a launch window.

    Returns (ascent_dv, rendezvous_dv, total_dv).
    """
    lan = target_orbit['lan']
    arg_pe = target_orbit['arg_pe']
    ecc = target_orbit['ecc']
    sma = target_orbit['sma']
    M0 = target_orbit['M0']
    inc = target_orbit['inc']

    # Ascent endpoint: position on Kerbin surface at KSC, velocity = 0 (relative)
    # After ascent, vessel is at altitude ascent_altitude above KSC longitude
    ksc_lon = ksc_longitude(window_ut, now)
    ascent_r = R_KERBIN + ascent_altitude

    # Ascent endpoint position (equatorial plane, KSC longitude)
    r1 = np.array([ascent_r * math.cos(ksc_lon),
                    ascent_r * math.sin(ksc_lon),
                    0.0])

    # Target position at arrival time
    arrival_ut = window_ut + ascent_time
    n = mean_motion(sma)
    M_arr = mean_anomaly_at_time(arrival_ut, now, M0, n)
    nu_arr = true_anomaly_from_mean(M_arr, ecc)
    r2, v2_target = orbital_position(inc, lan, arg_pe, sma, ecc, nu_arr)

    # Lambert solve: ascent endpoint → target position
    try:
        v1_lambert, v2_lambert = lambert_universal(r1, r2, ascent_time, MU_KERBIN)
    except Exception:
        logger.warning("Lambert solver failed for window at UT=%.1f", window_ut)
        return None, None, None

    # Circular orbit velocity at ascent altitude
    v_circ = math.sqrt(MU_KERBIN / ascent_r)

    # Ascent Δv: approximate as circularization from surface to ascent altitude
    # plus the Lambert Δv from circular orbit to transfer
    v1_mag = np.linalg.norm(v1_lambert)
    rendezvous_dv = abs(v1_mag - v_circ)

    # Ascent Δv: gravity losses + circularization (rough estimate)
    # Use Tsiolkovsky with actual vessel parameters later
    ascent_dv = v_circ  # rough: need to reach orbital velocity

    total_dv = ascent_dv + rendezvous_dv

    return ascent_dv, rendezvous_dv, total_dv


def read_target_orbit(conn):
    """Read target vessel orbit from kRPC connection.

    Returns dict with orbital elements plus M0 at current UT.
    Raises SystemExit if no target or inclination too high.
    """
    sc = conn.space_center
    target = sc.target_vessel

    if target is None:
        print("ERROR: No target vessel set. Set a target in KSP first.", file=sys.stderr)
        sys.exit(1)

    orbit = target.orbit
    inc = orbit.inclination
    if inc > MAX_INCLINATION_RAD:
        print(f"ERROR: Target inclination {math.degrees(inc):.2f}° exceeds "
              f"{MAX_INCLINATION_DEG}° limit (equatorial only for Phase 1).",
              file=sys.stderr)
        sys.exit(1)

    now = sc.ut
    sma = orbit.semi_major_axis
    ecc = orbit.eccentricity
    lan = orbit.longitude_of_ascending_node
    arg_pe = orbit.argument_of_periapsis

    # Compute mean anomaly at current time
    n = mean_motion(sma)
    true_anom = orbit.true_anomaly
    # Convert true anomaly to mean anomaly
    E = kepler_epoch(true_anom, ecc)  # approximate: use true anomaly as M for small e
    # Actually: E from true anomaly, then M from E
    # true_anomaly_from_mean gives f from M; we need M from f
    # M = E - e*sin(E), and E from true anomaly
    E_from_f = 2.0 * math.atan2(math.sqrt(1 - ecc) * math.sin(true_anom / 2),
                                  math.sqrt(1 + ecc) * math.cos(true_anom / 2))
    M0 = (E_from_f - ecc * math.sin(E_from_f)) % (2.0 * math.pi)

    return {
        'inc': inc,
        'lan': lan,
        'arg_pe': arg_pe,
        'sma': sma,
        'ecc': ecc,
        'M0': M0,
        'true_anomaly': true_anom,
    }


def read_vessel_resources(conn):
    """Read active vessel mass, fuel, engine Isp and thrust.

    Returns dict with vessel parameters for Δv estimation.
    """
    vessel = conn.space_center.active_vessel
    resources = vessel.resources

    # Total mass
    m0 = vessel.mass

    # Fuel (liquid fuel)
    fuel = resources.amount('LiquidFuel')

    # Dry mass estimate: current mass minus fuel mass
    # LiquidFuel density ≈ 5 kg/unit in KSP
    fuel_mass = fuel * 5.0 / 1000.0  # tonnes
    mf = m0 - fuel_mass
    if mf <= 0:
        mf = m0 * 0.3  # fallback

    # Engine info: aggregate from all engines
    parts = vessel.parts
    engines = parts.engines
    if engines:
        isp_list = [e.specific_impulse for e in engines]
        thrust_list = [e.thrust for e in engines]
        avg_isp = sum(isp_list) / len(isp_list)
        total_thrust = sum(thrust_list)
    else:
        avg_isp = 350.0  # reasonable default
        total_thrust = 200000.0

    return {
        'wet_mass': m0,
        'dry_mass': mf,
        'isp': avg_isp,
        'thrust': total_thrust,
        'fuel': fuel,
    }


def compute_available_dv(vessel_info: dict) -> float:
    """Compute available Δv from vessel info using Tsiolkovsky equation."""
    return delta_v_estimate(
        vessel_info['isp'],
        vessel_info['wet_mass'],
        vessel_info['dry_mass'],
    )


def format_table(windows_data: list) -> str:
    """Format window results as a readable table."""
    lines = []
    lines.append(f"{'#':>3}  {'Launch UT':>14}  {'Ascent Δv':>10}  "
                 f"{'Rendez Δv':>10}  {'Total Δv':>10}  {'Available':>10}")
    lines.append("-" * 70)

    for i, w in enumerate(windows_data, 1):
        lines.append(
            f"{i:3d}  {w['launch_ut']:14.1f}  "
            f"{w['ascent_dv']:10.1f}  {w['rendezvous_dv']:10.1f}  "
            f"{w['total_dv']:10.1f}  {w['available_dv']:10.1f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Launch window calculator for KSP kRPC direct ascent rendezvous"
    )
    parser.add_argument("--search-days", type=float, default=DEFAULT_SEARCH_DAYS,
                        help=f"Number of days to search (default: {DEFAULT_SEARCH_DAYS})")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_SECONDS,
                        help=f"Search step in seconds (default: {DEFAULT_STEP_SECONDS})")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_WINDOW_TOLERANCE_DEG,
                        help=f"Window tolerance in degrees (default: {DEFAULT_WINDOW_TOLERANCE_DEG})")
    parser.add_argument("--max-dv", type=float, default=None,
                        help="Maximum total Δv filter (m/s)")
    parser.add_argument("--window", type=int, default=None,
                        help="Select Nth window (1-indexed) by sort order")
    parser.add_argument("--latest", action="store_true",
                        help="Select the latest window")
    parser.add_argument("--skip-hours", type=float, default=5.0,
                        help="Hours to skip after finding a window (default: 5)")
    parser.add_argument("--ascent-alt", type=float, default=80000.0,
                        help="Target ascent altitude in meters (default: 80000)")
    parser.add_argument("--output", type=str, default="launch_window_output.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # ── Connect and read data ───────────────────────────────────────────
    print("Connecting to kRPC...")
    conn = connect_krpc()

    print("Reading target orbit...")
    target_orbit = read_target_orbit(conn)

    print("Reading vessel resources...")
    vessel_info = read_vessel_resources(conn)

    now = conn.space_center.ut

    # ── Compute available Δv ────────────────────────────────────────────
    available_dv = compute_available_dv(vessel_info)

    # ── Estimate ascent time ────────────────────────────────────────────
    stages = [{
        'dry_mass': vessel_info['dry_mass'],
        'wet_mass': vessel_info['wet_mass'],
        'isp': vessel_info['isp'],
        'thrust': vessel_info['thrust'],
    }]
    ascent_time = ascent_duration_estimate(stages, args.ascent_alt)

    # ── Find launch windows ──────────────────────────────────────────────
    print(f"Searching for launch windows over {args.search_days} day(s)...")
    skip_seconds = args.skip_hours * 3600.0

    raw_windows = find_launch_windows(
        now=now,
        lan=target_orbit['lan'],
        arg_pe=target_orbit['arg_pe'],
        ecc=target_orbit['ecc'],
        sma=target_orbit['sma'],
        M0=target_orbit['M0'],
        search_days=args.search_days,
        step=args.step,
        tolerance_deg=args.tolerance,
        skip_seconds=skip_seconds,
    )

    if not raw_windows:
        print("No launch windows found in the search period.", file=sys.stderr)
        sys.exit(1)

    # ── Compute Δv for each window ──────────────────────────────────────
    windows_data = []
    for window_ut, err in raw_windows:
        ascent_dv, rendezvous_dv, total_dv = compute_window_dv(
            window_ut, now, ascent_time, target_orbit, args.ascent_alt
        )
        if total_dv is None:
            continue

        windows_data.append({
            'launch_ut': window_ut,
            'window_ut': window_ut,
            'ascent_time': ascent_time,
            'ascent_dv': ascent_dv,
            'rendezvous_dv': rendezvous_dv,
            'total_dv': total_dv,
            'available_dv': available_dv,
            'angular_error_deg': math.degrees(err),
        })

    if not windows_data:
        print("No valid launch windows after Δv computation.", file=sys.stderr)
        sys.exit(1)

    # ── Sort by total Δv ────────────────────────────────────────────────
    windows_data.sort(key=lambda w: w['total_dv'])

    # ── Filter by max Δv ────────────────────────────────────────────────
    if args.max_dv is not None:
        windows_data = [w for w in windows_data if w['total_dv'] <= args.max_dv]

    if not windows_data:
        print(f"No windows within --max-dv={args.max_dv:.1f} m/s.", file=sys.stderr)
        sys.exit(1)

    # ── Select window ────────────────────────────────────────────────────
    if args.latest:
        selected = windows_data[-1]
    elif args.window is not None:
        idx = args.window - 1  # 1-indexed
        if idx < 0 or idx >= len(windows_data):
            print(f"Window index {args.window} out of range "
                  f"(1-{len(windows_data)}).", file=sys.stderr)
            sys.exit(1)
        selected = windows_data[idx]
    else:
        selected = windows_data[0]  # lowest Δv

    # ── Build output ─────────────────────────────────────────────────────
    output = {
        'launch_ut': selected['launch_ut'],
        'window_ut': selected['window_ut'],
        'ascent_time': selected['ascent_time'],
        'ascent_dv': selected['ascent_dv'],
        'rendezvous_dv': selected['rendezvous_dv'],
        'total_dv': selected['total_dv'],
        'available_dv': selected['available_dv'],
        'target_orbit': {
            'inc': target_orbit['inc'],
            'lan': target_orbit['lan'],
            'arg_pe': target_orbit['arg_pe'],
            'sma': target_orbit['sma'],
            'ecc': target_orbit['ecc'],
        },
        'pitch_curve': [(alt, pitch) for alt, pitch in PITCH_CURVE],
        'launch_azimuth': math.pi / 2,  # equatorial: due east
    }

    # ── Write JSON ───────────────────────────────────────────────────────
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to {args.output}")

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LAUNCH WINDOW RESULTS ({len(windows_data)} windows found)")
    print(f"{'='*70}")
    print(format_table(windows_data))
    print(f"\nSelected window: #{windows_data.index(selected) + 1}")
    print(f"  Launch UT:       {selected['launch_ut']:.1f}")
    print(f"  Ascent time:      {selected['ascent_time']:.1f} s")
    print(f"  Ascent Δv:        {selected['ascent_dv']:.1f} m/s")
    print(f"  Rendezvous Δv:    {selected['rendezvous_dv']:.1f} m/s")
    print(f"  Total Δv:         {selected['total_dv']:.1f} m/s")
    print(f"  Available Δv:     {selected['available_dv']:.1f} m/s")
    print(f"  Angular error:    {selected['angular_error_deg']:.2f}°")


if __name__ == '__main__':
    main()