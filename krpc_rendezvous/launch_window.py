"""Launch window calculator for KSP kRPC direct ascent rendezvous.

Finds optimal launch windows by searching for alignment between KSC
longitude and target orbital position, then estimates total Δv using
Lambert solver between ascent endpoint and target.

Phase 1: equatorial orbits only (inc ≤ 3°).
"""

import argparse
import json
import logging
import math
import sys

import numpy as np

from krpc_rendezvous.common.config import (
    KERBIN_ROTATION_RATE,
    MAX_INCLINATION_DEG,
    WINDOW_SEARCH_DAYS,
    WINDOW_SEARCH_STEP,
    WINDOW_SKIP_SECONDS,
    WINDOW_TOLERANCE_DEG,
)
from krpc_rendezvous.common.krpc_connection import connect_krpc
from krpc_rendezvous.common.orbit_utils import (
    MU_KERBIN,
    R_KERBIN,
    ascent_duration_estimate,
    delta_v_estimate,
    lambert_universal,
    mean_anomaly_at_time,
    mean_motion,
    orbital_position,
    true_anomaly_from_mean,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

MAX_INCLINATION_RAD = math.radians(MAX_INCLINATION_DEG)

# Pitch curve: (altitude_m, pitch_rad)
PITCH_CURVE = [
    (0, math.pi / 2),  # 90° vertical
    (4000, math.radians(85)),  # 85°
    (8000, math.radians(70)),  # 70°
    (20000, math.radians(35)),  # 35°
    (40000, math.radians(15)),  # 15°
    (70000, math.radians(5)),  # 5°
]


# ── Helpers ─────────────────────────────────────────────────────────────


def ksc_longitude(t: float) -> float:
    """KSC longitude at universal time *t*.

    Kerbin rotates at 2π/21600 rad/s. KSC starts at longitude 0 when UT=0.
    This is an ABSOLUTE computation in the same reference frame as target_longitude.
    """
    return (KERBIN_ROTATION_RATE * t) % (2.0 * math.pi)


def target_longitude(
    t: float, epoch_ut: float, lan: float, arg_pe: float, ecc: float, sma: float, M0: float
) -> float:
    """Target vessel longitude at universal time *t*.

    M0 is the mean anomaly at epoch_ut.
    For equatorial orbits: lon = Ω + ω + ν(t).
    """
    n = mean_motion(sma)
    M = mean_anomaly_at_time(t, epoch_ut, M0, n)
    nu = true_anomaly_from_mean(M, ecc)
    return (lan + arg_pe + nu) % (2.0 * math.pi)


def angular_distance(a: float, b: float) -> float:
    """Shortest angular distance between two angles [0, π]."""
    d = (a - b) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def find_launch_windows(
    now: float,
    lan: float,
    arg_pe: float,
    ecc: float,
    sma: float,
    M0: float,
    epoch_ut: float,
    ascent_time: float = 180.0,
    search_days: float = WINDOW_SEARCH_DAYS,
    step: float = WINDOW_SEARCH_STEP,
    tolerance_deg: float = WINDOW_TOLERANCE_DEG,
    skip_seconds: float = WINDOW_SKIP_SECONDS,
):
    """Search for launch windows over *search_days* starting from *now*.

    For each candidate launch time *t*, the rendezvous occurs at *t + ascent_time*.
    Window condition: target_longitude(t + ascent_time) ≈ ksc_longitude(t).

    Yields (window_ut, angular_error_rad) for each window found.
    After finding a window, skips *skip_seconds* before searching again.
    """
    tolerance_rad = math.radians(tolerance_deg)
    search_end = now + search_days * 86400.0
    t = now
    windows: list[tuple[float, float]] = []

    while t < search_end:
        ksc_lon = ksc_longitude(t)
        tgt_lon = target_longitude(t + ascent_time, epoch_ut, lan, arg_pe, ecc, sma, M0)
        err = angular_distance(ksc_lon, tgt_lon)

        # Debug: print first match details
        if err <= tolerance_rad and not windows:
            print(f'  [DEBUG] Window found at t={t:.1f}', file=sys.stderr)
            print(f'    ksc_lon(t)={math.degrees(ksc_lon):.2f}deg', file=sys.stderr)
            print(f'    tgt_lon(t+ascent)={math.degrees(tgt_lon):.2f}deg', file=sys.stderr)
            print(f'    err={math.degrees(err):.4f}deg', file=sys.stderr)
            print(f'    epoch_ut={epoch_ut:.1f}  now={now:.1f}', file=sys.stderr)
            print(f'    M0={M0:.4f}rad', file=sys.stderr)
            print(
                f'    lan={math.degrees(lan):.1f}deg arg_pe={math.degrees(arg_pe):.1f}deg',
                file=sys.stderr,
            )

        if err <= tolerance_rad:
            windows.append((t, err))
            t += skip_seconds
        else:
            t += step

    return windows


def compute_window_dv(
    window_ut: float,
    now: float,
    ascent_time: float,
    target_orbit: dict,
    ascent_altitude: float = 80000.0,
):
    """Compute total Δv for a launch window.

    Target position is computed at (window_ut + ascent_time) — the
    rendezvous occurs AFTER the ascent, not at launch.

    Returns (ascent_dv, rendezvous_dv, total_dv).
    """
    lan = target_orbit['lan']
    arg_pe = target_orbit['arg_pe']
    ecc = target_orbit['ecc']
    sma = target_orbit['sma']
    M0 = target_orbit['M0']
    epoch_ut = target_orbit['epoch']
    inc = target_orbit['inc']

    # Rendezvous time = launch + ascent
    arrival_ut = window_ut + ascent_time

    # Chaser position at arrival: near KSC longitude at arrival time
    ksc_lon_arrival = ksc_longitude(arrival_ut)
    ascent_r = R_KERBIN + ascent_altitude
    r1 = np.array([ascent_r * math.cos(ksc_lon_arrival), ascent_r * math.sin(ksc_lon_arrival), 0.0])

    # Target position at arrival time (NOT at launch time!)
    n = mean_motion(sma)
    M_arr = mean_anomaly_at_time(arrival_ut, epoch_ut, M0, n)
    nu_arr = true_anomaly_from_mean(M_arr, ecc)
    r2, v2_target = orbital_position(inc, lan, arg_pe, sma, ecc, nu_arr)

    # Lambert solve: ascent endpoint → target position
    # Transfer time: use a fixed ~600s (10 min) for low-orbit coast,
    # not ascent_time (82s) which is too short for orbital transfer
    transfer_time = max(300.0, ascent_time * 2)
    try:
        v1_lambert, v2_lambert = lambert_universal(r1, r2, transfer_time, MU_KERBIN)
    except Exception:
        logger.warning('Lambert solver failed for window at UT=%.1f', window_ut)
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
        print('ERROR: No target vessel set. Set a target in KSP first.', file=sys.stderr)
        sys.exit(1)

    orbit = target.orbit
    inc = orbit.inclination
    if inc > MAX_INCLINATION_RAD:
        print(
            f'ERROR: Target inclination {math.degrees(inc):.2f}° exceeds '
            f'{MAX_INCLINATION_DEG}° limit (equatorial only for Phase 1).',
            file=sys.stderr,
        )
        sys.exit(1)

    now = sc.ut
    sma = orbit.semi_major_axis
    ecc = orbit.eccentricity
    lan = orbit.longitude_of_ascending_node
    arg_pe = orbit.argument_of_periapsis

    # Mean anomaly at current time (direct from kRPC, most reliable)
    M0 = orbit.mean_anomaly

    return {
        'inc': inc,
        'lan': lan,
        'arg_pe': arg_pe,
        'sma': sma,
        'ecc': ecc,
        'M0': M0,
        'epoch': now,  # M0 is at this epoch
        'true_anomaly': orbit.true_anomaly,
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
    # LiquidFuel density: 1 unit = 5 kg
    fuel_mass = fuel * 5.0  # kg
    mf = m0 - fuel_mass
    if mf <= 0:
        mf = m0 * 0.3  # fallback

    # Engine info: aggregate from all engines
    # Use max_thrust (available even when inactive) and vacuum Isp (for Δv estimation)
    parts = vessel.parts
    engines = parts.engines
    if engines:
        isp_list = [e.vacuum_specific_impulse for e in engines]
        thrust_list = [e.max_thrust for e in engines]
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
    lines.append(
        f'{"#":>3}  {"Launch UT":>14}  {"Ascent Δv":>10}  '
        f'{"Rendez Δv":>10}  {"Total Δv":>10}  {"Available":>10}'
    )
    lines.append('-' * 70)

    for i, w in enumerate(windows_data, 1):
        lines.append(
            f'{i:3d}  {w["launch_ut"]:14.1f}  '
            f'{w["ascent_dv"]:10.1f}  {w["rendezvous_dv"]:10.1f}  '
            f'{w["total_dv"]:10.1f}  {w["available_dv"]:10.1f}'
        )

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Launch window calculator for KSP kRPC direct ascent rendezvous'
    )
    parser.add_argument(
        '--search-days',
        type=float,
        default=WINDOW_SEARCH_DAYS,
        help=f'Number of days to search (default: {WINDOW_SEARCH_DAYS})',
    )
    parser.add_argument(
        '--step',
        type=float,
        default=WINDOW_SEARCH_STEP,
        help=f'Search step in seconds (default: {WINDOW_SEARCH_STEP})',
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=WINDOW_TOLERANCE_DEG,
        help=f'Window tolerance in degrees (default: {WINDOW_TOLERANCE_DEG})',
    )
    parser.add_argument('--max-dv', type=float, default=None, help='Maximum total Δv filter (m/s)')
    parser.add_argument(
        '--window', type=int, default=None, help='Select Nth window (1-indexed) by sort order'
    )
    parser.add_argument('--latest', action='store_true', help='Select the latest window')
    parser.add_argument(
        '--skip-hours',
        type=float,
        default=5.0,
        help='Hours to skip after finding a window (default: 5)',
    )
    parser.add_argument(
        '--dv',
        type=float,
        default=None,
        help='Total vacuum Δv of rocket (m/s). If set, overrides auto-calc',
    )
    parser.add_argument(
        '--ascent-alt',
        type=float,
        default=80000.0,
        help='Target ascent altitude in meters (default: 80000)',
    )
    parser.add_argument(
        '--output', type=str, default='launch_window_output.json', help='Output JSON file path'
    )
    args = parser.parse_args()

    # ── Connect and read data ───────────────────────────────────────────
    print('Connecting to kRPC...')
    conn = connect_krpc()

    print('Reading target orbit...')
    target_orbit = read_target_orbit(conn)

    print('Reading vessel resources...')
    vessel_info = read_vessel_resources(conn)

    now = conn.space_center.ut

    # ── Compute available Δv ────────────────────────────────────────────
    if args.dv is not None:
        available_dv = args.dv
        print(f'Using user-specified Δv: {available_dv:.0f} m/s')
    else:
        available_dv = compute_available_dv(vessel_info)
        print(f'Computed Δv: {available_dv:.0f} m/s (single-stage estimate)')
        print('  (for multi-stage rockets, use --dv YOUR_VALUE)')

    # ── Estimate ascent time ────────────────────────────────────────────
    stages = [
        {
            'dry_mass': vessel_info['dry_mass'],
            'wet_mass': vessel_info['wet_mass'],
            'isp': vessel_info['isp'],
            'thrust': vessel_info['thrust'],
        }
    ]
    # ascent_duration_estimate only sees current stage fuel on multi-stage rockets.
    # Use empirical value: ~180s for Kerbin LKO ascent (overridable)
    computed_ascent = ascent_duration_estimate(stages, args.ascent_alt)
    ascent_time = max(computed_ascent, 140.0)
    print(f'Computed ascent time: {computed_ascent:.0f}s, using: {ascent_time:.0f}s')

    # ── Find launch windows ──────────────────────────────────────────────
    print(f'Searching for launch windows over {args.search_days} day(s)...')
    skip_seconds = args.skip_hours * 3600.0

    raw_windows = find_launch_windows(
        now=now,
        lan=target_orbit['lan'],
        arg_pe=target_orbit['arg_pe'],
        ecc=target_orbit['ecc'],
        sma=target_orbit['sma'],
        M0=target_orbit['M0'],
        epoch_ut=target_orbit['epoch'],
        ascent_time=ascent_time,
        search_days=args.search_days,
        step=args.step,
        tolerance_deg=args.tolerance,
        skip_seconds=skip_seconds,
    )

    if not raw_windows:
        print('No launch windows found in the search period.', file=sys.stderr)
        sys.exit(1)

    # ── Compute Δv for each window ──────────────────────────────────────
    windows_data = []
    for window_ut, err in raw_windows:
        ascent_dv, rendezvous_dv, total_dv = compute_window_dv(
            window_ut, now, ascent_time, target_orbit, args.ascent_alt
        )
        if total_dv is None:
            continue

        windows_data.append(
            {
                'launch_ut': window_ut,
                'window_ut': window_ut,
                'ascent_time': ascent_time,
                'ascent_dv': ascent_dv,
                'rendezvous_dv': rendezvous_dv,
                'total_dv': total_dv,
                'available_dv': available_dv,
                'angular_error_deg': math.degrees(err),
            }
        )

    if not windows_data:
        print('No valid launch windows after Δv computation.', file=sys.stderr)
        sys.exit(1)

    # ── Sort by total Δv ────────────────────────────────────────────────
    windows_data.sort(key=lambda w: w['total_dv'])

    # ── Filter by max Δv ────────────────────────────────────────────────
    if args.max_dv is not None:
        windows_data = [w for w in windows_data if w['total_dv'] <= args.max_dv]

    if not windows_data:
        print(f'No windows within --max-dv={args.max_dv:.1f} m/s.', file=sys.stderr)
        sys.exit(1)

    # ── Select window ────────────────────────────────────────────────────
    if args.latest:
        selected = windows_data[-1]
    elif args.window is not None:
        idx = args.window - 1  # 1-indexed
        if idx < 0 or idx >= len(windows_data):
            print(
                f'Window index {args.window} out of range (1-{len(windows_data)}).', file=sys.stderr
            )
            sys.exit(1)
        selected = windows_data[idx]
    else:
        windows_data.sort(key=lambda w: w['launch_ut'])
        selected = windows_data[0]

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
    print(f'\nOutput written to {args.output}')

    # ── Print table ──────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print(f'LAUNCH WINDOW RESULTS ({len(windows_data)} windows found)')
    print(f'{"=" * 70}')
    print(format_table(windows_data))
    print(f'\nSelected window: #{windows_data.index(selected) + 1}')
    print(f'  Launch UT:       {selected["launch_ut"]:.1f}')
    print(f'  Ascent time:      {selected["ascent_time"]:.1f} s')
    print(f'  Ascent Δv:        {selected["ascent_dv"]:.1f} m/s')
    print(f'  Rendezvous Δv:    {selected["rendezvous_dv"]:.1f} m/s')
    print(f'  Total Δv:         {selected["total_dv"]:.1f} m/s')
    print(f'  Available Δv:     {selected["available_dv"]:.1f} m/s')
    print(f'  Angular error:    {selected["angular_error_deg"]:.2f}°')


if __name__ == '__main__':
    main()
