"""Orbital rendezvous guidance for KSP kRPC.

Reads ascent_state.json (from atmospheric_ascent.py), performs iterative
Lambert transfers + terminal proportional navigation to rendezvous with
the target vessel.

Phases:
  1. Lambert phase: iterative Lambert solves to close distance
  2. Terminal phase: proportional navigation for final approach (<5 km)
  3. Success when dist < 200 m and relative velocity < 10 m/s
  4. Hohmann fallback if total delta-v budget exceeded
"""

import argparse
import json
import logging
import math
import sys
import time

import numpy as np

from krpc_rendezvous.common.config import (
    DV_BUDGET_DEFAULT,
    MANEUVER_WAIT,
    MAX_RENDEZVOUS_ITERATIONS,
    RENDEZVOUS_DIST_THRESHOLD,
    RENDEZVOUS_VEL_THRESHOLD,
    TERMINAL_DIST_THRESHOLD,
)
from krpc_rendezvous.common.dashboard import Dashboard
from krpc_rendezvous.common.krpc_connection import connect_krpc
from krpc_rendezvous.common.orbit_utils import (
    MU_KERBIN,
    lambert_universal,
    orbital_period,
)

logger = logging.getLogger(__name__)


# ── Proportional navigation ───────────────────────────────────────────


def proportional_navigation(vessel, target_vessel, nav_constant=3.0):
    """Compute proportional-navigation guidance commands.

    Uses line-of-sight (LOS) rate to generate acceleration commands:
        a_cmd = N * |v_rel| * (LOS_rate × LOS_unit)

    Returns (pitch, heading) for the autopilot in the orbital reference frame.
    """
    # Positions in the body-centered reference frame
    r_chaser = np.array(vessel.position(vessel.orbit.body.reference_frame))
    r_target = np.array(target_vessel.position(vessel.orbit.body.reference_frame))

    # Velocities in the body-centered reference frame
    v_chaser = np.array(vessel.velocity(vessel.orbit.body.reference_frame))
    v_target = np.array(target_vessel.velocity(vessel.orbit.body.reference_frame))

    # Relative position and velocity
    r_rel = r_target - r_chaser
    v_rel = v_target - v_chaser

    r_mag = np.linalg.norm(r_rel)
    if r_mag < 1.0:
        return 0.0, 0.0

    # Line-of-sight unit vector
    los = r_rel / r_mag

    # LOS rate: omega = (r_rel × v_rel) / |r_rel|²
    omega = np.cross(r_rel, v_rel) / (r_mag**2)

    # Proportional navigation command
    v_rel_mag = np.linalg.norm(v_rel)
    a_cmd = nav_constant * v_rel_mag * np.cross(omega, los)

    # Convert acceleration direction to pitch and heading
    # In orbital reference frame: pitch = elevation angle, heading = azimuth
    a_mag = np.linalg.norm(a_cmd)
    if a_mag < 1e-6:
        return 0.0, 0.0

    a_dir = a_cmd / a_mag

    # Pitch: angle from local horizontal plane (positive up)
    horizontal = math.sqrt(a_dir[0] ** 2 + a_dir[2] ** 2)
    pitch = math.degrees(math.atan2(a_dir[1], horizontal))

    # Heading: angle in horizontal plane from north (z-axis)
    heading = math.degrees(math.atan2(a_dir[0], a_dir[2]))

    return pitch, heading


# ── Helper functions ──────────────────────────────────────────────────


def get_vessel_state(vessel, body_rf):
    """Get position and velocity as numpy arrays in body reference frame."""
    pos = np.array(vessel.position(body_rf))
    vel = np.array(vessel.velocity(body_rf))
    return pos, vel


def compute_phase_angle(r_chaser, r_target):
    """Compute phase angle between chaser and target positions.

    Returns angle in degrees [0, 360).
    """
    # Project onto equatorial plane for phase angle
    angle_c = math.atan2(r_chaser[1], r_chaser[0])
    angle_t = math.atan2(r_target[1], r_target[0])
    phase = math.degrees(angle_t - angle_c) % 360.0
    return phase


def estimate_transfer_time(r_chaser, r_target, target_period):
    """Estimate transfer time based on phase angle and target period.

    For a phase angle θ, the transfer time is approximately:
        dt ≈ (θ / 360) * T_target
    but we want a shorter transfer, so use a fraction.
    """
    phase = compute_phase_angle(r_chaser, r_target)
    # Normalize phase to [0, 360)
    if phase < 0:
        phase += 360.0

    # For small phase angles, use a minimum transfer time
    # Transfer time ≈ phase_fraction * period / 2 (half-orbit transfer)
    phase_fraction = phase / 360.0
    dt = phase_fraction * target_period * 0.5

    # Minimum transfer time of 60 seconds
    dt = max(dt, 60.0)

    return dt


def execute_burn(vessel, dv_vector, body_rf):
    """Orient vessel and execute a burn for the given delta-v vector.

    Returns the magnitude of delta-v applied.
    """
    dv_mag = np.linalg.norm(dv_vector)
    if dv_mag < 0.01:
        return 0.0

    # Convert dv direction to pitch and heading for autopilot
    dv_dir = dv_vector / dv_mag

    # Pitch: angle from horizontal plane
    horizontal = math.sqrt(dv_dir[0] ** 2 + dv_dir[2] ** 2)
    pitch = math.degrees(math.atan2(dv_dir[1], horizontal))

    # Heading: angle in horizontal plane
    heading = math.degrees(math.atan2(dv_dir[0], dv_dir[2]))

    # Orient vessel
    ap = vessel.auto_pilot
    ap.reference_frame = vessel.orbital_reference_frame
    ap.target_pitch_and_heading(pitch, heading)
    ap.engage()
    time.sleep(1.0)  # Wait for orientation

    # Estimate burn duration
    thrust = 0.0
    for engine in vessel.parts.engines:
        thrust += engine.thrust
    if thrust <= 0:
        thrust = 200000.0  # fallback

    mass = vessel.mass
    burn_duration = dv_mag / (thrust / mass)
    burn_duration = min(burn_duration, 5.0)  # cap at 5 seconds

    # Execute burn
    vessel.control.throttle = 1.0
    time.sleep(burn_duration)
    vessel.control.throttle = 0.0
    ap.disengage()

    return dv_mag


def hohmann_fallback(vessel, body_rf):
    """Hohmann fallback: circularize at current altitude, then wait for phasing.

    Returns True if circularization burn was executed.
    """
    pos, vel = get_vessel_state(vessel, body_rf)
    r_mag = np.linalg.norm(pos)
    v_mag = np.linalg.norm(vel)

    # Circular orbit velocity at current radius
    v_circ = math.sqrt(MU_KERBIN / r_mag)

    # Delta-v to circularize
    dv_needed = abs(v_circ - v_mag)

    if dv_needed < 1.0:
        logger.info('Already in near-circular orbit, skipping circularization')
        return False

    # Direction: prograde or retrograde
    v_dir = vel / v_mag
    if v_mag < v_circ:
        dv_vector = v_dir * dv_needed  # prograde
    else:
        dv_vector = -v_dir * dv_needed  # retrograde

    logger.info(f'Hohmann fallback: circularization burn Δv={dv_needed:.1f} m/s')
    execute_burn(vessel, dv_vector, body_rf)
    return True


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Orbital rendezvous guidance for KSP kRPC')
    parser.add_argument(
        '--state-file',
        type=str,
        default='ascent_state.json',
        help='Path to ascent state JSON (default: ascent_state.json)',
    )
    parser.add_argument(
        '--dv-budget',
        type=float,
        default=None,
        help='Maximum delta-v budget in m/s (default: auto-detect)',
    )
    args = parser.parse_args()

    # ── Load ascent state ─────────────────────────────────────────────
    try:
        with open(args.state_file, 'r') as f:
            json.load(f)
    except FileNotFoundError:
        print(f"ERROR: State file '{args.state_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in '{args.state_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in '{args.state_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # ── Connect to kRPC ────────────────────────────────────────────────
    print('Connecting to kRPC...')
    conn = connect_krpc()
    sc = conn.space_center
    vessel = sc.active_vessel

    # ── Verify target vessel ───────────────────────────────────────────
    target_vessel = sc.target_vessel
    if target_vessel is None:
        print('ERROR: No target vessel set. Set a target in KSP first.', file=sys.stderr)
        sys.exit(1)

    print(f'Chaser: {vessel.name}')
    print(f'Target: {target_vessel.name}')

    # ── Determine delta-v budget ───────────────────────────────────────
    if args.dv_budget is not None:
        dv_budget = args.dv_budget
    else:
        # Estimate from remaining fuel
        resources = vessel.resources
        fuel = resources.amount('LiquidFuel')
        m0 = vessel.mass
        fuel_mass = fuel * 5.0 / 1000.0  # tonnes
        mf = max(m0 - fuel_mass, m0 * 0.3)

        # Aggregate engine Isp
        engines = vessel.parts.engines
        if engines:
            avg_isp = sum(e.specific_impulse for e in engines) / len(engines)
        else:
            avg_isp = 350.0

        dv_budget = avg_isp * 9.81 * math.log(m0 / mf)
        if dv_budget < DV_BUDGET_DEFAULT:
            dv_budget = DV_BUDGET_DEFAULT

    print(f'Delta-v budget: {dv_budget:.1f} m/s')

    # ── Setup ──────────────────────────────────────────────────────────
    body_rf = vessel.orbit.body.reference_frame
    target_orbit = target_vessel.orbit
    target_period = orbital_period(target_orbit.semi_major_axis)

    # Dashboard
    columns = [
        ('DIST', 'km', 2),
        ('dVEL', 'm/s', 1),
        ('DV', 'm/s', 0),
        ('PHASE', 'deg', 1),
        ('NEXT', 's', 0),
    ]
    dashboard = Dashboard(columns=columns)
    dashboard.start()

    total_dv_spent = 0.0
    iteration = 0
    status = 'ACTIVE'

    # ── Main loop ─────────────────────────────────────────────────────
    try:
        while iteration < MAX_RENDEZVOUS_ITERATIONS:
            iteration += 1

            # Read current state
            r_chaser, v_chaser = get_vessel_state(vessel, body_rf)
            r_target, v_target = get_vessel_state(target_vessel, body_rf)

            # Compute distance and relative velocity
            r_rel = r_target - r_chaser
            v_rel = v_target - v_chaser
            dist = np.linalg.norm(r_rel)
            dv_rel = np.linalg.norm(v_rel)

            # Phase angle
            phase = compute_phase_angle(r_chaser, r_target)

            # ── Check success ──────────────────────────────────────────
            if dist < RENDEZVOUS_DIST_THRESHOLD and dv_rel < RENDEZVOUS_VEL_THRESHOLD:
                status = 'SUCCESS'
                print(f'\n✅ RENDEZVOUS SUCCESS at iteration {iteration}')
                print(f'   Distance: {dist:.1f} m')
                print(f'   Relative velocity: {dv_rel:.2f} m/s')
                print(f'   Total Δv spent: {total_dv_spent:.1f} m/s')
                break

            # ── Check budget ───────────────────────────────────────────
            if total_dv_spent >= dv_budget:
                print(f'\n⚠️  Delta-v budget exceeded ({total_dv_spent:.1f}/{dv_budget:.1f} m/s)')
                print('   Switching to Hohmann fallback...')
                hohmann_fallback(vessel, body_rf)
                status = 'DEGRADED'
                break

            # ── Terminal phase: proportional navigation ────────────────
            if dist < TERMINAL_DIST_THRESHOLD:
                print(f'\n  Terminal phase: dist={dist:.0f} m, switching to prop-nav')

                pitch, heading = proportional_navigation(vessel, target_vessel)

                ap = vessel.auto_pilot
                ap.reference_frame = vessel.orbital_reference_frame
                ap.target_pitch_and_heading(pitch, heading)
                ap.engage()

                # Small RCS-like burn
                vessel.control.throttle = 0.1
                time.sleep(2.0)
                vessel.control.throttle = 0.0
                ap.disengage()

                # Estimate small delta-v for RCS burst
                rcs_dv = 0.1 * 2.0 * np.linalg.norm(v_chaser) / max(vessel.mass, 1.0)
                total_dv_spent += max(rcs_dv, 0.5)  # minimum accounting

                dashboard.update(
                    [dist / 1000.0, dv_rel, total_dv_spent, phase, 0],
                    ut=sc.ut,
                )
                continue

            # ── Lambert phase ──────────────────────────────────────────
            # Wait between maneuvers
            if iteration > 1:
                print(f'  Waiting {MANEUVER_WAIT:.0f}s before next maneuver...')
                dashboard.update(
                    [dist / 1000.0, dv_rel, total_dv_spent, phase, MANEUVER_WAIT],
                    ut=sc.ut,
                )
                time.sleep(MANEUVER_WAIT)

            # Estimate transfer time
            dt = estimate_transfer_time(r_chaser, r_target, target_period)

            # Solve Lambert problem
            try:
                v1_lambert, v2_lambert = lambert_universal(r_chaser, r_target, dt, MU_KERBIN)
            except Exception as e:
                logger.warning(f'Lambert solver failed (iteration {iteration}): {e}')
                # Try with longer transfer time
                dt *= 2.0
                try:
                    v1_lambert, v2_lambert = lambert_universal(r_chaser, r_target, dt, MU_KERBIN)
                except Exception as e2:
                    logger.error(f'Lambert solver failed again: {e2}')
                    print(f'  Lambert solver failed, skipping iteration {iteration}')
                    continue

            # Compute required delta-v
            dv_needed = v1_lambert - v_chaser
            dv_mag = np.linalg.norm(dv_needed)

            # Don't burn more than half the remaining budget in one go
            remaining_budget = dv_budget - total_dv_spent
            if dv_mag > remaining_budget * 0.5:
                dv_mag = remaining_budget * 0.5
                dv_needed = dv_needed / np.linalg.norm(dv_needed) * dv_mag

            # Execute burn
            print(
                f'  Iteration {iteration}: Δv={dv_mag:.1f} m/s, '
                f'dist={dist / 1000:.2f} km, dt={dt:.0f} s'
            )

            burn_dv = execute_burn(vessel, dv_needed, body_rf)
            total_dv_spent += burn_dv

            # Update dashboard
            dashboard.update(
                [dist / 1000.0, dv_rel, total_dv_spent, phase, 0],
                ut=sc.ut,
            )

        else:
            # Max iterations reached
            print(f'\n❌ FAILED: Max iterations ({MAX_RENDEZVOUS_ITERATIONS}) reached')
            print(f'   Final distance: {dist:.0f} m')
            print(f'   Total Δv spent: {total_dv_spent:.1f} m/s')
            status = 'FAILED'

    except KeyboardInterrupt:
        print('\n\nInterrupted by user.')
        status = 'INTERRUPTED'

    finally:
        dashboard.stop()

    # ── Report final status ────────────────────────────────────────────
    if status == 'SUCCESS':
        print('\n✅ SUCCESS: Rendezvous complete')
    elif status == 'DEGRADED':
        print('\n⚠️  DEGRADED: Hohmann fallback used (budget exceeded)')
    elif status == 'FAILED':
        print('\n❌ FAILED: Could not achieve rendezvous')

    print(f'   Iterations: {iteration}')
    print(f'   Total Δv spent: {total_dv_spent:.1f} m/s')
    print(f'   Budget: {dv_budget:.1f} m/s')


if __name__ == '__main__':
    main()
