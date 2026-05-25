"""Atmospheric ascent guidance for KSP kRPC.

Reads launch_window_output.json, waits for launch window, then executes
automated ascent through 3 phases:
  1. Vertical ascent (0→4km): SAS hold 90° pitch, collect flight data
  2. Gravity turn (4km→70km): PID closed-loop tracking pitch curve
  3. Termination: apogee ≥ target - 5km or altitude > 70km → MECO

Outputs ascent_state.json with orbital state for next script.
"""

import argparse
import csv
import json
import logging
import math
import time

import numpy as np

from krpc_rendezvous.common.config import (
    AOA_LIMIT,
    DATA_COLLECTION_DURATION,
    GRAVITY_TURN_END,
    PITCH_END_DEG,
    PITCH_OUTPUT_MAX,
    TARGET_HEADING,
    THROTTLE_MAX,
    THROTTLE_MIN,
    VERTICAL_ALT,
    YAW_DEADBAND,
    YAW_OUTPUT_MAX,
)
from krpc_rendezvous.common.dashboard import Dashboard, format_time
from krpc_rendezvous.common.krpc_connection import connect_krpc, safe_warp
from krpc_rendezvous.common.orbit_utils import (
    R_KERBIN,
    delta_v_estimate,
    gravity_turn_pitch_profile,
)
from krpc_rendezvous.common.pid import PIDController

logger = logging.getLogger(__name__)


# ── Response estimation ───────────────────────────────────────────────


def estimate_response(vessel):
    """Estimate vessel response class from mass and reaction wheel torque.

    Returns (response_class, gain_scale) tuple.
    """
    import numpy as np

    mass = vessel.mass
    torque = 0.0
    for rw in vessel.parts.reaction_wheels:
        t = rw.available_torque
        if isinstance(t, (list, tuple)) and len(t) > 0:
            if isinstance(t[0], (list, tuple)):
                t = float(np.linalg.norm(t[0]))
            else:
                t = float(t[0])
        if t > 0:
            torque += t

    if torque == 0:
        torque = 15.0  # fallback default

    proxy = torque / max(mass, 1.0)
    if proxy > 0.003:
        return 'fast', 0.7
    elif proxy > 0.001:
        return 'medium', 1.0
    else:
        return 'slow', 1.5


# ── Flight data collection ────────────────────────────────────────────


def collect_flight_data(vessel, conn, duration=10.0):
    """Collect pitch errors and yaw changes over *duration* seconds.

    Returns (pitch_std, yaw_activity) for Kp/Kd scaling.
    """
    pitch_samples = []
    yaw_changes = []
    prev_heading = None

    t0 = time.time()
    while time.time() - t0 < duration:
        f = vessel.flight(vessel.surface_velocity_reference_frame)
        pitch = f.pitch
        heading = f.heading
        pitch_samples.append(pitch)
        if prev_heading is not None:
            yaw_changes.append(abs(heading - prev_heading))
        prev_heading = heading
        time.sleep(0.1)

    pitch_std = float(np.std(pitch_samples)) if pitch_samples else 1.0
    yaw_activity = float(np.mean(yaw_changes)) if yaw_changes else 0.5

    return pitch_std, yaw_activity


# ── PID gain determination ─────────────────────────────────────────────


def get_pitch_gains(stage, pitch_std):
    """Return (kp, ki, kd) for the given flight stage.

    stage: 'vertical', 'low_dynamic', 'max_q', or 'high_altitude'
    pitch_std: standard deviation of pitch from Phase 1 data collection
    """
    if stage == 'vertical':
        return 0.15, 0.0, 0.02
    elif stage == 'low_dynamic':
        kp = min(0.3 / max(pitch_std, 0.1), 0.5)
        return kp, 0.01, 0.05
    elif stage == 'max_q':
        return 0.15, 0.005, 0.08
    else:  # high_altitude
        return 0.25, 0.02, 0.04


def get_gain_stage(altitude):
    """Determine gain scheduling stage from altitude."""
    if altitude < 8000:
        return 'low_dynamic'
    elif altitude < 20000:
        return 'max_q'
    else:
        return 'high_altitude'


# ── Main ascent logic ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Atmospheric ascent guidance for KSP kRPC')
    parser.add_argument('--tune', action='store_true', help='Enable CSV logging for PID tuning')
    parser.add_argument(
        '--window-file',
        type=str,
        default='launch_window_output.json',
        help='Path to launch window JSON (default: launch_window_output.json)',
    )
    args = parser.parse_args()

    # ── Load launch window data ────────────────────────────────────────
    with open(args.window_file) as f:
        window_data = json.load(f)

    launch_ut = window_data['launch_ut']
    target_sma = window_data['target_orbit']['sma']
    target_apogee = target_sma - R_KERBIN  # target apogee altitude

    # ── Connect to kRPC ────────────────────────────────────────────────
    print('Connecting to kRPC...')
    conn = connect_krpc()
    vessel = conn.space_center.active_vessel

    # ── Wait for launch window ─────────────────────────────────────────
    current_ut = conn.space_center.ut
    wait_seconds = launch_ut - current_ut
    if wait_seconds > 0:
        print(f'Waiting for launch window: {format_time(wait_seconds)} remaining')
        safe_warp(launch_ut, margin=10)

    # ── Setup streams ──────────────────────────────────────────────────
    flight = vessel.flight(vessel.surface_velocity_reference_frame)
    orbit = vessel.orbit

    alt_stream = conn.add_stream(getattr, flight, 'mean_altitude')
    speed_stream = conn.add_stream(getattr, flight, 'speed')
    pitch_stream = conn.add_stream(getattr, flight, 'pitch')
    heading_stream = conn.add_stream(getattr, flight, 'heading')
    ap_stream = conn.add_stream(getattr, orbit, 'apoapsis_altitude')
    pe_stream = conn.add_stream(getattr, orbit, 'periapsis_altitude')

    # ── Dashboard ──────────────────────────────────────────────────────
    columns = [
        ('ALT', 'km', 1),
        ('VEL', 'm/s', 0),
        ('PITCH', 'deg', 1),
        ('AP', 'km', 1),
        ('PE', 'km', 1),
        ('DV', 'm/s', 0),
    ]
    dashboard = Dashboard(columns=columns)
    dashboard.start()

    # ── CSV logging ────────────────────────────────────────────────────
    csv_file = None
    csv_writer = None
    if args.tune:
        csv_file = open('ascent_tune.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['time', 'altitude', 'target_pitch', 'actual_pitch', 'error', 'kp', 'ki_out', 'kd_out']
        )

    # ── Phase 1: Vertical Ascent (0→4km) ──────────────────────────────
    print('\n=== Phase 1: Vertical Ascent ===')
    ap = vessel.auto_pilot
    ap.reference_frame = vessel.surface_velocity_reference_frame
    ap.target_pitch_and_heading(90.0, TARGET_HEADING)
    ap.engage()

    vessel.control.throttle = 1.0

    # Activate first stage (launch clamps + engines)
    vessel.control.activate_next_stage()

    # Collect flight data for PID tuning
    print(f'Collecting flight data for {DATA_COLLECTION_DURATION:.0f}s...')
    pitch_std, yaw_activity = collect_flight_data(vessel, conn, duration=DATA_COLLECTION_DURATION)
    print(f'  pitch_std={pitch_std:.3f}, yaw_activity={yaw_activity:.3f}')

    # Estimate response class
    response_class, gain_scale = estimate_response(vessel)
    print(f'  Response class: {response_class}, gain_scale: {gain_scale}')

    # Wait for vertical altitude
    cached_mass = vessel.mass
    last_mass_update = conn.space_center.ut
    while alt_stream() < VERTICAL_ALT:
        altitude = alt_stream()
        speed = speed_stream()
        apogee = ap_stream()
        perigee = pe_stream()
        if conn.space_center.ut - last_mass_update > 2.0:
            cached_mass = vessel.mass
            last_mass_update = conn.space_center.ut
        dv = delta_v_estimate(350.0, cached_mass, cached_mass * 0.7)
        dashboard.update(
            [altitude / 1000.0, speed, pitch_stream(), apogee / 1000.0, perigee / 1000.0, dv],
            ut=conn.space_center.ut - launch_ut,
        )
        time.sleep(0.25)

    # ── Phase 2: Gravity Turn (4km→70km) ──────────────────────────────
    print('\n=== Phase 2: Gravity Turn ===')

    # Initialize PID controllers
    pitch_pid = PIDController(
        kp=0.15,
        ki=0.0,
        kd=0.02,
        output_min=-PITCH_OUTPUT_MAX,
        output_max=PITCH_OUTPUT_MAX,
    )
    yaw_pid = PIDController(
        kp=0.1,
        ki=0.0,
        kd=0.01,
        output_min=-YAW_OUTPUT_MAX,
        output_max=YAW_OUTPUT_MAX,
        deadband=YAW_DEADBAND,
    )
    throttle_pid = PIDController(
        kp=0.001,
        ki=0.0001,
        kd=0.0005,
        output_min=THROTTLE_MIN - 1.0,
        output_max=THROTTLE_MAX - 1.0,
    )

    prev_time = time.time()
    launch_time = launch_ut
    cached_mass = vessel.mass
    last_mass_update = 0.0

    while True:
        now = time.time()
        dt = now - prev_time
        if dt < 0.02:  # ~50 Hz max
            time.sleep(0.01)
            continue
        prev_time = now

        altitude = alt_stream()
        speed = speed_stream()
        current_pitch = pitch_stream()
        current_heading = heading_stream()
        apogee = ap_stream()
        perigee = pe_stream()

        # Check termination conditions
        if apogee >= target_apogee - 5000.0:
            print(f'\nApogee {apogee / 1000:.1f} km >= target {target_apogee / 1000:.1f} km - 5 km')
            break
        if altitude > GRAVITY_TURN_END:
            print(f'\nAltitude {altitude / 1000:.1f} km > {GRAVITY_TURN_END / 1000:.0f} km')
            break

        # Gain scheduling
        stage = get_gain_stage(altitude)
        kp, ki, kd = get_pitch_gains(stage, pitch_std)

        # ── Auto-staging ────────────────────────────────────────────────
        # Check if current stage has no active/producing engines
        any_active = any(
            e.active and e.has_fuel
            for e in vessel.parts.engines
            if e.part.stage == vessel.control.current_stage
        )
        if not any_active and vessel.control.current_stage > 0:
            print(f'\nAuto-staging (stage {vessel.control.current_stage} depleted)')
            for _ in range(3):
                try:
                    vessel.control.activate_next_stage()
                    break
                except Exception:
                    time.sleep(0.1)
            # Freeze PID for 1s after staging
            pitch_pid.reset()
            yaw_pid.reset()
            time.sleep(1.0)
            print(f'  New stage: {vessel.control.current_stage}')

        # Scale gains by response class
        kp *= gain_scale
        kd *= gain_scale

        # Update pitch PID gains
        pitch_pid.kp = kp
        pitch_pid.ki = ki
        pitch_pid.kd = kd

        # Compute target pitch from gravity turn profile
        target_pitch_rad = gravity_turn_pitch_profile(
            altitude, VERTICAL_ALT, GRAVITY_TURN_END, math.radians(PITCH_END_DEG)
        )
        target_pitch_deg = math.degrees(target_pitch_rad)

        # Pitch error
        pitch_error = current_pitch - target_pitch_deg

        # Yaw error (keep equatorial heading)
        yaw_error = current_heading - TARGET_HEADING

        # Apogee error for throttle control
        apogee_error = apogee - target_apogee

        # AoA protection: reduce gains if AoA too high
        # Approximate AoA as difference between pitch and velocity direction
        aoa = abs(current_pitch - target_pitch_deg)
        if aoa > AOA_LIMIT:
            pitch_pid.kp *= 0.5
            pitch_pid.kd *= 0.5

        # PID updates
        pitch_correction = pitch_pid.update(pitch_error, dt)
        yaw_correction = yaw_pid.update(yaw_error, dt)
        throttle_correction = throttle_pid.update(apogee_error, dt)

        # Apply to autopilot
        commanded_pitch = target_pitch_deg + pitch_correction
        commanded_pitch = max(0.0, min(90.0, commanded_pitch))
        ap.target_pitch_and_heading(commanded_pitch, TARGET_HEADING + yaw_correction)

        # Throttle control
        throttle = 1.0 + throttle_correction
        throttle = max(THROTTLE_MIN, min(THROTTLE_MAX, throttle))
        vessel.control.throttle = throttle

        if conn.space_center.ut - last_mass_update > 3.0:
            cached_mass = vessel.mass
            last_mass_update = conn.space_center.ut
        dv = delta_v_estimate(350.0, cached_mass, cached_mass * 0.7)

        # Dashboard update
        dashboard.update(
            [altitude / 1000.0, speed, current_pitch, apogee / 1000.0, perigee / 1000.0, dv],
            ut=conn.space_center.ut - launch_time,
        )

        # CSV logging
        if csv_writer is not None:
            csv_writer.writerow(
                [
                    f'{conn.space_center.ut - launch_time:.2f}',
                    f'{altitude:.1f}',
                    f'{target_pitch_deg:.2f}',
                    f'{current_pitch:.2f}',
                    f'{pitch_error:.3f}',
                    f'{kp:.4f}',
                    f'{pitch_pid.ki * pitch_pid._integral:.4f}',
                    f'{pitch_pid.kd * (pitch_error - (pitch_pid._prev_error or pitch_error)) / max(dt, 0.001):.4f}',
                ]
            )

    # ── Phase 3: Termination (MECO) ───────────────────────────────────
    print('\n=== Phase 3: MECO ===')
    vessel.control.throttle = 0.0
    ap.disengage()

    # ── Save ascent state ──────────────────────────────────────────────
    altitude = alt_stream()
    speed = speed_stream()
    apogee = ap_stream()
    perigee = pe_stream()

    # Get orbital elements
    orbit = vessel.orbit
    sma = orbit.semi_major_axis
    ecc = orbit.eccentricity
    inc = orbit.inclination

    # Position and velocity in orbital reference frame
    pos = vessel.position(orbit.body.reference_frame)
    vel = vessel.velocity(orbit.body.reference_frame)

    state = {
        'position': list(pos),
        'velocity': list(vel),
        'apogee': apogee,
        'perigee': perigee,
        'inclination': inc,
        'eccentricity': ecc,
        'semi_major_axis': sma,
        'remaining_mass': vessel.mass,
        'altitude': altitude,
        'speed': speed,
        'target_sma': target_sma,
        'target_apogee': target_apogee,
    }

    with open('ascent_state.json', 'w') as f:
        json.dump(state, f, indent=2)
    print('Ascent state saved to ascent_state.json')

    # ── Cleanup ────────────────────────────────────────────────────────
    alt_stream.remove()
    speed_stream.remove()
    pitch_stream.remove()
    heading_stream.remove()
    ap_stream.remove()
    pe_stream.remove()

    if csv_file is not None:
        csv_file.close()

    dashboard.stop()

    print('\nAscent complete!')
    print(f'  Altitude:    {altitude / 1000:.1f} km')
    print(f'  Apogee:      {apogee / 1000:.1f} km')
    print(f'  Perigee:     {perigee / 1000:.1f} km')
    print(f'  Eccentricity: {ecc:.4f}')
    print(f'  Inclination:  {math.degrees(inc):.2f}°')


if __name__ == '__main__':
    main()
