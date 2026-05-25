"""Shared configuration for the kRPC rendezvous autopilot.

Usage:
    from common.config import KSC_ADDRESS, KSC_RPC_PORT, safe_warp

Environment variable overrides:
    KRPCC_HOST       kRPC server address    (default: 172.17.64.1)
    KRPCC_RPC_PORT   kRPC RPC port          (default: 50000)
    KRPCC_STREAM     kRPC stream port       (default: 50001)
"""

import os

# ── kRPC connection defaults (overridable via env vars) ────────────────

KSC_ADDRESS = os.environ.get('KRPCC_HOST', '172.17.64.1')
KSC_RPC_PORT = int(os.environ.get('KRPCC_RPC_PORT', '50000'))
KSC_STREAM_PORT = int(os.environ.get('KRPCC_STREAM', '50001'))


# ── Kerbin rotation ────────────────────────────────────────────────────

KERBIN_ROTATION_PERIOD = 21600.0  # 6 h sidereal day [s]

import math

KERBIN_ROTATION_RATE = 2.0 * math.pi / KERBIN_ROTATION_PERIOD  # rad/s


# ── Launch / flight profile ────────────────────────────────────────────

VERTICAL_ALT = 4000.0  # Phase 1→2 transition altitude [m]
GRAVITY_TURN_END = 70000.0  # Phase 2→3 transition altitude [m]
TARGET_HEADING = 90.0  # Equatorial heading [deg]
PITCH_END_DEG = 5.0  # Pitch at end of gravity turn [deg]

# PID limits
PITCH_OUTPUT_MAX = 30.0  # ±30° pitch correction
YAW_OUTPUT_MAX = 10.0  # ±10° yaw correction
YAW_DEADBAND = 0.5  # ° deadband for yaw
THROTTLE_MIN = 0.3
THROTTLE_MAX = 1.0

# AoA protection
AOA_LIMIT = 5.0  # degrees

# Phase 1 data collection
DATA_COLLECTION_DURATION = 8.0  # seconds


# ── Launch window search ───────────────────────────────────────────────

MAX_INCLINATION_DEG = 3.0
WINDOW_TOLERANCE_DEG = 2.0
WINDOW_SKIP_SECONDS = 5 * 3600
WINDOW_SEARCH_STEP = 30  # seconds
WINDOW_SEARCH_DAYS = 1
ASCENT_TARGET_ALT = 80000.0  # m


# ── Orbital rendezvous ─────────────────────────────────────────────────

MAX_RENDEZVOUS_ITERATIONS = 20
RENDEZVOUS_DIST_THRESHOLD = 200.0  # m  – success
RENDEZVOUS_VEL_THRESHOLD = 10.0  # m/s – success
TERMINAL_DIST_THRESHOLD = 5000.0  # m  – switch to prop-nav
MANEUVER_WAIT = 30.0  # s  – between Lambert burns
DV_BUDGET_DEFAULT = 500.0  # m/s fallback
