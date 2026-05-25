"""Unit tests for launch window math (pure functions, no kRPC required).

Tests cover: longitude helpers, angular distance, window search logic,
Δv computation, and table formatting.
"""

import math

import pytest

from krpc_rendezvous.common.config import KERBIN_ROTATION_PERIOD
from krpc_rendezvous.common.orbit_utils import (
    MU_KERBIN,
    R_KERBIN,
    orbital_period,
)
from krpc_rendezvous.launch_window import (
    PITCH_CURVE,
    angular_distance,
    compute_window_dv,
    find_launch_windows,
    format_table,
    ksc_longitude,
    target_longitude,
)

# ── ksc_longitude ──────────────────────────────────────────────────────


def test_ksc_longitude_starts_at_zero():
    lon = ksc_longitude(0.0)
    assert lon == pytest.approx(0.0)


def test_ksc_longitude_one_full_rotation():
    """After one sidereal day, KSC should return to same longitude."""
    lon = ksc_longitude(KERBIN_ROTATION_PERIOD)
    assert lon == pytest.approx(0.0)


def test_ksc_longitude_half_rotation():
    lon = ksc_longitude(KERBIN_ROTATION_PERIOD / 2)
    assert lon == pytest.approx(math.pi)


def test_ksc_longitude_increases_with_time():
    lon1 = ksc_longitude(100.0)
    lon2 = ksc_longitude(200.0)
    assert lon2 > lon1


def test_ksc_longitude_mod_2pi():
    lon = ksc_longitude(KERBIN_ROTATION_PERIOD * 1.5)
    assert 0 <= lon <= 2 * math.pi


# ── target_longitude ───────────────────────────────────────────────────


def test_target_longitude_circular_equatorial():
    """A circular equatorial orbit at known position."""
    sma = R_KERBIN + 100_000
    ecc = 0.0
    lan = 0.0
    arg_pe = 0.0
    M0 = 0.0  # mean anomaly = 0 at epoch
    epoch_ut = 0.0

    # At epoch, should be at LAN + arg_pe + 0 = 0
    lon = target_longitude(0.0, epoch_ut, lan, arg_pe, ecc, sma, M0)
    assert lon == pytest.approx(0.0)


def test_target_longitude_moves_with_time():
    """Longitude should change as the target orbits."""
    sma = R_KERBIN + 100_000
    ecc = 0.0
    lan = 0.0
    arg_pe = 0.0
    M0 = 0.0
    epoch_ut = 0.0

    period = orbital_period(sma)
    lon_t0 = target_longitude(0.0, epoch_ut, lan, arg_pe, ecc, sma, M0)
    lon_t1 = target_longitude(period / 4, epoch_ut, lan, arg_pe, ecc, sma, M0)
    # After quarter period, mean anomaly advanced by ~π/2
    diff = (lon_t1 - lon_t0) % (2 * math.pi)
    assert diff == pytest.approx(math.pi / 2, abs=0.01)


def test_target_longitude_after_full_period():
    """After one full orbit, longitude should be same (no nodal precession)."""
    sma = R_KERBIN + 100_000
    ecc = 0.0
    lan = math.radians(45)
    arg_pe = math.radians(30)
    M0 = 0.0
    epoch_ut = 0.0

    period = orbital_period(sma)
    lon_start = target_longitude(0.0, epoch_ut, lan, arg_pe, ecc, sma, M0)
    lon_end = target_longitude(period, epoch_ut, lan, arg_pe, ecc, sma, M0)
    assert lon_end == pytest.approx(lon_start)


# ── angular_distance ───────────────────────────────────────────────────


def test_angular_distance_zero():
    assert angular_distance(1.0, 1.0) == pytest.approx(0.0)


def test_angular_distance_quarter():
    assert angular_distance(0.0, math.pi / 2) == pytest.approx(math.pi / 2)


def test_angular_distance_wraparound():
    """Distance across 2π boundary."""
    d = angular_distance(0.1, 2 * math.pi - 0.1)
    assert d == pytest.approx(0.2)


def test_angular_distance_max():
    """Maximum distance is π (cannot exceed half circumference)."""
    assert angular_distance(0.0, math.pi) == pytest.approx(math.pi)
    # distance between 0 and 3.2 is the short way: min(3.2, 2π-3.2)
    assert angular_distance(0.0, 3.2) == pytest.approx(2 * math.pi - 3.2)


# ── find_launch_windows ────────────────────────────────────────────────


def test_find_windows_basic():
    """Find at least one window for a circular equatorial target."""
    now = 10000.0
    sma = R_KERBIN + 100_000
    ecc = 0.0
    lan = 0.0
    arg_pe = 0.0
    M0 = 0.0
    epoch_ut = 0.0
    ascent_time = 180.0

    windows = list(
        find_launch_windows(
            now=now,
            lan=lan,
            arg_pe=arg_pe,
            ecc=ecc,
            sma=sma,
            M0=M0,
            epoch_ut=epoch_ut,
            ascent_time=ascent_time,
            search_days=5.0,
            step=30,
            tolerance_deg=2.0,
            skip_seconds=3600,
        )
    )
    assert len(windows) > 0
    for w in windows:
        assert len(w) == 2  # (ut, error)


def test_find_windows_error_within_tolerance():
    """All found windows should have error within tolerance."""
    now = 50000.0
    sma = R_KERBIN + 100_000
    tolerance_deg = 2.0

    windows = find_launch_windows(
        now=now,
        lan=0.0,
        arg_pe=0.0,
        ecc=0.0,
        sma=sma,
        M0=0.0,
        epoch_ut=0.0,
        ascent_time=180.0,
        search_days=3.0,
        step=30,
        tolerance_deg=tolerance_deg,
        skip_seconds=3600,
    )
    for _, err in windows:
        assert math.degrees(err) <= tolerance_deg


def test_find_windows_first_window_earliest():
    """First window should be earlier than second window."""
    now = 0.0
    sma = R_KERBIN + 100_000

    windows = find_launch_windows(
        now=now,
        lan=0.0,
        arg_pe=0.0,
        ecc=0.0,
        sma=sma,
        M0=0.0,
        epoch_ut=0.0,
        ascent_time=180.0,
        search_days=5.0,
        step=30,
        tolerance_deg=3.0,
        skip_seconds=3600,
    )
    windows = list(windows)
    if len(windows) >= 2:
        assert windows[0][0] < windows[1][0]


def test_find_windows_no_match_tight_tolerance():
    """Extremely tight tolerance should yield no windows."""
    windows = find_launch_windows(
        now=0.0,
        lan=0.0,
        arg_pe=0.0,
        ecc=0.0,
        sma=R_KERBIN + 100_000,
        M0=0.0,
        epoch_ut=0.0,
        ascent_time=180.0,
        search_days=1.0,
        step=30,
        tolerance_deg=0.01,
        skip_seconds=3600,
    )
    assert len(list(windows)) == 0


# ── compute_window_dv ──────────────────────────────────────────────────


@pytest.fixture
def circular_target():
    return {
        'lan': 0.0,
        'arg_pe': 0.0,
        'ecc': 0.0,
        'sma': R_KERBIN + 100_000,
        'M0': 0.0,
        'epoch': 0.0,
        'inc': 0.0,
        'true_anomaly': 0.0,
    }


def test_compute_window_dv_returns_floats(circular_target):
    ascent_dv, rendez_dv, total_dv = compute_window_dv(
        window_ut=10000.0,
        now=0.0,
        ascent_time=180.0,
        target_orbit=circular_target,
        ascent_altitude=80000.0,
    )
    assert ascent_dv is not None
    assert rendez_dv is not None
    assert total_dv is not None
    assert ascent_dv > 0
    assert rendez_dv >= 0
    assert total_dv > 0


def test_compute_window_dv_total_equals_sum(circular_target):
    ascent_dv, rendez_dv, total_dv = compute_window_dv(
        window_ut=20000.0,
        now=0.0,
        ascent_time=180.0,
        target_orbit=circular_target,
        ascent_altitude=80000.0,
    )
    assert total_dv == pytest.approx(ascent_dv + rendez_dv)


def test_compute_window_dv_ascent_reasonable():
    """Ascent Δv to 80 km should be near orbital velocity (~2279 m/s)."""
    target = {
        'lan': 0.0,
        'arg_pe': 0.0,
        'ecc': 0.0,
        'sma': R_KERBIN + 100_000,
        'M0': 0.0,
        'epoch': 0.0,
        'inc': 0.0,
    }
    ascent_dv, _, _ = compute_window_dv(
        10000.0,
        0.0,
        180.0,
        target,
        80000.0,
    )
    v_circ_80 = math.sqrt(MU_KERBIN / (R_KERBIN + 80000))
    assert ascent_dv == pytest.approx(v_circ_80, rel=0.01)


def test_compute_window_dv_different_results():
    """Different launch windows should produce different Δv."""
    target = {
        'lan': 0.0,
        'arg_pe': 0.0,
        'ecc': 0.0,
        'sma': R_KERBIN + 100_000,
        'M0': 0.0,
        'epoch': 0.0,
        'inc': 0.0,
    }
    # Two different window times
    _, _, t1 = compute_window_dv(10000.0, 0.0, 180.0, target, 80000.0)
    _, _, t2 = compute_window_dv(50000.0, 0.0, 180.0, target, 80000.0)
    assert t1 != pytest.approx(t2)


# ── format_table ───────────────────────────────────────────────────────


def test_format_table_basic():
    data = [
        {
            'launch_ut': 10000.0,
            'ascent_dv': 2200.0,
            'rendezvous_dv': 100.0,
            'total_dv': 2300.0,
            'available_dv': 3000.0,
        },
    ]
    output = format_table(data)
    assert 'Launch UT' in output
    assert 'Ascent Δv' in output
    assert 'Rendez Δv' in output
    assert '10000.0' in output
    assert '2300.0' in output


def test_format_table_multiple():
    data = [
        {
            'launch_ut': 10000.0,
            'ascent_dv': 2200.0,
            'rendezvous_dv': 100.0,
            'total_dv': 2300.0,
            'available_dv': 3000.0,
        },
        {
            'launch_ut': 50000.0,
            'ascent_dv': 2250.0,
            'rendezvous_dv': 50.0,
            'total_dv': 2300.0,
            'available_dv': 3000.0,
        },
    ]
    output = format_table(data)
    assert output.count('\n') >= 2  # header + separator + 2 rows


# ── PITCH_CURVE ────────────────────────────────────────────────────────


def test_pitch_curve_descending():
    """Pitch values should monotonically decrease with altitude."""
    pitches = [p for _, p in PITCH_CURVE]
    for i in range(len(pitches) - 1):
        assert pitches[i] >= pitches[i + 1]


def test_pitch_curve_starts_vertical():
    assert PITCH_CURVE[0][1] == pytest.approx(math.pi / 2)


def test_pitch_curve_ends_shallow():
    assert PITCH_CURVE[-1][1] == pytest.approx(math.radians(5.0))
