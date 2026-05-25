"""E2E launch window integration test against live kRPC/KSP.

Validates the full window search → warp → state verification pipeline.
Requires a live kRPC server with KSP in flight scene (vessel on launch pad).
"""

import math
import sys

import pytest

from krpc_rendezvous.common.config import WINDOW_SEARCH_STEP, WINDOW_SEARCH_DAYS, WINDOW_TOLERANCE_DEG
from krpc_rendezvous.launch_window import (
    find_launch_windows,
    compute_window_dv,
    read_target_orbit,
    ksc_longitude,
    angular_distance,
)

TARGET_NAME = 'Z-MAP卫星发射套装'
TARGET_SITUATION = 'orbiting'
ASCENT_TIME = 198.0  # seconds, from earlier empirical measurement


@pytest.fixture(scope='module')
def krpc_conn():
    import krpc
    conn = krpc.connect(name='e2e-window-test', address='172.17.64.1', rpc_port=50000, stream_port=50001)
    yield conn
    conn.close()


@pytest.fixture
def target_and_orbit(krpc_conn, request):
    sc = krpc_conn.space_center
    matches = [v for v in sc.vessels if v.name == TARGET_NAME and v.situation.name == TARGET_SITUATION]
    if not matches:
        pytest.skip(f'Target "{TARGET_NAME}" ({TARGET_SITUATION}) not found in current game')
    target = matches[0]
    sc.target_vessel = target
    orbit = read_target_orbit(krpc_conn)

    def _cleanup():
        sc.target_vessel = None

    request.addfinalizer(_cleanup)
    return target, orbit


class TestLaunchWindowE2E:
    """E2E tests for the full launch window pipeline."""

    @pytest.mark.live_krpc
    def test_set_target(self, krpc_conn, target_and_orbit):
        target, orbit = target_and_orbit
        assert krpc_conn.space_center.target_vessel.name == TARGET_NAME
        assert orbit['inc'] <= math.radians(3.0), f'Target inclination {math.degrees(orbit["inc"]):.2f}° exceeds equatorial limit'
        assert orbit['sma'] > 600000, f'SMA={orbit["sma"]:.0f}m too low (below Kerbin surface)'

    @pytest.mark.live_krpc
    def test_find_windows(self, krpc_conn, target_and_orbit):
        _, orbit = target_and_orbit
        sc = krpc_conn.space_center
        now = sc.ut

        windows = list(find_launch_windows(
            now=now, lan=orbit['lan'], arg_pe=orbit['arg_pe'],
            ecc=orbit['ecc'], sma=orbit['sma'], M0=orbit['M0'],
            epoch_ut=orbit['epoch'], ascent_time=ASCENT_TIME,
            search_days=WINDOW_SEARCH_DAYS, step=WINDOW_SEARCH_STEP,
            tolerance_deg=WINDOW_TOLERANCE_DEG, skip_seconds=3600,
        ))
        assert len(windows) > 0, 'No launch windows found'
        for win_ut, err_rad in windows:
            assert math.degrees(err_rad) <= WINDOW_TOLERANCE_DEG
        # Windows should be time-ordered
        for i in range(len(windows) - 1):
            assert windows[i][0] < windows[i + 1][0]

    @pytest.mark.live_krpc
    def test_compute_dv(self, krpc_conn, target_and_orbit):
        _, orbit = target_and_orbit
        sc = krpc_conn.space_center
        now = sc.ut

        windows = list(find_launch_windows(
            now=now, lan=orbit['lan'], arg_pe=orbit['arg_pe'],
            ecc=orbit['ecc'], sma=orbit['sma'], M0=orbit['M0'],
            epoch_ut=orbit['epoch'], ascent_time=ASCENT_TIME,
            search_days=WINDOW_SEARCH_DAYS, step=WINDOW_SEARCH_STEP,
            tolerance_deg=WINDOW_TOLERANCE_DEG, skip_seconds=3600,
        ))

        candidates = []
        for win_ut, err in windows:
            ad, rd, td = compute_window_dv(win_ut, now, ASCENT_TIME, orbit, 80000.0)
            if td:
                candidates.append((win_ut, td, err, ad, rd))
        assert len(candidates) > 0, 'No valid windows after Δv computation'

        # Select nearest window
        candidates.sort(key=lambda x: x[0])
        win_ut, total_dv, err, ascent_dv, rendez_dv = candidates[0]

        assert total_dv > 0
        assert ascent_dv > 0
        assert rendez_dv >= 0
        assert math.degrees(err) <= WINDOW_TOLERANCE_DEG
        assert win_ut >= now

        # Ascent Δv should approximate circular orbital velocity at 80km
        import krpc_rendezvous.common.orbit_utils as ou
        v_circ_80 = math.sqrt(ou.MU_KERBIN / (ou.R_KERBIN + 80000))
        assert ascent_dv == pytest.approx(v_circ_80, rel=0.02)

    @pytest.mark.live_krpc
    def test_warp_to_window(self, krpc_conn, target_and_orbit):
        """Time-warp to nearest window and verify state after warp."""
        target, orbit = target_and_orbit
        sc = krpc_conn.space_center
        now = sc.ut

        windows = list(find_launch_windows(
            now=now, lan=orbit['lan'], arg_pe=orbit['arg_pe'],
            ecc=orbit['ecc'], sma=orbit['sma'], M0=orbit['M0'],
            epoch_ut=orbit['epoch'], ascent_time=ASCENT_TIME,
            search_days=WINDOW_SEARCH_DAYS, step=WINDOW_SEARCH_STEP,
            tolerance_deg=WINDOW_TOLERANCE_DEG, skip_seconds=3600,
        ))
        candidates = [(wu, ad, rd, td, er) for wu, er in windows
                      for ad, rd, td in [compute_window_dv(wu, now, ASCENT_TIME, orbit, 80000.0)]
                      if td]
        candidates.sort(key=lambda x: x[0])
        win_ut, _, _, _, _ = candidates[0]
        warp_ut = win_ut - 10.0

        if warp_ut > sc.ut:
            sc.warp_to(warp_ut)
            # warp_to may leave residual warp active; kill it immediately
            sc.rails_warp_factor = 0
            import time
            time.sleep(0.5)  # let physics settle

        # ── Post-warp verification ──────────────────────────────────────
        vessel = sc.active_vessel
        assert vessel is not None
        assert vessel.name != ''

        # Warp accuracy: check only if warp actually executed
        if warp_ut > now:
            warp_delta = sc.ut - warp_ut
            assert abs(warp_delta) < 2.0, (
                f'Warp landing error: {warp_delta:.1f}s'
            )

        # Time warp should be 1x
        assert sc.rails_warp_factor == 0, f'Warp not reset: factor={sc.rails_warp_factor}'

        # Verify window prediction: compute alignment at the window's
        # own launch time (win_ut), using live target data at rendezvous
        rend_ut = win_ut + ASCENT_TIME
        nu = target.orbit.true_anomaly_at_ut(rend_ut)
        tgt_lon = (orbit['lan'] + orbit['arg_pe'] + nu) % (2.0 * math.pi)
        ksc_lon = ksc_longitude(win_ut)
        err = angular_distance(ksc_lon, tgt_lon)

        did_warp = sc.ut - warp_ut
        print(f'\n  [DEBUG] Warp: target={warp_ut:.1f}  actual={sc.ut:.1f}  '
              f'delta={sc.ut - warp_ut:.1f}s  '
              f'{"" if warp_ut > now else "(window already current, no warp)"}',
              file=sys.stderr)
        print(f'  [DEBUG] Window prediction: tgt_lon={math.degrees(tgt_lon):.2f}°  '
              f'ksc_lon={math.degrees(ksc_lon):.2f}°  '
              f'err={math.degrees(err):.3f}°', file=sys.stderr)

        assert math.degrees(err) <= WINDOW_TOLERANCE_DEG, (
            f'Window prediction error {math.degrees(err):.3f}° '
            f'exceeds tolerance {WINDOW_TOLERANCE_DEG}°'
        )
