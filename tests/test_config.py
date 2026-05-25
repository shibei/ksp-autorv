"""Tests for the shared config module (no external dependencies required).

Covers: env var overrides, default values, Kerbin rotation constants,
kRPC connection params, flight profile defaults, and internal consistency.
"""

from __future__ import annotations

import math
import os

import pytest


class TestConfigDefaults:
    """Verify every config constant has a sensible default value."""

    def test_ksc_address_default(self):
        from krpc_rendezvous.common.config import KSC_ADDRESS

        assert KSC_ADDRESS == '172.17.64.1'

    def test_ksc_ports_default(self):
        from krpc_rendezvous.common.config import KSC_RPC_PORT, KSC_STREAM_PORT

        assert KSC_RPC_PORT == 50000
        assert KSC_STREAM_PORT == 50001

    def test_vertical_alt(self):
        from krpc_rendezvous.common.config import VERTICAL_ALT

        assert VERTICAL_ALT == 4000.0

    def test_gravity_turn_end(self):
        from krpc_rendezvous.common.config import GRAVITY_TURN_END

        assert GRAVITY_TURN_END == 70000.0

    def test_target_heading(self):
        from krpc_rendezvous.common.config import TARGET_HEADING

        assert TARGET_HEADING == 90.0

    def test_pitch_end_deg(self):
        from krpc_rendezvous.common.config import PITCH_END_DEG

        assert PITCH_END_DEG == 5.0

    def test_pid_limits(self):
        from krpc_rendezvous.common.config import (
            PITCH_OUTPUT_MAX,
            THROTTLE_MAX,
            THROTTLE_MIN,
            YAW_DEADBAND,
            YAW_OUTPUT_MAX,
        )

        assert PITCH_OUTPUT_MAX == 30.0
        assert YAW_OUTPUT_MAX == 10.0
        assert abs(YAW_DEADBAND) > 0
        assert 0 <= THROTTLE_MIN < THROTTLE_MAX <= 1.0

    def test_window_search(self):
        from krpc_rendezvous.common.config import (
            WINDOW_SEARCH_DAYS,
            WINDOW_SEARCH_STEP,
            WINDOW_SKIP_SECONDS,
            WINDOW_TOLERANCE_DEG,
        )

        assert WINDOW_TOLERANCE_DEG == 2.0
        assert WINDOW_SKIP_SECONDS == 18000  # 5 hours
        assert WINDOW_SEARCH_STEP == 30
        assert WINDOW_SEARCH_DAYS == 1

    def test_rendezvous_constants(self):
        from krpc_rendezvous.common.config import (
            DV_BUDGET_DEFAULT,
            MANEUVER_WAIT,
            MAX_RENDEZVOUS_ITERATIONS,
            RENDEZVOUS_DIST_THRESHOLD,
            RENDEZVOUS_VEL_THRESHOLD,
            TERMINAL_DIST_THRESHOLD,
        )

        assert MAX_RENDEZVOUS_ITERATIONS == 20
        assert RENDEZVOUS_DIST_THRESHOLD == 200.0
        assert RENDEZVOUS_VEL_THRESHOLD == 10.0
        assert 0 < RENDEZVOUS_DIST_THRESHOLD < TERMINAL_DIST_THRESHOLD
        assert MANEUVER_WAIT > 0
        assert DV_BUDGET_DEFAULT > 0

    def test_ascent_target_alt(self):
        from krpc_rendezvous.common.config import ASCENT_TARGET_ALT

        assert ASCENT_TARGET_ALT == 80000.0

    def test_aoa_limit(self):
        from krpc_rendezvous.common.config import AOA_LIMIT

        assert AOA_LIMIT == 5.0

    def test_data_collection_duration(self):
        from krpc_rendezvous.common.config import DATA_COLLECTION_DURATION

        assert DATA_COLLECTION_DURATION == 8.0


class TestKerbinRotation:
    """Kerbin rotation period and rate consistency."""

    def test_rotation_period(self):
        from krpc_rendezvous.common.config import KERBIN_ROTATION_PERIOD

        assert KERBIN_ROTATION_PERIOD == 21600.0  # 6 hours

    def test_rotation_rate_consistency(self):
        from krpc_rendezvous.common.config import (
            KERBIN_ROTATION_PERIOD,
            KERBIN_ROTATION_RATE,
        )

        expected = 2.0 * math.pi / KERBIN_ROTATION_PERIOD
        assert pytest.approx(expected) == KERBIN_ROTATION_RATE

    def test_rotation_rate_sign(self):
        from krpc_rendezvous.common.config import KERBIN_ROTATION_RATE

        assert KERBIN_ROTATION_RATE > 0  # counterclockwise


class TestEnvOverrides:
    """Environment variable overrides for kRPC connection params."""

    def test_env_host_override(self, env_with_host):
        """Must re-import config to pick up env changes."""
        import importlib

        import krpc_rendezvous.common.config as cfg

        importlib.reload(cfg)
        assert cfg.KSC_ADDRESS == '10.0.0.99'
        assert cfg.KSC_RPC_PORT == 50002
        assert cfg.KSC_STREAM_PORT == 50003

    def test_env_host_default(self, env_reset):
        """Without env vars, should fall back to defaults."""
        import importlib

        import krpc_rendezvous.common.config as cfg

        importlib.reload(cfg)
        assert cfg.KSC_ADDRESS == '172.17.64.1'
        assert cfg.KSC_RPC_PORT == 50000
        assert cfg.KSC_STREAM_PORT == 50001

    def test_env_partial_override(self, env_reset):
        """Setting only some env vars should keep defaults for others."""
        os.environ['KRPCC_HOST'] = '192.168.1.1'
        import importlib

        import krpc_rendezvous.common.config as cfg

        importlib.reload(cfg)
        assert cfg.KSC_ADDRESS == '192.168.1.1'
        assert cfg.KSC_RPC_PORT == 50000  # unchanged
        assert cfg.KSC_STREAM_PORT == 50001  # unchanged

    def test_env_port_parsing(self, env_reset):
        """Port numbers from env vars must be valid."""
        os.environ['KRPCC_RPC_PORT'] = '50099'
        import importlib

        import krpc_rendezvous.common.config as cfg

        importlib.reload(cfg)
        assert cfg.KSC_RPC_PORT == 50099


class TestConfigInternalConsistency:
    """Cross-field consistency checks."""

    def test_gravity_turn_range(self):
        from krpc_rendezvous.common.config import GRAVITY_TURN_END, VERTICAL_ALT

        assert VERTICAL_ALT < GRAVITY_TURN_END

    def test_inclination_limit_sane(self):
        from krpc_rendezvous.common.config import MAX_INCLINATION_DEG

        assert 0 < MAX_INCLINATION_DEG <= 90

    def test_throttle_range(self):
        from krpc_rendezvous.common.config import THROTTLE_MAX, THROTTLE_MIN

        assert 0.0 <= THROTTLE_MIN < THROTTLE_MAX <= 1.0
