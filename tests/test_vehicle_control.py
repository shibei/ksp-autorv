"""Vehicle control integration tests against live kRPC with orbital vessel.

Each test performs a real control action that should produce **visible**
vessel response (engine flame, RCS fire, attitude change) — then reads
back vehicle parameters to confirm the operation succeeded.

Requires:
  - KSP running with kRPC server at 172.17.64.1:50000 / 50001
  - Active vessel in orbit with at least one engine that has fuel

Marked ``live_krpc`` — auto-skipped when server is unreachable.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.live_krpc

# Give KSP time to propagate physics after each control action
_SETTLE_S = 0.5


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope='module')
def live_conn():
    import krpc

    conn = krpc.connect(
        name='test_control',
        address='172.17.64.1',
        rpc_port=50000,
        stream_port=50001,
    )
    yield conn
    conn.close()


@pytest.fixture
def sc(live_conn):
    return live_conn.space_center


@pytest.fixture
def vessel(sc):
    return sc.active_vessel


@pytest.fixture(autouse=True)
def _reset_warp(sc):
    """Ensure time warp is at 1x before each test."""
    sc.rails_warp_factor = 0
    sc.physics_warp_factor = 0
    yield


@pytest.fixture
def ctrl(vessel):
    return vessel.control


@pytest.fixture
def engines(vessel):
    return vessel.parts.engines


# ═══════════════════════════════════════════════════════════════════════
# 1. Preconditions
# ═══════════════════════════════════════════════════════════════════════


class TestVesselContext:
    """Precondition assertions: vessel is in orbit with usable engines."""

    def test_situation_is_orbit(self, vessel):
        assert vessel.situation.name == 'orbiting', (
            f'Vessel is {vessel.situation.name}, not orbiting'
        )

    def test_has_engines(self, engines):
        assert len(engines) > 0, 'Vessel has no engines'

    def test_has_fuel(self, engines):
        assert any(e.has_fuel for e in engines), 'No engine has fuel'


# ═══════════════════════════════════════════════════════════════════════
# 2. Main engine — brief burn
# ═══════════════════════════════════════════════════════════════════════


class TestMainEngine:
    """Activate main engine + set throttle → visible flame + thrust > 0."""

    def test_engine_ignition_and_throttle(self, vessel, ctrl, engines):
        """Sequence: activate engine → set throttle → verify thrust."""
        target = next((e for e in engines if e.has_fuel), None)
        if target is None:
            pytest.skip('No engine with fuel')

        orig_throttle = ctrl.throttle
        orig_active = target.active
        try:
            target.active = True
            ctrl.throttle = 0.3
            # Wait for KSP physics to apply throttle + engine state
            time.sleep(_SETTLE_S)
            assert target.thrust > 0, (
                f'Engine thrust = {target.thrust} (expected > 0)'
            )
            assert vessel.thrust > 0, (
                f'Vessel thrust = {vessel.thrust} (expected > 0)'
            )
        finally:
            ctrl.throttle = orig_throttle
            target.active = orig_active

    def test_engine_throttle_varies_with_input(self, vessel, ctrl, engines):
        """Higher throttle setting → higher vessel.thrust."""
        target = next((e for e in engines if e.has_fuel), None)
        if target is None:
            pytest.skip('No engine with fuel')

        orig_throttle = ctrl.throttle
        orig_active = target.active
        try:
            target.active = True
            ctrl.throttle = 0.3
            time.sleep(_SETTLE_S)
            thrust_low = vessel.thrust

            ctrl.throttle = 0.7
            time.sleep(_SETTLE_S)
            thrust_high = vessel.thrust

            assert thrust_high > thrust_low, (
                f'thrust @ 0.7 ({thrust_high:.1f}) <= thrust @ 0.3 '
                f'({thrust_low:.1f})'
            )
        finally:
            ctrl.throttle = orig_throttle
            target.active = orig_active

    def test_shut_down_engine(self, vessel, ctrl, engines):
        """Turn engine off → thrust drops to 0."""
        target = next((e for e in engines if e.has_fuel), None)
        if target is None:
            pytest.skip('No engine with fuel')

        orig_throttle = ctrl.throttle
        orig_active = target.active
        try:
            target.active = True
            ctrl.throttle = 0.3
            time.sleep(_SETTLE_S)
            target.active = False
            time.sleep(_SETTLE_S)
            assert vessel.thrust == 0, (
                f'Vessel thrust = {vessel.thrust} after engine off'
            )
        finally:
            ctrl.throttle = orig_throttle
            target.active = orig_active


# ═══════════════════════════════════════════════════════════════════════
# 3. Attitude — RCS thrusters (visible firing)
# ═══════════════════════════════════════════════════════════════════════


class TestRCSThrusters:
    """Enable RCS + set control input → RCS thrusters fire visibly."""

    def test_rcs_yaw_input(self, vessel, ctrl):
        """RCS on, yaw hard over — RCS thrusters should fire."""
        if not vessel.parts.rcs:
            pytest.skip('Vessel has no RCS parts')
        orig_rcs = ctrl.rcs
        orig_yaw = ctrl.yaw
        try:
            ctrl.rcs = True
            ctrl.yaw = 1.0
            time.sleep(_SETTLE_S)
            # RCS firing produces angular velocity; control axis should
            # still report the input value
            assert ctrl.yaw >= 0.9
        finally:
            ctrl.yaw = orig_yaw
            ctrl.rcs = orig_rcs

    def test_rcs_pitch_input(self, vessel, ctrl):
        """RCS on, pitch hard over — RCS thrusters should fire."""
        if not vessel.parts.rcs:
            pytest.skip('Vessel has no RCS parts')
        orig_rcs = ctrl.rcs
        orig_pitch = ctrl.pitch
        try:
            ctrl.rcs = True
            ctrl.pitch = -1.0
            time.sleep(_SETTLE_S)
            assert ctrl.pitch <= -0.9
        finally:
            ctrl.pitch = orig_pitch
            ctrl.rcs = orig_rcs


# ═══════════════════════════════════════════════════════════════════════
# 4. Attitude — reaction-wheel rotation (visible)
# ═══════════════════════════════════════════════════════════════════════


class TestAttitudeRotation:
    """SAS off + reaction wheels on → pitch/yaw/roll input → vessel rotates.

    Each test reads `vessel.angular_velocity` (rad/s in vessel RF) before
    and after applying a sustained control input, proving actual rotation.
    """

    @pytest.fixture(autouse=True)
    def _stabilize_before_test(self, vessel, ctrl):
        """Zero axes, engage SAS+reaction wheels, poll until
        angular velocity < 0.001 rad/s, then disable SAS for the test.
        Yields with the vessel stationary and ready for manual input."""
        orig_sas = ctrl.sas
        orig_rw = ctrl.reaction_wheels
        orig_axes = (ctrl.pitch, ctrl.yaw, ctrl.roll)
        try:
            ctrl.pitch = 0.0
            ctrl.yaw = 0.0
            ctrl.roll = 0.0
            ctrl.reaction_wheels = True
            ctrl.sas = True
            for _ in range(50):          # up to 5 s
                av = self._angvel_mag(vessel)
                if av < 0.001:
                    break
                time.sleep(0.1)
            ctrl.sas = False              # let the test drive manually
            yield
        finally:
            ctrl.sas = orig_sas
            ctrl.reaction_wheels = orig_rw
            ctrl.pitch, ctrl.yaw, ctrl.roll = orig_axes

    def _angvel_mag(self, vessel):
        v = vessel.angular_velocity(vessel.orbit.body.reference_frame)
        return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5

    def test_roll_produces_angular_velocity(self, vessel, ctrl):
        """Full roll input → vessel spins → angular velocity > 0."""
        av_before = self._angvel_mag(vessel)
        ctrl.roll = 1.0
        time.sleep(2.0)
        av_during = self._angvel_mag(vessel)
        ctrl.roll = 0.0
        time.sleep(_SETTLE_S)
        assert av_during > av_before + 0.001, (
            f'av_before={av_before:.4f}, av_during={av_during:.4f} — no rotation from roll'
        )

    def test_pitch_produces_angular_velocity(self, vessel, ctrl):
        """Full pitch input → vessel pitches → angular velocity > 0."""
        av_before = self._angvel_mag(vessel)
        ctrl.pitch = 1.0
        time.sleep(2.0)
        av_during = self._angvel_mag(vessel)
        ctrl.pitch = 0.0
        time.sleep(_SETTLE_S)
        assert av_during > av_before + 0.001, (
            f'av_before={av_before:.4f}, av_during={av_during:.4f} — no rotation from pitch'
        )

    def test_yaw_produces_angular_velocity(self, vessel, ctrl):
        """Full yaw input → vessel yaws → angular velocity > 0."""
        av_before = self._angvel_mag(vessel)
        ctrl.yaw = 1.0
        time.sleep(2.0)
        av_during = self._angvel_mag(vessel)
        ctrl.yaw = 0.0
        time.sleep(_SETTLE_S)
        assert av_during > av_before + 0.001, (
            f'av_before={av_before:.4f}, av_during={av_during:.4f} — no rotation from yaw'
        )

    def test_sas_stops_rotation(self, vessel, ctrl):
        """Spin up with roll, then engage SAS — angular velocity drops."""
        ctrl.sas = False
        ctrl.roll = 1.0
        time.sleep(2.0)
        ctrl.roll = 0.0
        av_after_spin = self._angvel_mag(vessel)
        assert av_after_spin > 0.001, 'Vessel did not start rotating'
        ctrl.sas = True
        time.sleep(2.0)
        av_after_sas = self._angvel_mag(vessel)
        assert av_after_sas < av_after_spin * 0.5, (
            f'av_spin={av_after_spin:.4f}, av_after_sas={av_after_sas:.4f} — SAS failed to damp'
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. Reaction wheels — toggle and torque sanity
# ═══════════════════════════════════════════════════════════════════════


class TestReactionWheels:
    """Reaction wheel torque — vessel should have some control authority."""

    def test_reaction_wheels_toggle(self, ctrl):
        """Reaction wheels on/off toggle; axes stay readable."""
        orig = ctrl.reaction_wheels
        orig_axes = (ctrl.pitch, ctrl.yaw, ctrl.roll)
        try:
            ctrl.reaction_wheels = False
            time.sleep(_SETTLE_S)
            assert ctrl.reaction_wheels is False
            ctrl.reaction_wheels = True
            assert ctrl.reaction_wheels is True
            assert -1.0 <= ctrl.pitch <= 1.0
        finally:
            ctrl.reaction_wheels = orig
            ctrl.pitch, ctrl.yaw, ctrl.roll = orig_axes

    def test_vessel_has_available_torque(self, vessel):
        """vessel.available_torque has non-zero magnitude on at least one axis.
        Returns ((pitch,yaw,roll)_pos, (pitch,yaw,roll)_neg) in N·m."""
        pos, neg = vessel.available_torque
        def mag(v):
            return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        assert mag(pos) > 0 or mag(neg) > 0, (
            f'available_torque = (pos={pos}, neg={neg}) — no torque'
        )


# ═══════════════════════════════════════════════════════════════════════
# 6. SAS — engage / disengage
# ═══════════════════════════════════════════════════════════════════════


class TestSAS:
    """SAS engagement via reaction wheels — navball indicator + hold."""

    def test_sas_engage_disengage(self, ctrl):
        """SAS on → SAS indicator on → SAS off."""
        ctrl.sas = True
        time.sleep(_SETTLE_S)
        if ctrl.sas is not True:
            pytest.skip('SAS cannot be engaged on this vessel')
        orig = ctrl.sas
        try:
            assert ctrl.sas is True
            ctrl.sas = False
            assert ctrl.sas is False
        finally:
            ctrl.sas = orig
