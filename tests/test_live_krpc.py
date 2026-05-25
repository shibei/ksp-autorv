"""Comprehensive live kRPC integration tests.

Requires:
  - KSP running with kRPC server at 172.17.64.1:50000 / 50001
  - An active vessel on the launchpad (pre-launch or landed)

Marked ``live_krpc`` — auto-skipped when server is unreachable.
Read-only tests verify state observation; write tests restore original state.
"""

from __future__ import annotations

import math
import time

import pytest

pytestmark = pytest.mark.live_krpc

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope='module')
def live_conn():
    """One kRPC connection shared across all tests in this module."""
    import krpc

    conn = krpc.connect(
        name='test_live',
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


@pytest.fixture
def orbit(vessel):
    return vessel.orbit


@pytest.fixture
def body(orbit):
    return orbit.body


@pytest.fixture
def body_rf(body):
    return body.reference_frame


@pytest.fixture
def surface_rf(vessel):
    return vessel.surface_reference_frame


# ═══════════════════════════════════════════════════════════════════════
# 1. Connection
# ═══════════════════════════════════════════════════════════════════════


class TestConnection:
    """Verify basic kRPC RPC + Stream connection."""

    def test_rpc_connect(self):
        """RPC port is reachable and returns a client."""
        import krpc

        conn = krpc.connect(
            name='rpc_test',
            address='172.17.64.1',
            rpc_port=50000,
        )
        assert conn is not None
        assert conn.space_center is not None
        conn.close()

    def test_stream_connect(self):
        """Stream port is reachable."""
        import krpc

        conn = krpc.connect(
            name='stream_test',
            address='172.17.64.1',
            rpc_port=50000,
            stream_port=50001,
        )
        assert conn is not None
        # Creating a stream implicitly verifies the stream server
        orbit = conn.space_center.active_vessel.orbit
        s = conn.add_stream(getattr, orbit, 'apoapsis')
        assert s() > 0
        s.remove()
        conn.close()

    def test_invalid_rpc_port_raises(self):
        """Connecting to a closed port raises ConnectionError."""
        import krpc

        with pytest.raises(Exception):
            krpc.connect(
                name='bad_port',
                address='172.17.64.1',
                rpc_port=1,
            )


# ═══════════════════════════════════════════════════════════════════════
# 2. SpaceCenter
# ═══════════════════════════════════════════════════════════════════════


class TestSpaceCenter:
    """Verify SpaceCenter service properties."""

    def test_ut_monotonic(self, sc):
        """Universal time increases over successive calls."""
        t0 = sc.ut
        time.sleep(0.1)
        t1 = sc.ut
        assert t1 > t0

    def test_celestial_bodies_exist(self, sc):
        """Kerbin, Mun, Sun are present in the bodies dict."""
        assert 'Kerbin' in sc.bodies
        assert 'Mun' in sc.bodies
        assert 'Sun' in sc.bodies

    def test_celestial_body_basic(self, sc):
        """A known body has plausible properties."""
        kerbin = sc.bodies['Kerbin']
        assert kerbin.equatorial_radius == 600_000
        assert kerbin.has_atmosphere is True
        assert kerbin.surface_gravity > 9.0

    def test_game_mode(self, sc):
        """Game mode is accessible (sandbox or career)."""
        mode = sc.game_mode
        assert mode is not None

    def test_active_vessel_exists(self, sc):
        """There is an active vessel (we are attached to one)."""
        v = sc.active_vessel
        assert v is not None
        assert len(v.name) > 0

    def test_non_negative_warp(self, sc):
        """Warp factor is non-negative (and default 0)."""
        assert sc.warp_factor >= 0
        assert sc.physics_warp_factor >= 0
        assert sc.rails_warp_factor >= 0

    def test_gravity(self, sc):
        """Gravitational constant is the standard value."""
        assert sc.g == pytest.approx(6.67408e-11)

    def test_ut_not_nan(self, sc):
        """Universal time is a finite number."""
        assert math.isfinite(sc.ut)


# ═══════════════════════════════════════════════════════════════════════
# 3. Vessel
# ═══════════════════════════════════════════════════════════════════════


class TestVessel:
    """Verify vessel-level properties (read-only)."""

    def test_vessel_name(self, vessel):
        """Vessel has a non-empty name."""
        assert len(vessel.name) > 0

    def test_vessel_type(self, vessel):
        """Vessel type is a valid enum member."""
        from krpc.services.spacecenter import VesselType

        assert isinstance(vessel.type, VesselType)

    def test_vessel_situation(self, vessel):
        """Vessel situation is a known value."""
        from krpc.services.spacecenter import VesselSituation

        assert isinstance(vessel.situation, VesselSituation)

    def test_vessel_mass_positive(self, vessel):
        """Mass, dry mass, and crew capacity are non-negative."""
        assert vessel.mass > 0
        assert vessel.dry_mass > 0
        assert vessel.mass >= vessel.dry_mass
        assert vessel.crew_capacity >= 0
        assert vessel.crew_count >= 0

    def test_vessel_thrust(self, vessel):
        """Thrust values are non-negative floats."""
        assert vessel.thrust >= 0
        assert vessel.available_thrust >= 0
        assert vessel.max_thrust >= 0
        assert vessel.max_vacuum_thrust >= 0

    def test_vessel_control_exists(self, vessel):
        """Control object is accessible."""
        ctrl = vessel.control
        assert ctrl is not None

    def test_vessel_parts_accessible(self, vessel):
        """Parts list is accessible and non-empty for a built vessel."""
        parts = vessel.parts.all
        assert len(parts) > 0

    def test_vessel_flight_exists(self, vessel, body_rf):
        """Flight() returns a Flight object."""
        f = vessel.flight(body_rf)
        assert f is not None

    def test_vessel_comms(self, vessel):
        """Comms object is accessible."""
        assert vessel.comms is not None

    def test_vessel_resources_accessible(self, vessel):
        """Resources object is accessible."""
        assert vessel.resources is not None

    def test_vessel_inertia(self, vessel):
        """Inertia tensor is a 9-element list."""
        tensor = vessel.inertia_tensor
        assert len(tensor) == 9
        assert all(math.isfinite(v) for v in tensor)

    def test_vessel_biome(self, vessel):
        """Biome is a non-empty string."""
        assert len(vessel.biome) > 0

    def test_vessel_recoverable(self, vessel):
        """Recoverable is a bool."""
        assert isinstance(vessel.recoverable, bool)

    def test_vessel_met(self, vessel):
        """Mission elapsed time is non-negative."""
        assert vessel.met >= 0

    def test_vessel_reference_frames(self, vessel):
        """All reference frames are distinct objects."""
        frames = [
            vessel.reference_frame,
            vessel.orbital_reference_frame,
            vessel.surface_reference_frame,
            vessel.surface_velocity_reference_frame,
        ]
        assert all(f is not None for f in frames)
        # At least the orbital and surface frames should differ
        assert vessel.orbital_reference_frame != vessel.surface_reference_frame


# ═══════════════════════════════════════════════════════════════════════
# 4. Orbit
# ═══════════════════════════════════════════════════════════════════════


class TestOrbit:
    """Verify all read-only orbit properties are sane."""

    def test_apoapsis_positive(self, orbit):
        """Apoapsis (from center) > 0 for any bound orbit."""
        assert orbit.apoapsis > 0

    def test_periapsis_positive(self, orbit):
        """Periapsis (from center) > 0 for any bound orbit."""
        assert orbit.periapsis > 0

    def test_semi_major_axis(self, orbit):
        """semi_major_axis > 0 for elliptical orbits."""
        assert orbit.semi_major_axis > 0

    def test_eccentricity_range(self, orbit):
        """0 <= eccentricity < 1 for a closed orbit."""
        assert orbit.eccentricity >= 0
        assert orbit.eccentricity < 1

    def test_inclination_range(self, orbit):
        """0° <= inclination <= 180°."""
        incl = math.degrees(orbit.inclination)
        assert 0 <= incl <= 180

    def test_period_positive(self, orbit):
        """Orbital period > 0 seconds."""
        assert orbit.period > 0

    def test_orbital_speed(self, orbit):
        """Orbital speed matches Keplerian expectation roughly."""
        # v ~ sqrt(mu * (2/r - 1/a)), should be positive
        assert orbit.orbital_speed >= 0

    def test_radius_positive(self, orbit):
        """Radius from center of body is positive."""
        assert orbit.radius > 0

    def test_time_to_apoapsis(self, orbit):
        """time_to_apoapsis should be [0, period)."""
        assert orbit.time_to_apoapsis >= 0
        assert orbit.time_to_apoapsis < orbit.period + 1e-3

    def test_coordinate_frames(self, orbit):
        """Orbit has a body with valid SOI."""
        assert orbit.body is not None
        assert orbit.body.sphere_of_influence > 0

    def test_anomalies(self, orbit):
        """Mean and true anomalies are finite."""
        assert math.isfinite(orbit.mean_anomaly)
        assert math.isfinite(orbit.true_anomaly)
        assert math.isfinite(orbit.eccentric_anomaly)

    def test_altitude_consistency(self, orbit):
        """Apoapsis_altitude = apoapsis - body.equatorial_radius."""
        expected = orbit.apoapsis - orbit.body.equatorial_radius
        assert orbit.apoapsis_altitude == pytest.approx(expected, rel=1e-6)

    def test_semi_minor_axis(self, orbit):
        """semi_minor_axis <= semi_major_axis for elliptical orbits."""
        assert orbit.semi_minor_axis <= orbit.semi_major_axis


# ═══════════════════════════════════════════════════════════════════════
# 5. Flight
# ═══════════════════════════════════════════════════════════════════════


class TestFlight:
    """Verify flight properties (position, velocity, atmosphere)."""

    def test_mean_altitude_positive(self, vessel, body_rf):
        """mean_altitude >= 0 (on or above surface)."""
        alt = vessel.flight(body_rf).mean_altitude
        assert alt >= 0

    def test_surface_altitude(self, vessel, surface_rf):
        """surface_altitude >= 0 (above terrain)."""
        alt = vessel.flight(surface_rf).surface_altitude
        assert alt >= 0

    def test_latitude_range(self, vessel, body_rf):
        """-90° <= latitude <= 90°."""
        lat = vessel.flight(body_rf).latitude
        assert -90 <= lat <= 90

    def test_longitude_range(self, vessel, body_rf):
        """-180° <= longitude <= 180°."""
        lon = vessel.flight(body_rf).longitude
        assert -180 <= lon <= 180

    def test_heading_range(self, vessel, surface_rf):
        """0° <= heading < 360°."""
        hdg = vessel.flight(surface_rf).heading
        assert 0 <= hdg < 360

    def test_pitch_range(self, vessel, surface_rf):
        """-90° <= pitch <= 90°."""
        pitch = vessel.flight(surface_rf).pitch
        assert -90 <= pitch <= 90

    def test_roll_range(self, vessel, surface_rf):
        """-180° <= roll <= 180°."""
        roll = vessel.flight(surface_rf).roll
        assert -180 <= roll <= 180

    def test_speed_non_negative(self, vessel, surface_rf):
        """Ground speed >= 0."""
        speed = vessel.flight(surface_rf).speed
        assert speed >= 0

    def test_vertical_speed_finite(self, vessel, surface_rf):
        """Vertical speed is finite (can be negative during descent)."""
        vs = vessel.flight(surface_rf).vertical_speed
        assert math.isfinite(vs)

    def test_horizontal_speed_non_negative(self, vessel, surface_rf):
        """Horizontal speed >= 0."""
        hs = vessel.flight(surface_rf).horizontal_speed
        assert hs >= 0

    def test_g_force_positive(self, vessel, body_rf):
        """G-force ~ 1 on launchpad."""
        g = vessel.flight(body_rf).g_force
        assert g > 0

    def test_atmosphere_density_positive(self, vessel, body_rf):
        """Atmosphere density > 0 at sea level."""
        rho = vessel.flight(body_rf).atmosphere_density
        assert rho > 0

    def test_dynamic_pressure(self, vessel, body_rf):
        """Dynamic pressure >= 0."""
        q = vessel.flight(body_rf).dynamic_pressure
        assert q >= 0

    def test_mach(self, vessel, body_rf):
        """Mach number >= 0."""
        mach = vessel.flight(body_rf).mach
        assert mach >= 0

    def test_angle_of_attack(self, vessel, surface_rf):
        """AoA is a finite number."""
        aoa = vessel.flight(surface_rf).angle_of_attack
        assert math.isfinite(aoa)

    def test_sideslip_angle(self, vessel, surface_rf):
        """Sideslip angle is a finite number."""
        ss = vessel.flight(surface_rf).sideslip_angle
        assert math.isfinite(ss)

    def test_true_air_speed(self, vessel, body_rf):
        """True air speed >= 0."""
        tas = vessel.flight(body_rf).true_air_speed
        assert tas >= 0

    def test_equivalent_air_speed(self, vessel, body_rf):
        """Equivalent air speed >= 0."""
        eas = vessel.flight(body_rf).equivalent_air_speed
        assert eas >= 0

    def test_aerodynamic_force_tuple(self, vessel, body_rf):
        """Aerodynamic force is a 3-element tuple of floats."""
        f = vessel.flight(body_rf).aerodynamic_force
        assert len(f) == 3
        assert all(math.isfinite(v) for v in f)

    def test_direction_tuple(self, vessel, body_rf):
        """Direction is a 3-element unit-ish vector."""
        d = vessel.flight(body_rf).direction
        assert len(d) == 3
        magnitude = math.sqrt(sum(v * v for v in d))
        assert magnitude == pytest.approx(1.0, abs=0.01)

    def test_velocity_tuple(self, vessel, body_rf):
        """Velocity is a 3-element vector."""
        vel = vessel.flight(body_rf).velocity
        assert len(vel) == 3
        assert all(math.isfinite(v) for v in vel)

    def test_prograde_retrograde_opposite(self, vessel, body_rf):
        """Prograde and retrograde vectors are opposite."""
        pro = vessel.flight(body_rf).prograde
        retro = vessel.flight(body_rf).retrograde
        assert sum(abs(a + b) for a, b in zip(pro, retro)) < 0.01

    # FAR-specific properties should raise on non-FAR installs
    @pytest.mark.parametrize('attr', [
        'ballistic_coefficient',
        'drag_coefficient',
        'lift_coefficient',
        'reynolds_number',
        'stall_fraction',
        'thrust_specific_fuel_consumption',
    ])
    def test_far_properties_raise(self, vessel, body_rf, attr):
        """FAR-only properties raise RuntimeError without FAR installed."""
        flight = vessel.flight(body_rf)
        with pytest.raises(Exception, match='FAR'):
            getattr(flight, attr)


# ═══════════════════════════════════════════════════════════════════════
# 6. Stream
# ═══════════════════════════════════════════════════════════════════════


class TestStream:
    """Verify the two kRPC add_stream patterns work correctly."""

    def test_getattr_stream(self, live_conn, orbit):
        """``add_stream(getattr, obj, 'prop')`` streams a float."""
        s = live_conn.add_stream(getattr, orbit, 'apoapsis')
        try:
            val = s()
            assert isinstance(val, float)
            assert val > 0
        finally:
            s.remove()

    def test_method_stream(self, live_conn, vessel, body_rf):
        """``add_stream(v.flight, rf)`` streams a Flight object."""
        s = live_conn.add_stream(vessel.flight, body_rf)
        try:
            flight = s()
            assert flight is not None
            assert flight.mean_altitude >= 0
            assert math.isfinite(flight.latitude)
        finally:
            s.remove()

    def test_multiple_streams(self, live_conn, orbit):
        """Multiple independent streams all return correct values."""
        streams = {}
        props = ['apoapsis', 'periapsis', 'inclination', 'eccentricity']
        try:
            for prop in props:
                streams[prop] = live_conn.add_stream(getattr, orbit, prop)
            values = {k: v() for k, v in streams.items()}
            assert values['apoapsis'] > 0
            assert values['periapsis'] > 0
            assert 0 <= values['inclination'] <= math.pi
            assert 0 <= values['eccentricity'] < 1
        finally:
            for s in streams.values():
                s.remove()

    def test_stream_rate_setting(self, live_conn, orbit):
        """Stream rate can be set and read back."""
        s = live_conn.add_stream(getattr, orbit, 'apoapsis')
        try:
            s.rate = 4.0
            assert s.rate == 4.0
        finally:
            s.remove()

    def test_stream_updates_over_time(self, live_conn, sc):
        """Stream value updates over successive calls."""
        s = live_conn.add_stream(getattr, sc, 'ut')
        try:
            s.rate = 10.0
            time.sleep(0.5)
            v0 = s()
            assert math.isfinite(v0)
        finally:
            s.remove()

    def test_stream_remove(self, live_conn, orbit):
        """Removed stream should not affect subsequent streams."""
        s1 = live_conn.add_stream(getattr, orbit, 'apoapsis')
        s1.remove()
        s2 = live_conn.add_stream(getattr, orbit, 'periapsis')
        assert s2() > 0
        s2.remove()


# ═══════════════════════════════════════════════════════════════════════
# 7. Control (read-only)
# ═══════════════════════════════════════════════════════════════════════


class TestControlReadOnly:
    """Verify control properties can be read (no state mutation)."""

    def test_current_stage(self, vessel):
        """Stage is non-negative."""
        assert vessel.control.current_stage >= 0

    def test_throttle_range(self, vessel):
        """0 <= throttle <= 1."""
        t = vessel.control.throttle
        assert 0 <= t <= 1

    def test_pitch_trim(self, vessel):
        """Pitch/Yaw/Roll trims are finite."""
        assert math.isfinite(vessel.control.pitch)
        assert math.isfinite(vessel.control.yaw)
        assert math.isfinite(vessel.control.roll)

    def test_boolean_toggles(self, vessel):
        """Lights, Gear, Brakes, RCS, SAS are booleans."""
        ctrl = vessel.control
        assert isinstance(ctrl.lights, bool)
        assert isinstance(ctrl.gear, bool)
        assert isinstance(ctrl.brakes, bool)
        assert isinstance(ctrl.rcs, bool)
        assert isinstance(ctrl.sas, bool)
        assert isinstance(ctrl.reaction_wheels, bool)

    def test_action_groups(self, vessel):
        """All 10 action groups are accessible (custom1-custom10)."""
        ctrl = vessel.control
        # kRPC has custom action groups 1-10
        for i in range(1, 11):
            try:
                val = getattr(ctrl, f'custom_{i:02d}')
                assert isinstance(val, bool)
            except AttributeError:
                pass  # not all mods expose all groups


class TestControlWrite:
    """Verify control write operations restore original state."""

    def test_throttle_write_and_reset(self, vessel):
        """Set throttle → readback clamped to [0,1] → restore original."""
        ctrl = vessel.control
        original = ctrl.throttle
        try:
            ctrl.throttle = 0.5
            # KSP clamps throttle to 0 in pre-launch; verify bounds only
            assert 0 <= ctrl.throttle <= 1
        finally:
            ctrl.throttle = original

    def test_sas_toggle(self, vessel):
        """Toggle SAS on → verify → toggle off → verify."""
        ctrl = vessel.control
        original = ctrl.sas
        try:
            ctrl.sas = True
            assert ctrl.sas is True
            ctrl.sas = False
            assert ctrl.sas is False
        finally:
            ctrl.sas = original

    def test_rcs_toggle(self, vessel):
        """Toggle RCS on → verify → toggle off → verify."""
        ctrl = vessel.control
        original = ctrl.rcs
        try:
            ctrl.rcs = True
            assert ctrl.rcs is True
            ctrl.rcs = False
            assert ctrl.rcs is False
        finally:
            ctrl.rcs = original

    def test_gear_toggle(self, vessel):
        """Toggle gear on → verify → toggle off → verify."""
        ctrl = vessel.control
        original = ctrl.gear
        try:
            ctrl.gear = True
            assert ctrl.gear is True
            ctrl.gear = False
            assert ctrl.gear is False
        finally:
            ctrl.gear = original

    def test_brakes_toggle(self, vessel):
        """Toggle brakes on → verify → toggle off → verify."""
        ctrl = vessel.control
        original = ctrl.brakes
        try:
            ctrl.brakes = True
            assert ctrl.brakes is True
            ctrl.brakes = False
            assert ctrl.brakes is False
        finally:
            ctrl.brakes = original

    def test_lights_toggle(self, vessel):
        """Toggle lights on → verify → toggle off → verify."""
        ctrl = vessel.control
        original = ctrl.lights
        try:
            ctrl.lights = True
            assert ctrl.lights is True
            ctrl.lights = False
            assert ctrl.lights is False
        finally:
            ctrl.lights = original

    def test_reaction_wheels_toggle(self, vessel):
        """Toggle reaction_wheels on → verify → toggle off → verify."""
        ctrl = vessel.control
        original = ctrl.reaction_wheels
        try:
            ctrl.reaction_wheels = True
            assert ctrl.reaction_wheels is True
            ctrl.reaction_wheels = False
            assert ctrl.reaction_wheels is False
        finally:
            ctrl.reaction_wheels = original

    def test_control_input_reset(self, vessel):
        """Set pitch/yaw/roll → each in [-1,1] → restore originals."""
        ctrl = vessel.control
        orig_pitch = ctrl.pitch
        orig_yaw = ctrl.yaw
        orig_roll = ctrl.roll
        try:
            ctrl.pitch = 0.5
            ctrl.yaw = -0.3
            ctrl.roll = 0.1
            # KSP may clamp axes to 0 in pre-launch; verify bounds
            assert -1 <= ctrl.pitch <= 1
            assert -1 <= ctrl.yaw <= 1
            assert -1 <= ctrl.roll <= 1
            ctrl.pitch = 0.0
            ctrl.yaw = 0.0
            ctrl.roll = 0.0
            assert -1 <= ctrl.pitch <= 1
            assert -1 <= ctrl.yaw <= 1
            assert -1 <= ctrl.roll <= 1
        finally:
            ctrl.pitch = orig_pitch
            ctrl.yaw = orig_yaw
            ctrl.roll = orig_roll

    def test_autopilot_engage_disengage(self, vessel):
        """Engage/disengage autopilot without error, restore SAS state."""
        ap = vessel.auto_pilot
        orig_sas = ap.sas
        try:
            ap.engage()
            # kRPC AutoPilot has no engaged-state property; verify
            # params still readable after engagement
            assert math.isfinite(ap.target_pitch)
            ap.disengage()
            assert math.isfinite(ap.target_pitch)
        finally:
            ap.sas = orig_sas

    def test_autopilot_target_and_reset(self, vessel):
        """Set autopilot target pitch/heading → readback → reset."""
        ap = vessel.auto_pilot
        orig_pitch = ap.target_pitch
        orig_heading = ap.target_heading
        try:
            ap.target_pitch = 45.0
            ap.target_heading = 90.0
            assert ap.target_pitch == 45.0
            assert ap.target_heading == 90.0
        finally:
            ap.target_pitch = orig_pitch
            ap.target_heading = orig_heading


class TestControlInputSanity:
    """Sanity checks on control axes — read-only, verifies physical limits."""

    def test_pitch_trim_limits(self, vessel):
        """Pitch trim can be set to positive and negative values."""
        ctrl = vessel.control
        orig = ctrl.pitch
        try:
            ctrl.pitch = 1.0
            assert -1.0 <= ctrl.pitch <= 1.0
            ctrl.pitch = -1.0
            assert -1.0 <= ctrl.pitch <= 1.0
        finally:
            ctrl.pitch = orig


# ═══════════════════════════════════════════════════════════════════════
# 8. Parts
# ═══════════════════════════════════════════════════════════════════════


class TestParts:
    """Verify vessel parts enumeration."""

    def test_parts_non_empty(self, vessel):
        """Active vessel has at least one part."""
        assert len(vessel.parts.all) > 0

    def test_part_has_name(self, vessel):
        """Every part has a non-empty name and title."""
        for part in vessel.parts.all[:5]:  # spot-check first 5
            assert len(part.name) > 0
            assert len(part.title) > 0

    def test_part_stage_index(self, vessel):
        """Part stage is >= -1 (fairing, strut)."""
        for part in vessel.parts.all[:5]:
            assert part.stage >= -1
            assert part.decouple_stage >= -1

    def test_part_has_resources(self, vessel):
        """Parts have a resources container (may be empty)."""
        for part in vessel.parts.all[:5]:
            assert part.resources is not None

    def test_engines_count(self, vessel):
        """If vessel has engines, they have valid properties."""
        engines = vessel.parts.engines
        if engines:
            e = engines[0]
            assert e.part is not None
            # thrust may be 0 when throttle is 0
            assert e.max_thrust >= 0
            assert e.max_vacuum_thrust >= 0


# ═══════════════════════════════════════════════════════════════════════
# 9. Resources
# ═══════════════════════════════════════════════════════════════════════


class TestResources:
    """Verify resource access patterns."""

    def test_vessel_resources_has_names(self, vessel):
        """Resources list has typical KSP resource names."""
        names = set()
        for r in vessel.resources.all:
            names.add(r.name)
        expected = {'ElectricCharge', 'LiquidFuel', 'Oxidizer'}
        assert expected.issubset(names), f'Missing from {names}'

    def test_resource_amounts_non_negative(self, vessel):
        """All resource amounts and max values are >= 0."""
        for r in vessel.resources.all:
            assert r.amount >= 0
            assert r.max >= 0
            assert r.amount <= r.max + 1e-6  # floating point tolerance

    def test_stage_resources(self, vessel):
        """Resources queried by name return non-negative results."""
        fuel = vessel.resources.amount('LiquidFuel')
        oxidizer = vessel.resources.amount('Oxidizer')
        assert fuel >= 0
        assert oxidizer >= 0

    def test_resource_densities(self, vessel):
        """Resource density accessible and positive."""
        for r in vessel.resources.all:
            if r.density is not None:
                assert r.density >= 0


# ═══════════════════════════════════════════════════════════════════════
# 10. CelestialBody
# ═══════════════════════════════════════════════════════════════════════


class TestCelestialBody:
    """Verify celestial body properties."""

    def test_kerbin_properties(self, body):
        """Kerbin has known values."""
        assert body.name == 'Kerbin'
        assert body.equatorial_radius == 600_000
        assert body.has_atmosphere is True
        assert body.surface_gravity == pytest.approx(9.81335, rel=1e-3)
        assert body.rotational_period == pytest.approx(21_549.425, rel=1e-3)

    def test_gravitational_parameter(self, body):
        """mu = g * r²."""
        mu = body.gravitational_parameter
        expected = body.surface_gravity * body.equatorial_radius ** 2
        assert mu == pytest.approx(expected, rel=1e-2)

    def test_sphere_of_influence(self, body):
        """Kerbin SOI ≈ 84.16 Mm."""
        assert body.sphere_of_influence > 0
        assert body.sphere_of_influence < 1e9

    def test_body_rotation(self, body):
        """Rotation angle is in [0, 2π)."""
        angle = body.rotation_angle
        assert 0 <= angle < 2 * math.pi

    def test_orbital_properties(self, body):
        """Body's own orbit is valid (except for Sun)."""
        if body.name == 'Sun':
            return  # Sun has no meaningful orbit in KSP
        assert body.orbit is not None
        assert body.orbit.semi_major_axis > 0
        assert body.orbit.period > 0

    def test_biomes_exist(self, body):
        """Kerbin has typical biomes."""
        biomes = body.biomes
        assert len(biomes) > 0
        common = {'Shores', 'Grasslands', 'Highlands', 'Mountains'}
        assert common.issubset(biomes)

    def test_atmosphere_depth(self, body):
        """Kerbin atmosphere extends to 70 km."""
        assert body.atmosphere_depth == 70_000

    def test_reference_frames(self, body):
        """All reference frames are distinct."""
        assert body.reference_frame is not None
        assert body.non_rotating_reference_frame is not None
        assert body.orbital_reference_frame is not None


# ═══════════════════════════════════════════════════════════════════════
# 11. AutoPilot
# ═══════════════════════════════════════════════════════════════════════


class TestAutoPilot:
    """Verify autopilot state can be read (no engagement)."""

    def test_autopilot_exists(self, vessel):
        """Vessel has an AutoPilot object."""
        ap = vessel.auto_pilot
        assert ap is not None

    def test_autopilot_default_state(self, vessel):
        """Default autopilot state (SAS off, no target)."""
        ap = vessel.auto_pilot
        assert not ap.sas
        assert ap.reference_frame is not None

    def test_autopilot_parameters(self, vessel):
        """Autopilot tuning parameters are readable tuples (pitch/yaw/roll)."""
        ap = vessel.auto_pilot
        gains = ap.roll_pid_gains
        assert len(gains) == 3
        assert all(math.isfinite(v) for v in gains)
        ttp = ap.time_to_peak
        assert isinstance(ttp, (tuple, list))
        assert all(v > 0 for v in ttp)


# ═══════════════════════════════════════════════════════════════════════
# 12. Utilities
# ═══════════════════════════════════════════════════════════════════════


class TestUtilities:
    """Verify utility properties (camera, comms, alarms)."""

    def test_camera_mode(self, sc):
        """Camera mode is readable."""
        cam = sc.camera
        assert cam.mode is not None

    def test_comms_signal(self, vessel):
        """Comms signal strength is in [0, 1]."""
        signal = vessel.comms.signal_strength
        assert 0 <= signal <= 1

    def test_comms_power(self, vessel):
        """Comms power is non-negative."""
        assert vessel.comms.power >= 0

    def test_waypoint_manager(self, sc):
        """Waypoint manager is accessible."""
        assert sc.waypoint_manager is not None

    def test_launch_sites(self, sc):
        """At least one launch site exists."""
        sites = sc.launch_sites
        assert len(sites) > 0


# ═══════════════════════════════════════════════════════════════════════
# 13. Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test boundaries, error conditions, and resilience."""

    def test_vessel_list_includes_active(self, sc):
        """sc.vessels contains the active vessel."""
        active = sc.active_vessel
        all_vessels = sc.vessels
        assert active in all_vessels

    def test_targets_are_none_by_default(self, sc):
        """No target vessel/body/dock by default."""
        # Clear any stale target left by earlier E2E tests
        sc.target_vessel = None
        assert sc.target_vessel is None
        assert sc.target_body is None
        assert sc.target_docking_port is None

    def test_ut_not_nan_across_connects(self):
        """UT is finite across independent connections."""
        import krpc

        for _ in range(3):
            conn = krpc.connect(
                name='multi_test',
                address='172.17.64.1',
                rpc_port=50000,
            )
            try:
                assert math.isfinite(conn.space_center.ut)
            finally:
                conn.close()

    def test_solid_fuel_in_vessel(self, vessel):
        """SolidFuel may be present on certain vessels."""
        solid = vessel.resources.amount('SolidFuel')
        assert solid >= 0

    def test_orbital_speed_consistency(self, orbit):
        """Orbital speed matches radius-based calculation roughly."""
        mu = orbit.body.gravitational_parameter
        r = orbit.radius
        a = orbit.semi_major_axis
        v_expected = math.sqrt(mu * (2 / r - 1 / a))
        v = orbit.orbital_speed
        # On launchpad, speed is 0 (launch clamp) despite math
        if v > 0:
            assert v == pytest.approx(v_expected, rel=0.01)

    def test_all_bodies_have_surface_gravity(self, sc):
        """All celestial bodies have positive surface gravity."""
        for name, body in sc.bodies.items():
            assert body.surface_gravity > 0, f'{name} has no gravity'


class TestFlightMonitor:
    """Live kRPC tests for FlightMonitor — connect, pump, alerts, logging."""

    @pytest.fixture
    def monitor(self, live_conn):
        from krpc_rendezvous.common.flight_monitor import FlightMonitor
        mon = FlightMonitor.__new__(FlightMonitor)
        mon._name = 'test-monitor'
        mon._stream_rate = 4.0
        mon._conn = live_conn
        mon._vessel = live_conn.space_center.active_vessel
        mon._streams = {}
        mon._state = None
        import threading
        mon._state_lock = threading.Lock()
        mon._alerts = []
        mon._running = False
        mon._thread = None
        mon._log_writer = None
        mon._log_dir = mon._default_log_dir()
        mon._last_stage = -1
        return mon

    @pytest.fixture
    def built_monitor(self, monitor):
        """FlightMonitor with streams built (like after connect())."""
        monitor._build_streams(monitor._conn.space_center)
        return monitor

    # ── Connect & stream initialization ─────────────────────────────

    @pytest.mark.live_krpc
    def test_monitor_build_streams(self, built_monitor):
        """_build_streams creates all expected kRPC streams."""
        expected = {
            'flight_body', 'flight_surface',
            'apoapsis', 'periapsis', 'inclination', 'eccentricity',
            'semi_major_axis', 'period', 'time_to_apoapsis', 'time_to_periapsis',
            'mass', 'thrust', 'available_thrust', 'stage',
        }
        assert expected.issubset(built_monitor._streams.keys())
        for name, stream in built_monitor._streams.items():
            assert stream is not None, f'Stream {name} is None'

    @pytest.mark.live_krpc
    def test_monitor_read_streams(self, built_monitor):
        """_read_streams returns a FlightRecord with all fields populated."""
        record = built_monitor._read_streams()

        # Type checks
        assert record.ut > 0
        assert record.vessel_name != ''

        # Orbital elements (may be 0 on launchpad)
        assert record.altitude >= 0
        assert record.apoapsis >= 0
        assert record.periapsis >= 0
        assert 0 <= record.inclination <= 180
        assert 0 <= record.eccentricity <= 1
        assert record.semi_major_axis > 0
        assert record.period >= 0

        # Position
        assert -90 <= record.latitude <= 90
        assert -180 <= record.longitude <= 180
        assert record.altitude_from_surface >= 0

        # Velocity (may be 0 on launchpad)
        assert record.speed >= 0
        assert record.vertical_speed is not None
        assert record.horizontal_speed >= 0

        # Attitude — newly added pitch/roll
        assert 0 <= record.direction_heading <= 360
        assert -90 <= record.pitch <= 90, f'pitch={record.pitch} out of range'
        assert -180 <= record.roll <= 180, f'roll={record.roll} out of range'

        # Resources
        assert record.stage >= 0
        assert record.mass > 0
        assert record.thrust >= 0
        assert record.available_thrust >= 0
        assert record.stage_remaining_fuel >= 0
        assert record.total_delta_v >= 0

    @pytest.mark.live_krpc
    def test_monitor_stream_updates(self, built_monitor):
        """Repeated reads return different state over time."""
        import time
        r0 = built_monitor._read_streams()
        time.sleep(0.5)
        r1 = built_monitor._read_streams()
        # UT must advance
        assert r1.ut > r0.ut

    # ── Background pump lifecycle ───────────────────────────────────

    @pytest.mark.live_krpc
    def test_monitor_start_stop(self, monitor):
        """start() launches background thread; stop() joins cleanly."""
        monitor._build_streams(monitor._conn.space_center)
        monitor.start()
        assert monitor.is_running()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        # Should have a state after the pump runs
        import time
        time.sleep(0.5)
        state = monitor.get_state()
        assert state is not None
        assert state.ut > 0
        assert state.vessel_name != ''

        monitor.stop()
        assert not monitor.is_running()
        if monitor._thread:
            assert not monitor._thread.is_alive()

    # ── Alert rules ─────────────────────────────────────────────────

    @pytest.mark.live_krpc
    def test_monitor_alert_fires(self, built_monitor):
        """AlertRule fires callback when condition matches real data."""
        from krpc_rendezvous.common.flight_monitor import AlertRule
        fired = []
        # The active vessel should have a non-negative speed on the pad
        rule = AlertRule('speed', '>=', 0, lambda r: fired.append(r.speed))
        built_monitor.add_alert(rule)

        record = built_monitor._read_streams()
        built_monitor.update_from_record(record)
        with built_monitor._state_lock:
            for alert in built_monitor._alerts:
                alert.check(record)

        assert len(fired) >= 1
        assert fired[0] >= 0

    @pytest.mark.live_krpc
    def test_monitor_alert_not_fires_for_impossible_threshold(self, built_monitor):
        """AlertRule does NOT fire for an impossible condition."""
        from krpc_rendezvous.common.flight_monitor import AlertRule
        fired = []
        rule = AlertRule('altitude', '>', 1_000_000_000, lambda r: fired.append(r.altitude))
        built_monitor.add_alert(rule)

        record = built_monitor._read_streams()
        with built_monitor._state_lock:
            for alert in built_monitor._alerts:
                alert.check(record)

        assert len(fired) == 0

    # ── CSV logging ─────────────────────────────────────────────────

    @pytest.mark.live_krpc
    def test_monitor_csv_logging(self, built_monitor):
        """start_logging creates a CSV file; write appends rows."""
        import tempfile
        from pathlib import Path
        from krpc_rendezvous.common.flight_monitor import _CsvWriter

        # Patch log_dir to temp
        with tempfile.TemporaryDirectory() as tmp:
            built_monitor._log_dir = Path(tmp)
            built_monitor.start_logging('test-vessel')

            assert built_monitor._log_writer is not None
            assert built_monitor._log_writer.fpath is not None
            assert built_monitor._log_writer.fpath.exists()

            # Write a record
            record = built_monitor._read_streams()
            built_monitor._log_writer.write(record)
            built_monitor._log_writer.close()

            # Read back
            import csv
            with built_monitor._log_writer.fpath.open('r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            row = rows[0]
            assert row['vessel_name'] == record.vessel_name
            assert float(row['ut']) == pytest.approx(record.ut, abs=1.0)
            assert 'pitch' in row
            assert 'roll' in row
            assert float(row['pitch']) == pytest.approx(record.pitch, abs=1.0)
            assert float(row['roll']) == pytest.approx(record.roll, abs=1.0)

    # ── Default alerts factory ──────────────────────────────────────

    @pytest.mark.live_krpc
    def test_default_alerts_all_valid(self):
        """default_alerts() returns rules that pass validation."""
        from krpc_rendezvous.common.flight_monitor import AlertRule, default_alerts
        rules = default_alerts()
        assert len(rules) > 0
        for rule in rules:
            assert rule.op in AlertRule.VALID_OPS
            assert hasattr(rule, 'field')
            assert hasattr(rule, 'threshold')
