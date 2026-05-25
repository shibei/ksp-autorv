import pytest
from dataclasses import dataclass, fields
from krpc_rendezvous.common.flight_monitor import FlightRecord, AlertRule

def test_flight_record_fields():
    fields_list = [f.name for f in fields(FlightRecord)]
    expected = [
        'ut', 'vessel_name',
        # orbital
        'altitude', 'apoapsis', 'periapsis', 'inclination', 'eccentricity', 'semi_major_axis',
        'period', 'time_to_ap', 'time_to_pe',
        # position
        'latitude', 'longitude', 'altitude_from_surface',
        # velocity
        'speed', 'orbital_speed', 'target_speed', 'vertical_speed', 'horizontal_speed', 'direction_heading',
        'pitch', 'roll',
        # resources
        'stage', 'mass', 'thrust', 'available_thrust',
        # controls
        'stage_remaining_fuel', 'total_delta_v', 'per_stage_delta_v',
        # attitude
        'from_pro', 'angle_of_attack', 'sideslip_angle', 'dynamic_pressure',
        'current_stage_delta_v', 'active_engines', 'throttle',
    ]
    for e in expected:
        assert e in fields_list, f"Missing field: {e}"

def test_flight_record_defaults():
    r = FlightRecord()
    assert r.vessel_name == ''
    assert r.ut == 0.0


def test_alert_rule_less_than_triggers():
    rule = AlertRule("altitude", "<", 70000, lambda r: None)
    state = FlightRecord(altitude=65000)
    assert rule.check(state) is True

def test_alert_rule_not_triggers_when_above():
    rule = AlertRule("altitude", "<", 70000, lambda r: None)
    state = FlightRecord(altitude=75000)
    assert rule.check(state) is False

def test_alert_rule_greater_than():
    rule = AlertRule("inclination", ">", 5.0, lambda r: None)
    assert rule.check(FlightRecord(inclination=10.0)) is True
    assert rule.check(FlightRecord(inclination=2.0)) is False

def test_alert_rule_callback():
    triggered = []
    rule = AlertRule("speed", ">", 500, lambda r: triggered.append(r.speed))
    rule.check(FlightRecord(speed=600))
    assert triggered == [600.0]

def test_alert_rule_invalid_op():
    with pytest.raises(ValueError, match="Invalid op"):
        AlertRule("altitude", "~", 70000, lambda r: None)


# ── FlightMonitor tests ──────────────────────────────────────────────────

from unittest.mock import MagicMock, patch


def test_flight_monitor_init_no_connection():
    """Can be instantiated without kRPC — for unit testing."""
    import threading
    from krpc_rendezvous.common.flight_monitor import FlightMonitor
    monitor = FlightMonitor.__new__(FlightMonitor)
    monitor._state = None
    monitor._conn = None
    monitor._state_lock = threading.Lock()
    from krpc_rendezvous.common.flight_monitor import FlightRecord
    monitor.update_from_record(FlightRecord(vessel_name="test", altitude=1000))
    assert monitor.get_state().vessel_name == "test"


def test_flight_monitor_default_log_dir():
    from krpc_rendezvous.common.flight_monitor import FlightMonitor
    monitor = FlightMonitor.__new__(FlightMonitor)
    log_dir = monitor._default_log_dir()
    assert log_dir.name == 'logs'
    assert '.krpc-rendezvous' in str(log_dir)


def test_csv_writer_path():
    from krpc_rendezvous.common.flight_monitor import _CsvWriter
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        writer = _CsvWriter(Path(tmp), "TestVessel")
        writer.open()
        assert writer.fpath.name.startswith("TestVessel")
        writer.close()


def test_dashboard_renders_without_crashing():
    """FlightDashboard.print_state should not raise with a FlightRecord."""
    from krpc_rendezvous.common.flight_monitor import FlightDashboard, FlightRecord
    dash = FlightDashboard.__new__(FlightDashboard)
    record = FlightRecord(
        vessel_name="Kerbal One",
        altitude=250000,
        apoapsis=280000,
        periapsis=220000,
        inclination=28.5,
        speed=1200.0,
        vertical_speed=150.0,
        horizontal_speed=800.0,
        latitude=10.0,
        longitude=45.0,
        altitude_from_surface=245000,
        stage=2,
        thrust=500.0,
        available_thrust=600.0,
        mass=50000.0,
        stage_remaining_fuel=1200.0,
        total_delta_v=4902.0,
        direction_heading=90.0,
        pitch=5.0,
        roll=-2.0,
        period=3600.0,
        time_to_ap=1200.0,
        time_to_pe=1800.0,
        eccentricity=0.5,
        semi_major_axis=3500000.0,
    )
    # should not raise
    dash._print_state(record)