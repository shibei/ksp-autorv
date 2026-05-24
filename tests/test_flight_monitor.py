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
        'speed', 'vertical_speed', 'horizontal_speed', 'direction_heading',
        # resources
        'stage', 'mass', 'thrust', 'available_thrust',
        # controls
        'stage_remaining_fuel', 'stage_delta_v',
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