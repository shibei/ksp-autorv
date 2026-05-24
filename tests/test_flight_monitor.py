import pytest
from dataclasses import dataclass, fields
from krpc_rendezvous.common.flight_monitor import FlightRecord

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