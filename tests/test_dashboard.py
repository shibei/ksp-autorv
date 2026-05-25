"""Tests for terminal dashboard formatting (no kRPC required)."""

from krpc_rendezvous.common.dashboard import Dashboard, format_time, format_value


def test_format_value_small():
    assert format_value(0.001, 'm', 2) == '0.00'


def test_format_value_medium():
    assert format_value(100.5, 'm', 2) == '100.50'


def test_format_value_large():
    assert format_value(1e5, 'm', 2) == '1.00e+05'


def test_format_value_none():
    assert format_value(None, 'm') == '-'


def test_format_time():
    assert format_time(90) == '01:30'
    assert format_time(5) == '00:05'
    assert format_time(600) == '10:00'


def test_dashboard_init():
    cols = [('ALT', 'm', 1), ('VEL', 'm/s', 1), ('AP', 'm', 0)]
    db = Dashboard(columns=cols)
    assert len(db.columns) == 3
    assert db.columns[0][0] == 'ALT'


def test_dashboard_format_row():
    db = Dashboard(columns=[('ALT', 'm', 1), ('VEL', 'm/s', 1)])
    row = db.format_row(values=[12345.6, 2345.1])
    assert 'ALT' in row
    assert 'VEL' in row
    assert '12345.6' in row
    assert '2345.1' in row
