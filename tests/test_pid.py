"""Unit tests for PIDController (no external dependencies required)."""

import math

import pytest

from krpc_rendezvous.common.pid import PIDController


def test_init_defaults():
    pid = PIDController()
    assert pid.kp == 0.0
    assert pid.ki == 0.0
    assert pid.kd == 0.0
    assert pid.output_min == -1.0
    assert pid.output_max == 1.0
    assert pid.deadband == 0.0
    assert pid._integral == 0.0
    assert pid._prev_error is None
    assert pid._prev_output is None


def test_init_custom():
    pid = PIDController(kp=2.0, ki=0.5, kd=0.1, output_min=-10.0, output_max=10.0, deadband=0.5)
    assert pid.kp == 2.0
    assert pid.ki == 0.5
    assert pid.kd == 0.1
    assert pid.output_min == -10.0
    assert pid.output_max == 10.0
    assert pid.deadband == 0.5


def test_reset():
    pid = PIDController(kp=1.0, ki=0.1)
    pid.update(10.0, 0.1)
    assert pid._integral != 0.0
    assert pid._prev_error is not None
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error is None
    assert pid._prev_output is None


def test_proportional_only():
    pid = PIDController(kp=2.0, output_min=-100.0, output_max=100.0)
    out = pid.update(error=5.0, dt=0.1)
    assert out == pytest.approx(10.0)
    assert pid._integral == 0.5  # integral still accumulates


def test_derivative():
    pid = PIDController(kd=1.0, output_min=-1000.0, output_max=1000.0)
    pid.update(error=10.0, dt=0.1)
    out2 = pid.update(error=20.0, dt=0.1)
    # derivative = kd * (error - prev_error) / dt = 1.0 * (10.0) / 0.1 = 100
    out3 = pid.update(error=20.0, dt=0.1)  # zero derivative when error is unchanged
    assert out2 == pytest.approx(100.0)
    assert out3 == pytest.approx(0.0)


def test_integral_accumulates():
    pid = PIDController(ki=0.1, output_min=-100.0, output_max=100.0)
    pid.update(10.0, 0.1)  # integral += 10.0 * 0.1 = 1.0
    out = pid.update(10.0, 0.1)  # integral += 10.0 * 0.1 = 2.0, output = 0.1 * 2.0 = 0.2
    assert out == pytest.approx(0.2)


def test_output_clamping():
    pid = PIDController(kp=100.0, output_min=-5.0, output_max=5.0)
    out = pid.update(error=1.0, dt=0.1)
    assert out == 5.0  # clamped
    out = pid.update(error=-1.0, dt=0.1)
    assert out == -5.0  # clamped, reversed


def test_deadband():
    pid = PIDController(kp=1.0, deadband=2.0, output_min=-100.0, output_max=100.0)
    assert pid.update(error=1.0, dt=0.1) == 0.0
    assert pid.update(error=-1.5, dt=0.1) == 0.0
    assert pid.update(error=2.0, dt=0.1) == 2.0  # boundary: abs(err) < deadband is strict
    out = pid.update(error=2.5, dt=0.1)
    assert out > 0.0


def test_zero_dt_returns_zero():
    pid = PIDController(kp=1.0)
    assert pid.update(error=5.0, dt=0.0) == 0.0
    assert pid.update(error=5.0, dt=-1.0) == 0.0


def test_convergence():
    """PID should bring error toward zero with appropriate gains."""
    pid = PIDController(kp=0.5, ki=0.1, kd=0.05, output_min=-1000.0, output_max=1000.0)
    error = 100.0
    for _ in range(200):
        correction = pid.update(error, 0.1)
        error -= correction * 0.1  # simulate plant response
    assert abs(error) < 10.0  # should have reduced significantly


def test_anti_windup_saturation():
    """Integral should not grow unbounded during sustained saturation."""
    pid = PIDController(kp=0.1, ki=0.5, output_min=-10.0, output_max=10.0)
    for _ in range(50):
        pid.update(100.0, 0.1)
    assert abs(pid._integral) < 500


def test_anti_windup_recovery():
    """After saturation clears, output should stabilize at a finite value
    (integral can only drain via opposite error, but never blows up)."""
    pid = PIDController(kp=0.1, ki=0.5, output_min=-10.0, output_max=10.0)
    for _ in range(20):
        pid.update(100.0, 0.1)
    outputs = []
    for _ in range(50):
        outputs.append(pid.update(0.0, 0.1))
    # Output should be stable (not changing) and finite
    last = outputs[-1]
    assert all(abs(o - last) < 1e-9 for o in outputs[-10:])
    assert math.isfinite(last)


def test_deterministic():
    """Same inputs produce same outputs."""
    pid1 = PIDController(kp=1.0, ki=0.1, kd=0.05)
    pid2 = PIDController(kp=1.0, ki=0.1, kd=0.05)
    assert pid1.update(5.0, 0.1) == pid2.update(5.0, 0.1)


def test_negative_error():
    pid = PIDController(kp=1.0, output_min=-100.0, output_max=100.0)
    out = pid.update(-5.0, 0.1)
    assert out < 0.0
    assert out == pytest.approx(-5.0)


def test_error_sign_change():
    """PID should handle error crossing zero (sign change)."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05, output_min=-100.0, output_max=100.0)
    pid.update(10.0, 0.1)
    out2 = pid.update(-10.0, 0.1)
    # Error went from 10 to -10, derivative = -20/0.1 = -200, kd*derivative = -10
    assert out2 < 0


def test_integral_only():
    pid = PIDController(kp=0.0, ki=0.5, kd=0.0, output_min=-100.0, output_max=100.0)
    pid.update(10.0, 0.1)  # integral = 1.0
    out = pid.update(10.0, 0.1)  # integral = 2.0
    assert out == pytest.approx(1.0)


def test_derivative_only():
    pid = PIDController(kp=0.0, ki=0.0, kd=2.0, output_min=-1000.0, output_max=1000.0)
    pid.update(0.0, 0.1)  # first call, no prev_error -> d_out = 0
    out = pid.update(5.0, 0.1)  # derivative = 5.0/0.1 = 50, output = 2.0 * 50 = 100
    assert out == pytest.approx(100.0)


def test_consecutive_no_change():
    """With steady error, integral continues to accumulate each step."""
    pid = PIDController(kp=0.5, ki=0.2, kd=0.1, output_min=-100.0, output_max=100.0)
    pid.update(10.0, 0.1)  # i=0.2*1.0=0.2, total=5.2
    pid.update(10.0, 0.1)  # i+=1: i=0.2*2=0.4, total=5.4
    assert pid.update(10.0, 0.1) == pytest.approx(5.0 + 0.2 * 3.0)  # i=0.2*3=0.6


def test_zero_ki_no_division_error():
    """ki=0 should not cause division-by-zero in clamping logic."""
    pid = PIDController(kp=1.0, ki=0.0, output_min=-100.0, output_max=100.0)
    pid.update(100.0, 1.0)  # large error, large dt
    assert math.isfinite(pid.update(50.0, 1.0))


def test_integral_freeze_on_saturation():
    """Integral should stop accumulating when output is clamped."""
    pid = PIDController(kp=0.1, ki=1.0, output_min=-10.0, output_max=10.0)
    # First update saturates output at 10.0
    pid.update(100.0, 0.1)
    integral_after_first = pid._integral
    # Second update should NOT accumulate integral (frozen on saturation)
    pid.update(100.0, 0.1)
    assert pid._integral != pytest.approx(integral_after_first * 2)
