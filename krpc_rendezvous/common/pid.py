"""PID controller with anti-windup, deadband, and output clamping.

Extracted from atmospheric_ascent.py to eliminate duplication across the
codebase (was: inline PD in launch.py, simplified PID in run_ascent.py,
full PID in atmospheric_ascent.py).
"""

import logging

logger = logging.getLogger(__name__)


class PIDController:
    """PID controller with anti-windup and output clamping.

    Supports deadband, output clamping, and integral anti-windup via
    conditional freezing and clamping.
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, output_min=-1.0, output_max=1.0, deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.deadband = deadband
        self._integral = 0.0
        self._prev_error = None
        self._prev_output = None

    def reset(self):
        """Reset internal state."""
        self._integral = 0.0
        self._prev_error = None
        self._prev_output = None

    def update(self, error, dt):
        """Compute PID output for given error and time step.

        Anti-windup: clamp integral, freeze on saturation.
        Deadband: return 0 if |error| < deadband.
        """
        if dt <= 0:
            return 0.0

        # Deadband
        if abs(error) < self.deadband:
            return 0.0

        # Proportional
        p_out = self.kp * error

        # Integral with anti-windup
        self._integral += error * dt
        # Clamp integral to prevent windup
        if self.ki != 0:
            i_max = (self.output_max - p_out) / self.ki if self.ki != 0 else 1e6
            i_min = (self.output_min - p_out) / self.ki if self.ki != 0 else -1e6
            self._integral = max(min(self._integral, i_max), i_min)
        i_out = self.ki * self._integral

        # Derivative
        d_out = 0.0
        if self._prev_error is not None:
            d_out = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p_out + i_out + d_out

        # Clamp output
        output = max(self.output_min, min(self.output_max, output))

        # Freeze integral on saturation (anti-windup)
        if output == self.output_max or output == self.output_min:
            self._integral -= error * dt

        self._prev_output = output
        return output
