"""Real-time flight state monitoring, HUD, alerts, and logging."""

from __future__ import annotations

import csv
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FlightRecord:
    """Canonical shape of one time-step flight state snapshot."""
    ut: float = 0.0
    vessel_name: str = ''
    # orbital
    altitude: float = 0.0          # orbit altitude (m)
    apoapsis: float = 0.0         # m
    periapsis: float = 0.0        # m
    inclination: float = 0.0       # degrees
    eccentricity: float = 0.0
    semi_major_axis: float = 0.0  # m
    period: float = 0.0           # seconds
    time_to_ap: float = 0.0       # seconds
    time_to_pe: float = 0.0       # seconds
    # position
    latitude: float = 0.0         # degrees
    longitude: float = 0.0        # degrees
    altitude_from_surface: float = 0.0  # m (altitude above terrain)
    # velocity
    speed: float = 0.0             # m/s
    vertical_speed: float = 0.0   # m/s
    horizontal_speed: float = 0.0 # m/s
    direction_heading: float = 0.0  # degrees (0-360)
    # resources
    stage: int = 0
    mass: float = 0.0             # kg
    thrust: float = 0.0           # kN
    available_thrust: float = 0.0 # kN
    stage_remaining_fuel: float = 0.0  # kg
    stage_delta_v: float = 0.0    # m/s


@dataclass(slots=True)
class AlertRule:
    """Threshold alert: fires callback when field op value is True."""
    field: str
    op: str          # "<" | ">" | "<=" | ">=" | "==" | "!="
    threshold: float
    callback: Callable[[FlightRecord], None]

    VALID_OPS = {"<", ">", "<=", ">=", "==", "!="}

    def __post_init__(self):
        if self.op not in self.VALID_OPS:
            raise ValueError(f"Invalid op {self.op!r}, expected one of {self.VALID_OPS}")

    def check(self, record: FlightRecord) -> bool:
        value = getattr(record, self.field, None)
        if value is None:
            return False
        op_fn = {
            "<":  lambda v, t: v < t,
            ">":  lambda v, t: v > t,
            "<=": lambda v, t: v <= t,
            ">=": lambda v, t: v >= t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t,
        }[self.op]
        triggered = op_fn(float(value), float(self.threshold))
        if triggered:
            self.callback(record)
        return triggered