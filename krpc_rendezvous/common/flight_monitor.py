"""Real-time flight state monitoring, HUD, alerts, and logging."""

from __future__ import annotations

import csv
import logging
import os
import sys
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


class FlightMonitor:
    """Real-time flight state monitor.

    Connects to kRPC, streams active vessel state at 4Hz,
    stores latest FlightRecord, fires AlertRules,
    and persists records to CSV.

    Usage:
        monitor = FlightMonitor()
        monitor.connect()          # kRPC + streams
        monitor.start_logging()    # start CSV persistence
        for alert in default_alerts():
            monitor.add_alert(alert)
        monitor.start()            # start stream pump in background thread

        # anywhere, any time:
        state = monitor.get_state()  # FlightRecord snapshot

        monitor.stop()             # graceful shutdown
    """

    DEFAULT_STREAM_RATE = 4.0  # Hz

    def __init__(
        self,
        name: str = 'FlightMonitor',
        stream_rate: float = DEFAULT_STREAM_RATE,
        log_dir: Optional[Path] = None,
    ):
        self._name = name
        self._stream_rate = stream_rate
        self._conn = None
        self._vessel = None
        self._streams: dict[str, object] = {}
        self._state: Optional[FlightRecord] = None
        self._state_lock = threading.Lock()
        self._alerts: list[AlertRule] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._log_writer: Optional[_CsvWriter] = None
        self._log_dir = log_dir or self._default_log_dir()

    def _default_log_dir(self) -> Path:
        base = Path.home() / '.krpc-rendezvous' / 'logs'
        base.mkdir(parents=True, exist_ok=True)
        return base

    # ── Connection ─────────────────────────────────────────────────────

    def connect(self, address=None, rpc_port=None, stream_port=None):
        from krpc_rendezvous.common.config import KSC_ADDRESS, KSC_RPC_PORT, KSC_STREAM_PORT
        import krpc

        addr = address or KSC_ADDRESS
        rport = rpc_port or KSC_RPC_PORT
        sport = stream_port or KSC_STREAM_PORT

        self._conn = krpc.connect(
            name=self._name,
            address=addr,
            rpc_port=rport,
            stream_port=sport,
        )
        sc = self._conn.space_center
        self._vessel = sc.active_vessel
        self._build_streams(sc)
        logger.info(f"FlightMonitor connected to vessel: {self._vessel.name}")
        return self

    def _build_streams(self, sc):
        """Create all kRPC streams for the active vessel."""
        v = self._vessel
        orbit = v.orbit
        body_rf = v.orbit.body.reference_frame
        surface_rf = v.surface_reference_frame

        def add(name: str, expr):
            self._streams[name] = self._conn.add_stream(expr)

        # orbital
        add('altitude', orbit.altitude)
        add('apoapsis', orbit.apoapsis)
        add('periapsis', orbit.periapsis)
        add('inclination', orbit.inclination)
        add('eccentricity', orbit.eccentricity)
        add('semi_major_axis', orbit.semi_major_axis)
        add('period', orbit.period)
        add('time_to_ap', orbit.time_to_apoapsis)
        add('time_to_pe', orbit.time_to_periapsis)
        # position
        add('latitude', v.flight(body_rf).latitude)
        add('longitude', v.flight(body_rf).longitude)
        add('altitude_from_surface', v.flight(surface_rf).surface_altitude)
        # velocity
        add('speed', v.flight(surface_rf).speed)
        add('vertical_speed', v.flight(surface_rf).vertical_speed)
        add('horizontal_speed', v.flight(surface_rf).horizontal_speed)
        add('direction_heading', v.flight(surface_rf).heading)
        # resources
        add('stage', v.control.current_stage)
        add('mass', v.mass)
        add('thrust', v.thrust)
        add('available_thrust', v.available_thrust)

    def _read_streams(self) -> FlightRecord:
        """Read all stream values and return a FlightRecord."""
        s = self._streams
        stage_num = int(s['stage'].value)
        stage_fuel = 0.0
        stage_dv = 0.0
        try:
            for part in self._vessel.parts.all:
                if part.stage == stage_num:
                    for resource in part.resources.all:
                        if resource.name == 'LiquidFuel':
                            stage_fuel += resource.amount
                    if hasattr(part, 'engine') and part.engine is not None:
                        stage_dv += getattr(part.engine, 'max_vacuum_thrust', 0.0) * 1000
        except Exception:
            pass

        return FlightRecord(
            ut=self._conn.space_center.ut,
            vessel_name=self._vessel.name,
            altitude=float(s['altitude'].value or 0),
            apoapsis=float(s['apoapsis'].value or 0),
            periapsis=float(s['periapsis'].value or 0),
            inclination=float(s['inclination'].value or 0),
            eccentricity=float(s['eccentricity'].value or 0),
            semi_major_axis=float(s['semi_major_axis'].value or 0),
            period=float(s['period'].value or 0),
            time_to_ap=float(s['time_to_ap'].value or 0),
            time_to_pe=float(s['time_to_pe'].value or 0),
            latitude=float(s['latitude'].value or 0),
            longitude=float(s['longitude'].value or 0),
            altitude_from_surface=float(s['altitude_from_surface'].value or 0),
            speed=float(s['speed'].value or 0),
            vertical_speed=float(s['vertical_speed'].value or 0),
            horizontal_speed=float(s['horizontal_speed'].value or 0),
            direction_heading=float(s['direction_heading'].value or 0),
            stage=int(stage_num),
            mass=float(s['mass'].value or 0),
            thrust=float(s['thrust'].value or 0),
            available_thrust=float(s['available_thrust'].value or 0),
            stage_remaining_fuel=stage_fuel,
            stage_delta_v=stage_dv,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def start(self):
        """Start the background stream pump at 4Hz."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True, name='FlightMonitor-pump')
        self._thread.start()
        logger.info("FlightMonitor pump started")
        return self

    def stop(self):
        """Stop the background pump gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._log_writer:
            self._log_writer.close()
        logger.info("FlightMonitor stopped")

    def get_state(self) -> Optional[FlightRecord]:
        """Return the latest FlightRecord snapshot (thread-safe)."""
        with self._state_lock:
            return self._state

    def update_from_record(self, record: FlightRecord):
        """Update state from an external record (for testing)."""
        with self._state_lock:
            self._state = record

    def add_alert(self, rule: AlertRule):
        """Register an alert rule."""
        self._alerts.append(rule)

    def start_logging(self, vessel_name: Optional[str] = None):
        """Start CSV logging to ~/.krpc-rendezvous/logs/[vessel]-[date].csv"""
        name = vessel_name or (self._vessel.name if self._vessel else 'unknown')
        self._log_writer = _CsvWriter(self._log_dir, name)
        self._log_writer.open()
        logger.info(f"Flight logging started: {self._log_writer.fpath}")

    # ── Internal pump ──────────────────────────────────────────────────

    def _pump(self):
        interval = 1.0 / self._stream_rate
        while self._running:
            try:
                record = self._read_streams()
                with self._state_lock:
                    self._state = record
                for rule in self._alerts:
                    rule.check(record)
                if self._log_writer:
                    self._log_writer.write(record)
            except Exception as exc:
                logger.warning(f"FlightMonitor pump error: {exc}")
            import time
            time.sleep(interval)

    def is_running(self) -> bool:
        return self._running


class _CsvWriter:
    """Backs a FlightRecord CSV log file with daily rotation."""

    HEADER = [
        'ut', 'vessel_name',
        'altitude', 'apoapsis', 'periapsis', 'inclination', 'eccentricity',
        'semi_major_axis', 'period', 'time_to_ap', 'time_to_pe',
        'latitude', 'longitude', 'altitude_from_surface',
        'speed', 'vertical_speed', 'horizontal_speed', 'direction_heading',
        'stage', 'mass', 'thrust', 'available_thrust',
        'stage_remaining_fuel', 'stage_delta_v',
    ]

    def __init__(self, log_dir: Path, vessel_name: str):
        self._log_dir = log_dir
        self._vessel_name = vessel_name
        self._fpath: Optional[Path] = None
        self._file = None
        self._writer = None

    def open(self):
        date = datetime.utcnow().strftime('%Y-%m-%d')
        self._fpath = self._log_dir / f"{self._vessel_name}-{date}.csv"
        if self._fpath.exists():
            stem = self._fpath.stem
            suffix = self._fpath.suffix
            i = 1
            while self._fpath.exists():
                self._fpath = self._log_dir / f"{stem}-{i}{suffix}"
                i += 1
        self._file = open(self._fpath, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        self._writer.writeheader()

    def write(self, record: FlightRecord):
        if self._writer is None:
            return
        row = {f: getattr(record, f, 0.0) for f in self.HEADER}
        self._writer.writerow(row)

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    @property
    def fpath(self) -> Path:
        return self._fpath


class FlightDashboard:
    """Terminal HUD that prints FlightRecord state at 4Hz.

    Usage:
        dash = FlightDashboard(monitor)
        dash.start()    # prints in background thread
        # ...
        dash.stop()
    """

    def __init__(self, monitor: 'FlightMonitor', print_rate: float = 1.0):
        self._monitor = monitor
        self._print_rate = print_rate
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True, name='FlightDashboard-pump')
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _pump(self):
        import time
        interval = 1.0 / self._print_rate
        while self._running:
            state = self._monitor.get_state()
            if state:
                self._print_state(state)
            time.sleep(interval)

    def _print_state(self, r: FlightRecord):
        out = sys.stdout
        lines = [
            "",
            f"  ✈  {r.vessel_name or 'Unknown'}  ·  Stage {r.stage}  ·  UT {r.ut:.1f}",
            f"  ALT {r.altitude/1000:.1f}km  AP {r.apoapsis/1000:.1f}km  PE {r.periapsis/1000:.1f}km  INC {r.inclination:.1f}°",
            f"  SPD {r.speed:.0f}  VSPD {r.vertical_speed:+.0f}  HSPD {r.horizontal_speed:.0f}",
            f"  HDG {r.direction_heading:.0f}°  LAT {r.latitude:+.2f}°  LON {r.longitude:+.2f}°",
            f"  MASS {r.mass/1000:.1f}t  THRUST {r.thrust:.0f}/{r.available_thrust:.0f} kN",
            f"  FUEL {r.stage_remaining_fuel:.0f} kg  ΔV {r.stage_delta_v/1000:.1f} km/s",
            "",
        ]
        cursor_up = f"\033[{len(lines)}A"
        clear = "\033[K"
        out.write(cursor_up)
        for line in lines:
            out.write(line + clear + "\r\n")
        out.flush()