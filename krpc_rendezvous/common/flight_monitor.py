"""Real-time flight state monitoring, HUD, alerts, and logging."""

from __future__ import annotations

import csv
import logging
import math
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ['AlertRule', 'FlightDashboard', 'FlightMonitor', 'FlightRecord', 'default_alerts']

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
    speed: float = 0.0             # m/s, surface speed (relative to ground)
    orbital_speed: float = 0.0    # m/s, orbital speed (inertial, relative to body center)
    target_speed: float = 0.0     # m/s, speed relative to target vessel (0 if no target)
    vertical_speed: float = 0.0   # m/s
    horizontal_speed: float = 0.0 # m/s
    direction_heading: float = 0.0  # degrees (0-360)
    pitch: float = 0.0              # degrees (-90..90)
    roll: float = 0.0               # degrees (-180..180)
    # resources
    stage: int = 0
    mass: float = 0.0             # kg
    thrust: float = 0.0           # kN
    available_thrust: float = 0.0 # kN
    stage_remaining_fuel: float = 0.0  # kg
    total_delta_v: float = 0.0    # m/s, from MechJeb StageStats
    per_stage_delta_v: str = ''   # JSON dict of stage->Δv, e.g. '{"6":1251,"5":1758}'
    from_pro: float = 0.0         # degrees, true angle between vessel forward and prograde (0=prograde, 90=perpendicular, 180=retrograde)
    angle_of_attack: float = 0.0  # degrees, angle between forward and velocity in pitch plane
    sideslip_angle: float = 0.0   # degrees, angle between velocity and vessel xz-plane
    dynamic_pressure: float = 0.0  # Pa, aerodynamic pressure (0 in vacuum)
    current_stage_delta_v: float = 0.0  # m/s, Δv remaining for current stage
    active_engines: str = ''            # comma-separated names of active engines, e.g. "LV-T30,LV-T30"
    throttle: float = 0.0               # throttle setting (0.0–1.0)


@dataclass(slots=True)
class AlertRule:
    """Threshold alert: fires callback when field op value is True."""
    field: str
    op: str          # "<" | ">" | "<=" | ">=" | "==" | "!="
    threshold: float
    callback: Callable[[FlightRecord], None]
    VALID_OPS: ClassVar[frozenset[str]] = frozenset({"<", ">", "<=", ">=", "==", "!="})

    def __post_init__(self) -> None:
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
        log_dir: Path | None = None,
    ):
        self._name = name
        self._stream_rate = stream_rate
        self._conn: Any = None
        self._vessel: Any = None
        self._streams: dict[str, Any] = {}
        self._state: FlightRecord | None = None
        self._state_lock = threading.Lock()
        self._alerts: list[AlertRule] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._log_writer: _CsvWriter | None = None
        self._log_dir = log_dir or self._default_log_dir()
        self._last_stage: int = -1

    def _default_log_dir(self) -> Path:
        base = Path.home() / '.krpc-rendezvous' / 'logs'
        base.mkdir(parents=True, exist_ok=True)
        return base

    # ── Connection ─────────────────────────────────────────────────────

    def connect(self, address=None, rpc_port=None, stream_port=None):
        import krpc

        from krpc_rendezvous.common.config import KSC_ADDRESS, KSC_RPC_PORT, KSC_STREAM_PORT

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
        """Create all kRPC streams for the active vessel.

        kRPC 0.5.x ``add_stream`` accepts bound methods *or* ``getattr``
        for property-like access.  Flight position/velocity is obtained by
        streaming ``v.flight(ref)`` (a method) and reading properties on the
        returned Flight proxy at read time.
        """
        v = self._vessel
        body_rf = v.orbit.body.reference_frame
        surface_rf = v.surface_reference_frame

        # Stream flight METHOD results (returns fresh Flight proxy objects)
        self._streams['flight_body'] = self._conn.add_stream(v.flight, body_rf)
        self._streams['flight_surface'] = self._conn.add_stream(v.flight, surface_rf)

        # Non-rotating body frame for true orbital speed
        try:
            non_rot_rf = v.orbit.body.non_rotating_reference_frame
            self._streams['flight_inertial'] = self._conn.add_stream(v.flight, non_rot_rf)
        except Exception:
            self._streams['flight_inertial'] = None

        # Stream vessel direction in body frame (for from_pro computation)
        self._streams['direction'] = self._conn.add_stream(v.direction, body_rf)

        # Stream orbital properties via getattr
        for prop in ('apoapsis', 'periapsis', 'inclination', 'eccentricity',
                     'semi_major_axis', 'period', 'time_to_apoapsis',
                     'time_to_periapsis'):
            self._streams[prop] = self._conn.add_stream(getattr, v.orbit, prop)
        # Stream vessel/control properties via getattr
        for prop in ('mass', 'thrust', 'available_thrust'):
            self._streams[prop] = self._conn.add_stream(getattr, v, prop)
        self._streams['stage'] = self._conn.add_stream(getattr, v.control,
                                                        'current_stage')
        self._streams['throttle'] = self._conn.add_stream(getattr, v.control,
                                                           'throttle')

    def _read_streams(self) -> FlightRecord:
        s = self._streams
        stream_call = lambda k: self._streams[k]()

        flight_body = stream_call('flight_body')
        flight_surface = stream_call('flight_surface')

        stage_num = int(stream_call('stage'))
        throttle_val = float(stream_call('throttle') or 0)
        stage_fuel = 0.0
        old_stage_dv = 0.0
        try:
            for part in self._vessel.parts.all:
                if part.stage == stage_num:
                    for resource in part.resources.all:
                        if resource.name == 'LiquidFuel':
                            stage_fuel += resource.amount
                    if hasattr(part, 'engine') and part.engine is not None:
                        old_stage_dv += getattr(part.engine, 'max_vacuum_thrust', 0.0) * 1000
        except Exception:
            pass

        # Read MechJeb delta-v (graceful fallback if unavailable)
        total_dv = 0.0
        cur_stage_dv = 0.0
        per_stage_str = ''
        try:
            mj = self._conn.mech_jeb
            if mj.api_ready:
                ss = mj.stage_stats
                total_dv = float(ss.total_delta_v or 0)
                stage_dict = dict(ss.stage_delta_v) if ss.stage_delta_v else {}
                if stage_dict:
                    import json
                    per_stage_str = json.dumps(stage_dict, separators=(',', ':'))
                    cur_stage_dv = float(stage_dict.get(str(stage_num), 0))
        except Exception:
            pass  # MechJeb not available

        # Target-relative speed
        target_spd = 0.0
        try:
            tgt = self._conn.space_center.target_vessel
            if tgt is not None:
                rel_vel = self._vessel.velocity(tgt.reference_frame)
                target_spd = math.sqrt(rel_vel[0]**2 + rel_vel[1]**2 + rel_vel[2]**2)
        except Exception:
            pass

        # Detect staging event
        if self._last_stage >= 0 and stage_num != self._last_stage:
            logger.info("STAGE  %d → %d  @ UT %.1f", self._last_stage, stage_num, self._conn.space_center.ut)
        self._last_stage = stage_num

        # Compute from_pro: true angle between vessel forward and prograde
        fwd = stream_call('direction')
        pro = flight_body.prograde
        dot = float(fwd[0] * pro[0] + fwd[1] * pro[1] + fwd[2] * pro[2])
        from_pro = math.degrees(math.acos(max(-1.0, min(1.0, dot))))

        # Orbital speed from non-rotating frame (true inertial speed)
        flight_inertial = None
        flight_inertial_stream = self._streams.get('flight_inertial')
        if flight_inertial_stream:
            flight_inertial = flight_inertial_stream()
        orb_spd = float(flight_inertial.speed) if flight_inertial is not None else 0.0
        aoa = float(flight_body.angle_of_attack or 0)
        slip = float(flight_body.sideslip_angle or 0)
        q = float(flight_body.dynamic_pressure or 0)

        # Collect active engine names
        try:
            active_eng_str = ','.join(
                e.part.title for e in self._vessel.parts.engines if e.active
            )
        except Exception:
            active_eng_str = ''

        return FlightRecord(
            ut=self._conn.space_center.ut,
            vessel_name=self._vessel.name,
            altitude=float(flight_body.mean_altitude or 0),
            apoapsis=float(stream_call('apoapsis') or 0),
            periapsis=float(stream_call('periapsis') or 0),
            inclination=float(stream_call('inclination') or 0),
            eccentricity=float(stream_call('eccentricity') or 0),
            semi_major_axis=float(stream_call('semi_major_axis') or 0),
            period=float(stream_call('period') or 0),
            time_to_ap=float(stream_call('time_to_apoapsis') or 0),
            time_to_pe=float(stream_call('time_to_periapsis') or 0),
            latitude=float(flight_body.latitude or 0),
            longitude=float(flight_body.longitude or 0),
            altitude_from_surface=float(flight_surface.surface_altitude or 0),
            speed=float(flight_surface.speed or 0),
            orbital_speed=orb_spd,
            target_speed=target_spd,
            vertical_speed=float(flight_surface.vertical_speed or 0),
            horizontal_speed=float(flight_surface.horizontal_speed or 0),
            direction_heading=float(flight_surface.heading or 0),
            pitch=float(flight_surface.pitch or 0),
            roll=float(flight_surface.roll or 0),
            stage=int(stage_num),
            mass=float(stream_call('mass') or 0),
            thrust=float(stream_call('thrust') or 0),
            available_thrust=float(stream_call('available_thrust') or 0),
            stage_remaining_fuel=stage_fuel,
            total_delta_v=total_dv,
            per_stage_delta_v=per_stage_str,
            from_pro=from_pro,
            angle_of_attack=aoa,
            sideslip_angle=slip,
            dynamic_pressure=q,
            current_stage_delta_v=cur_stage_dv,
            active_engines=active_eng_str,
            throttle=throttle_val,
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

    def get_state(self) -> FlightRecord | None:
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

    def start_logging(self, vessel_name: str | None = None):
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

    HEADER: ClassVar[list[str]] = [
        'ut', 'vessel_name',
        'altitude', 'apoapsis', 'periapsis', 'inclination', 'eccentricity',
        'semi_major_axis', 'period', 'time_to_ap', 'time_to_pe',
        'latitude', 'longitude', 'altitude_from_surface',
        'speed', 'orbital_speed', 'target_speed', 'vertical_speed', 'horizontal_speed', 'direction_heading',
        'pitch', 'roll',
        'stage', 'mass', 'thrust', 'available_thrust',
        'stage_remaining_fuel', 'total_delta_v', 'per_stage_delta_v',
        'from_pro', 'angle_of_attack', 'sideslip_angle', 'dynamic_pressure',
        'current_stage_delta_v', 'active_engines', 'throttle',
    ]

    def __init__(self, log_dir: Path, vessel_name: str):
        self._log_dir = log_dir
        self._vessel_name = vessel_name
        self._fpath: Path | None = None
        self._file: Any = None
        self._writer: Any = None

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
    def fpath(self) -> Path | None:
        return self._fpath


class FlightDashboard:
    """Terminal HUD that prints FlightRecord state at 4Hz.

    Usage:
        dash = FlightDashboard(monitor)
        dash.start()    # prints in background thread
        # ...
        dash.stop()
    """

    def __init__(self, monitor: FlightMonitor, print_rate: float = 1.0):
        self._monitor = monitor
        self._print_rate = print_rate
        self._running = False
        self._thread: threading.Thread | None = None

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
            (f"  ALT {r.altitude/1000:.1f}km  AP {r.apoapsis/1000:.1f}km"
             f"  PE {r.periapsis/1000:.1f}km  INC {r.inclination:.1f}°"),
            f"  SPD {r.speed:.0f}  ORB {r.orbital_speed:.0f}  TGT {r.target_speed:.0f}  VSPD {r.vertical_speed:+.0f}  HSPD {r.horizontal_speed:.0f}",
            f"  HDG {r.direction_heading:.0f}°  PIT {r.pitch:+.0f}°  ROL {r.roll:+.0f}°",
            f"  LAT {r.latitude:+.2f}°  LON {r.longitude:+.2f}°",
            f"  AOA {r.angle_of_attack:+.0f}°  SS {r.sideslip_angle:+.0f}°  PRO {r.from_pro:.0f}°  THR {r.throttle:.2f}  Q {r.dynamic_pressure:.0f} Pa",
            f"  MASS {r.mass/1000:.1f}t  THRUST {r.thrust:.0f}/{r.available_thrust:.0f} kN",
            f"  FUEL {r.stage_remaining_fuel:.0f} kg  ΔV {r.total_delta_v:.0f}  STGΔV {r.current_stage_delta_v:.0f}  {r.active_engines}",
            "",
        ]
        cursor_up = f"\033[{len(lines)}A"
        clear = "\033[K"
        out.write(cursor_up)
        for line in lines:
            out.write(line + clear + "\r\n")
        out.flush()


def default_alerts() -> list[AlertRule]:
    """Factory for commonly useful flight alerts."""
    alerts = []

    def print_alert(msg: str) -> Callable:
        def cb(r: FlightRecord):
            print(f"\n⚠️  FLIGHT ALERT  {msg}")
        return cb

    # Orbit degradation warnings
    alerts.append(AlertRule("periapsis", "<", 70000, print_alert("Periapsis below 70km")))
    alerts.append(AlertRule("apoapsis", "<", 70000, print_alert("Apoapsis below 70km")))

    # Inclination issues for rendezvous
    alerts.append(AlertRule("inclination", ">", 30.0, print_alert("High inclination")))
    alerts.append(AlertRule("inclination", "<", 1.0, print_alert("Low inclination")))

    # Descent/ascent rate
    alerts.append(AlertRule("vertical_speed", "<", -500, print_alert("Rapid descent!")))
    alerts.append(AlertRule("vertical_speed", ">", 500, print_alert("Rapid ascent!")))

    return alerts
