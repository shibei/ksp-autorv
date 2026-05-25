"""Singleton kRPC connection manager with auto-reconnect and safe time warp."""

import logging
import time

from krpc_rendezvous.common.config import KSC_ADDRESS, KSC_RPC_PORT, KSC_STREAM_PORT

logger = logging.getLogger(__name__)


class KrpcConnection:
    """Singleton wrapper around kRPC connection."""

    _instance = None

    def __init__(self):
        self._conn = None
        self._connected = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance and cls._instance._connected:
            cls._instance.close()
        cls._instance = None

    def connect(self, name='DirectAscentRendezvous', address=None, rpc_port=None, stream_port=None):
        if address is None:
            address = KSC_ADDRESS
        if rpc_port is None:
            rpc_port = KSC_RPC_PORT
        if stream_port is None:
            stream_port = KSC_STREAM_PORT
        import krpc

        self._conn = krpc.connect(
            name=name, address=address, rpc_port=rpc_port, stream_port=stream_port
        )
        self._connected = True
        logger.info(f'Connected to kRPC server at {address}:{rpc_port}')
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._connected = False

    @property
    def conn(self):
        if not self._connected:
            raise RuntimeError('kRPC not connected. Call connect() first.')
        return self._conn

    @property
    def space_center(self):
        return self.conn.space_center

    @property
    def vessel(self):
        return self.space_center.active_vessel

    @property
    def target_vessel(self):
        return self.space_center.target_vessel

    @property
    def ut(self):
        return self.space_center.ut

    def add_stream(self, *args, **kwargs):
        return self.conn.add_stream(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith('_') or name in ('conn',):
            raise AttributeError(name)
        return getattr(self.conn, name)


def connect_krpc(name='DirectAscentRendezvous'):
    """Convenience function: connect and return KrpcConnection instance."""
    inst = KrpcConnection.get_instance()
    inst.connect(name=name)
    return inst


def get_connection():
    """Get existing KrpcConnection instance (must already be connected)."""
    return KrpcConnection.get_instance()


def safe_warp(target_ut, margin=60.0, max_rate=None):
    """Safely warp to near target UT, then drop to 1x."""
    conn = get_connection()
    sc = conn.space_center

    current_ut = sc.ut
    if target_ut <= current_ut + margin:
        return

    vessel = sc.active_vessel
    if vessel.situation.name != 'pre_launch':
        if vessel.flight(vessel.orbit.body.reference_frame).mean_altitude < 70000:
            return

    warp_to = target_ut - margin
    if max_rate is not None:
        current_rate = sc.rails_warp_factor
        if current_rate > max_rate:
            sc.rails_warp_factor = max_rate

    sc.warp_to(warp_to)
    sc.rails_warp_factor = 0
    while sc.ut < target_ut - 5:
        time.sleep(0.1)
