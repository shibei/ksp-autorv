"""Pytest configuration, fixtures, and shared helpers.

All tests in this suite follow these conventions:
  - `integration` marker: requires kRPC / KSP runtime (skipped in CI)
  - `online` marker: requires network access (skipped in CI)
  - Everything else: pure unit test, runs offline with no external deps
"""

from __future__ import annotations

import os
from collections.abc import Generator  # noqa: TC003 (used in yield type annotations)
from unittest.mock import MagicMock, patch

import pytest

# ── Test helpers ───────────────────────────────────────────────────────


def requires_krpc() -> bool:
    """Check whether kRPC integration tests can run."""
    try:
        import krpc  # noqa: F401

        return True
    except ImportError:
        return False


def requires_live_krpc() -> bool:
    """Check whether a live kRPC server is reachable."""
    try:
        import krpc  # noqa: F401

        conn = krpc.connect(
            name='probe',
            address='172.17.64.1',
            rpc_port=50000,
            stream_port=50001,
        )
        conn.close()
        return True
    except Exception:
        return False


def requires_numpy() -> bool:
    """Check whether numpy-dependent tests can run."""
    try:
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def env_reset() -> Generator[None, None, None]:
    """Temporarily clear kRPC-related env vars, restores after test."""
    saved = {}
    for key in ('KRPCC_HOST', 'KRPCC_RPC_PORT', 'KRPCC_STREAM'):
        saved[key] = os.environ.get(key)
        os.environ.pop(key, None)
    yield
    for key, val in saved.items():
        if val is not None:
            os.environ[key] = val


@pytest.fixture
def env_with_host(env_reset: None) -> None:
    """Set KRPCC_HOST env var to a test value."""
    os.environ['KRPCC_HOST'] = '10.0.0.99'
    os.environ['KRPCC_RPC_PORT'] = '50002'
    os.environ['KRPCC_STREAM'] = '50003'


@pytest.fixture
def mock_krpc_connect() -> Generator[MagicMock, None, None]:
    """Mock krpc.connect() returning a MagicMock space_center.

    Usage:
        def test_something(mock_krpc_connect):
            mock = mock_krpc_connect
            mock.space_center.active_vessel.name = 'My Rocket'
            # ... test code that calls connect_krpc() ...
    """
    with patch('krpc.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        yield mock_conn


# ── Auto-collection filters ───────────────────────────────────────────


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip integration/online tests when their preconditions aren't met."""
    for item in items:
        if 'integration' in item.keywords and not requires_krpc():
            item.add_marker(
                pytest.mark.skip(reason='kRPC not available — integration test skipped')
            )
        if 'live_krpc' in item.keywords and not requires_live_krpc():
            item.add_marker(
                pytest.mark.skip(reason='kRPC server not reachable — live test skipped')
            )
        if 'numpy' in item.keywords and not requires_numpy():
            item.add_marker(pytest.mark.skip(reason='numpy not available — test skipped'))
