"""Tests for kRPC connection manager (using mock)."""

from unittest.mock import MagicMock, patch

from krpc_rendezvous.common.krpc_connection import (
    KrpcConnection,
    connect_krpc,
    safe_warp,
)


def test_krpc_connection_singleton():
    c1 = KrpcConnection.get_instance()
    c2 = KrpcConnection.get_instance()
    assert c1 is c2


@patch('krpc.connect')
def test_connect_krpc(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    connect_krpc(name='test')
    mock_connect.assert_called_once_with(
        name='test', address='172.17.64.1', rpc_port=50000, stream_port=50001
    )


def test_safe_warp_signature():
    import inspect

    sig = inspect.signature(safe_warp)
    params = list(sig.parameters.keys())
    assert 'target_ut' in params
    assert 'margin' in params
    assert 'max_rate' in params
