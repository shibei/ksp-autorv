import numpy as np

MU_KERBIN = 3.5316e12
R_KERBIN = 600_000
G0 = 9.81


def test_kepler_epoch_circular():
    from krpc_rendezvous.common.orbit_utils import kepler_epoch

    E = kepler_epoch(M=1.5, e=0.0)
    assert abs(E - 1.5) < 1e-8


def test_kepler_epoch_low_eccentricity():
    from krpc_rendezvous.common.orbit_utils import kepler_epoch

    E = kepler_epoch(M=2.0, e=0.1)
    assert abs(E - 0.1 * np.sin(E) - 2.0) < 1e-8


def test_orbital_period_80km():
    from krpc_rendezvous.common.orbit_utils import orbital_period

    sma = R_KERBIN + 80_000
    T = orbital_period(sma, MU_KERBIN)
    assert 1820 < T < 1900


def test_true_anomaly_from_mean_circular():
    from krpc_rendezvous.common.orbit_utils import true_anomaly_from_mean

    f = true_anomaly_from_mean(M=1.0, e=0.0)
    assert abs(f - 1.0) < 1e-8


def test_lambert_circular_coplanar():
    from krpc_rendezvous.common.orbit_utils import lambert_universal, orbital_period

    R0 = R_KERBIN + 100_000
    r1 = np.array([R0, 0.0, 0.0])
    r2 = np.array([R0 * np.cos(np.pi / 3), R0 * np.sin(np.pi / 3), 0.0])
    dt = orbital_period(R0, MU_KERBIN) / 6
    v1, v2 = lambert_universal(r1, r2, dt, MU_KERBIN)
    assert np.all(np.isfinite(v1))
    assert np.all(np.isfinite(v2))
    v_circular = np.sqrt(MU_KERBIN / R0)
    dv = np.linalg.norm(v1) + np.linalg.norm(v2) - 2 * v_circular
    assert dv < 500.0


def test_orbital_position_circular():
    from krpc_rendezvous.common.orbit_utils import orbital_position

    r, v = orbital_position(
        inc=0.0,
        lan=0.0,
        arg_pe=0.0,
        sma=R_KERBIN + 100_000,
        ecc=0.0,
        true_anomaly=0.0,
        mu=MU_KERBIN,
    )
    assert abs(r[0] - (R_KERBIN + 100_000)) < 1.0
    assert abs(v[1] - np.sqrt(MU_KERBIN / (R_KERBIN + 100_000))) < 1.0


def test_cartesian_to_kepler_roundtrip():
    from krpc_rendezvous.common.orbit_utils import cartesian_to_kepler, orbital_position

    R0 = R_KERBIN + 100_000
    v_circ = np.sqrt(MU_KERBIN / R0)
    r_in = np.array([R0, 0.0, 0.0])
    v_in = np.array([0.0, v_circ, 0.0])
    el = cartesian_to_kepler(r_in, v_in, MU_KERBIN)
    r_out, v_out = orbital_position(
        el['inc'], el['lan'], el['arg_pe'], el['sma'], el['ecc'], el['true_anomaly'], MU_KERBIN
    )
    assert np.allclose(r_in, r_out, atol=1.0)
    assert np.allclose(v_in, v_out, atol=0.1)


def test_gravity_turn_profile_shape():
    from krpc_rendezvous.common.orbit_utils import gravity_turn_pitch_profile

    curve = gravity_turn_pitch_profile(
        h=10000, h_start=4000, h_end=70000, pitch_end=np.radians(5.0)
    )
    pitch_deg = np.degrees(curve)
    assert 85 < pitch_deg < 90


def test_delta_v_estimate():
    from krpc_rendezvous.common.orbit_utils import delta_v_estimate

    dv = delta_v_estimate(isp=[300.0], m0=50000, mf=10000)
    expected = 300 * G0 * np.log(5)
    assert abs(dv - expected) < 1.0
