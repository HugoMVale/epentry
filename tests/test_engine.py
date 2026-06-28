import numpy as np
import pytest

import epentry.engine as engine
from epentry import Box


def test_RSA_feasible():
    # Initialization
    rs = [1.0, 0.5, 0.25]
    vfs = [0.1, 0.01, 0.001]
    Nt = 1000
    box = Box(rs, vfs, Nt)
    assert np.allclose(box._nbox.rs, rs)
    assert np.allclose(box._nbox.vfs_target, vfs)
    assert box._nbox.Nt_target == Nt
    assert np.allclose(box._nbox.vfs(), 0.0)
    assert np.allclose(box._nbox.Ns, 0)
    assert box._nbox.Nt == 0
    assert np.isclose(box._nbox.length, 0.0)
    assert not box._nbox.success
    # RSA (all particles should be placed successfully)
    for periodic in [False, True]:
        success_rsa = box.place_particles(method="RSA", periodic=periodic)
        assert success_rsa
        assert box._nbox.success
        assert box._nbox.Nt == Nt
        assert len(box._nbox.centers) == Nt
        assert len(box._nbox.groups) == Nt
        assert len(box._nbox.radii) == Nt
        assert not np.allclose(box._nbox.centers[0], box._nbox.centers[-1])


def test_RSA_infeasible():
    # This should break due to high volume fraction
    box = Box([1.0], [0.7], 100)
    for periodic in [False, True]:
        box.place_particles(method="RSA", periodic=periodic)
        assert not box._nbox.success
        assert box._nbox.Nt > 0
        assert box._nbox.Nt < box._nbox.Nt_target
        assert len(box._nbox.centers) == box._nbox.Nt
        assert len(box._nbox.groups) == box._nbox.Nt
        assert len(box._nbox.radii) == box._nbox.Nt


def test_BCC_feasible():
    Nt = 100
    vf = 0.321456
    box = Box([1.0], [vf], Nt)
    success_bcc = box.place_particles(method="BCC")
    assert success_bcc
    assert box._nbox.success
    assert len(box._nbox.centers) == box._nbox.Nt
    assert len(box._nbox.groups) == box._nbox.Nt
    assert len(box._nbox.radii) == box._nbox.Nt
    assert np.allclose(box._nbox.vfs(), [vf])


def test_BCC_infeasible():
    Nt = 100
    vf = 0.7
    box = Box([1.0], [vf], Nt)
    with pytest.raises(ValueError):
        box.place_particles(method="BCC")


def test_cell_build_RSA():
    box = engine.NBox(np.array([1.0]), np.array([0.3]), 1000)
    for periodic in [False, True]:
        engine.rsa(box, periodic)
        X = np.random.uniform(0, box.length, size=(1000, 3))
        once_inside = False
        once_outside = False
        for x in X:
            inside1, idx1 = engine.point_inside_any_particle(box, x)
            inside2, idx2 = engine._point_inside_any_particle(box, x)
            assert inside1 == inside2
            assert idx1 == idx2
            if inside1:
                once_inside = True
            if not inside1:
                R1 = engine.clearance_radius(box, x)
                R2 = engine._clearance_radius(box, x)
                assert np.isclose(R1, R2)
                once_outside = True
        assert once_inside
        assert once_outside


def test_Equilibrium():
    box = Box([1.0], [0.63], Nt=100)
    box.place_particles(method="Equilibrium")
    nbox = box._nbox
    for i in range(len(nbox.centers)):
        for j in range(i + 1, len(nbox.centers)):
            assert not engine.particles_overlap(nbox, i, j)


def test_simulate_walk():
    box = Box([1.0], [0.3], Nt=100)
    for periodic in [False, True]:
        box.place_particles(method="RSA", periodic=periodic)
        walk = box.simulate_walk(D=1e-8)
        assert walk.success
        assert isinstance(walk.trajectory, np.ndarray) and walk.trajectory.shape[1] == 3
        assert walk.particle_group >= 0
        assert walk.time > 0.0


# def test_simulate_multiple():
#     vf = 0.01
#     walks = simulate_multiple(
#         rs=np.array([1.0]),
#         vfs=np.array([vf]),
#         number_boxes=10,
#         number_particles_per_box=200,
#         number_walks_per_box=200,
#         D=1.0,
#     )
#     assert isinstance(walks, list)
#     assert len(walks) == 10 * 200
#     times = [walk.time for walk in walks if walk.success]
#     time_mean = np.mean(times)
#     enhancement_factor = 1 / (time_mean * 3 * vf * (1.0 - vf))
#     assert np.isclose(enhancement_factor, 1.0, rtol=1.0)
