import numpy as np

from epentry.engine import Box, simulate_multiple, simulate_walk


def test_Box_feasible():
    # Initialization
    rs = np.array([1.0, 0.5, 0.25])
    vfs = np.array([0.1, 0.01, 0.001])
    Nt = 1000
    box = Box(rs, vfs, Nt)
    assert np.allclose(box.rs, rs)
    assert np.allclose(box.vfs_target, vfs)
    assert box.Nt_target == Nt
    assert np.allclose(box.vfs(), 0.0)
    assert np.allclose(box.Ns, 0)
    assert box.Nt == 0
    assert np.isclose(box.Lbox, 0.0)
    assert not box.success_rsa
    # RSA (all particles should be placed successfully)
    success_rsa = box.rsa()
    assert success_rsa
    assert box.success_rsa
    assert box.Nt == Nt
    assert len(box.centers) == Nt
    assert len(box.groups) == Nt
    assert len(box.radii) == Nt
    assert not np.allclose(box.centers[0], box.centers[-1])


def test_Box_infeasible():
    # This should break due to high volume fraction
    box = Box(np.array([1.0]), np.array([0.7]), 100)
    _ = box.rsa()
    assert not box.success_rsa
    assert box.Nt > 0
    assert box.Nt < box.Nt_target
    assert len(box.centers) == box.Nt
    assert len(box.groups) == box.Nt
    assert len(box.radii) == box.Nt


def test_simulate_walk():
    box = Box(np.array([1.0]), np.array([0.3]), 100)
    _ = box.rsa()
    walk = simulate_walk(box)
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
