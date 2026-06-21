import numpy as np

from epentry.engine import Box, Particle, simulate_multiple, simulate_walk


def test_Particle():
    group = 42
    radius = 0.69
    center = np.array([1.0, 2.0, 3.0])
    particle1 = Particle(group=group, radius=radius, center=center)
    assert particle1.group == group
    assert np.isclose(particle1.radius, radius)
    assert np.allclose(particle1.center, center)
    particle2 = Particle(0, radius=0.1, center=np.zeros(3))
    assert particle1.is_overlap(particle1)
    assert particle2.is_overlap(particle2)
    assert not particle1.is_overlap(particle2)


def test_Box_feasible():
    # Initialization
    rs = np.array([1.0, 0.5, 0.25])
    vfs = np.array([0.1, 0.01, 0.001])
    Nt = 1000
    box = Box(rs, vfs, Nt)
    assert np.allclose(box.rs, rs)
    assert np.allclose(box.vfs_target, vfs)
    assert box.Nt_target == Nt
    assert np.allclose(box.vfs, 0.0)
    assert np.allclose(box.Ns, 0)
    assert box.Nt == 0
    assert np.isclose(box.Lbox, 0.0)
    assert len(box.particles) == 0
    assert not box.success_rsa
    # RSA (all particles should be placed successfully)
    success_rsa = box.rsa()
    assert success_rsa
    assert box.success_rsa
    assert box.Nt == Nt
    assert len(box.particles) == Nt
    assert not np.allclose(box.particles[0].center, box.particles[-1].center)
    assert np.isclose(
        sum([p.volume for p in box.particles]) / box.Lbox**3,
        sum(box.vfs),
    )


def test_Box_infeasible():
    # This should break due to high volume fraction
    box = Box(np.array([1.0]), np.array([0.7]), 100)
    _ = box.rsa()
    assert not box.success_rsa
    assert box.Nt > 0
    assert box.Nt < box.Nt_target
    assert len(box.particles) == box.Nt


def test_simulate_walk():
    box = Box(np.array([1.0]), np.array([0.3]), 100)
    _ = box.rsa()
    walk = simulate_walk(box)
    assert walk.success
    assert isinstance(walk.trajectory, np.ndarray) and walk.trajectory.shape[1] == 3
    assert isinstance(walk.particle_hit, Particle)
    assert walk.time > 0.0


def test_simulate_multiple():
    vf = 0.01
    walks = simulate_multiple(
        rs=np.array([1.0]),
        vfs=np.array([vf]),
        number_boxes=10,
        number_particles_per_box=200,
        number_walks_per_box=200,
        D=1.0,
    )
    assert isinstance(walks, list)
    assert len(walks) == 10 * 200
    times = [walk.time for walk in walks if walk.success]
    time_mean = np.mean(times)
    enhancement_factor = 1 / (time_mean * 3 * vf * (1.0 - vf))
    assert np.isclose(enhancement_factor, 1.0, rtol=1.0)
