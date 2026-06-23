from __future__ import annotations

import math
from math import pi

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.experimental.jitclass
class Box:
    r"""
    Ensemble of particles in a rectangular box.

    Parameters
    ----------
    rs : NDArray
        Particle radii of each group.
    vfs : NDArray
        Target volume fractions of each group.
    Nt : int
        Target total number of particles in the box.

    Attributes
    ----------
    rs : NDArray
        Particle radii.
    vfs_target : NDArray
        Target volume fractions.
    Nt_target : int
        Target total number of particles.
    Ns : NDArray
        Actual number of particles per group after placement.
    Nt : int
        Actual total number of particles after placement.
    Lbox : float
        Simulation box length.
    centers : NDArray
        Particle center coordinates.
    radii : NDArray
        Particle radii.
    groups : NDArray
        Particle group identifiers.
    success_rsa : bool
        Status of random sequential addition procedure. `False` if not all particles
        could be inserted in the box.
    """

    rs: nb.float64[:]
    vfs_target: nb.float64[:]
    Nt_target: nb.int64
    Ns: nb.int64[:]
    Nt: nb.int64
    Lbox: nb.float64
    centers: nb.float64[:, :]
    radii: nb.float64[:]
    groups: nb.int64[:]
    success_rsa: bool

    def __init__(
        self,
        rs: NDArray,
        vfs: NDArray,
        Nt: int,
    ) -> None:

        # Sort the groups by decreasing radius
        rev_idx = np.argsort(rs)[::-1]

        # Store target values for particle groups
        self.rs = np.asarray(rs[rev_idx], dtype=np.float64)
        self.vfs_target = np.asarray(vfs[rev_idx], dtype=np.float64)
        self.Nt_target = Nt

        # Initialize actual values to zero until particles are placed
        self.Ns = np.zeros(len(self.rs), dtype=np.int64)
        self.Nt = 0
        self.Lbox = 0.0
        self.radii = np.zeros(Nt, dtype=np.float64)
        self.centers = np.zeros((Nt, 3), dtype=np.float64)
        self.groups = np.zeros(Nt, dtype=np.int64)
        self.success_rsa = False

    def point_inside_box(self, point: NDArray) -> bool:
        r"""
        Check if a point is inside the box.

        Parameters
        ----------
        point : NDArray
            Point coordinates.

        Returns
        -------
        bool
            `True` if the point is inside the box, `False` otherwise.
        """
        return (
            0.0 <= point[0] <= self.Lbox
            and 0.0 <= point[1] <= self.Lbox
            and 0.0 <= point[2] <= self.Lbox
        )

    def point_inside_particle(
        self,
        i: int,
        point: NDArray,
        rtol: float = 0.0,
    ) -> bool:
        r"""
        Check if a point is inside a particle.

        Parameters
        ----------
        i : int
            Index of the particle.
        point : NDArray
            Point coordinates.
        rtol : float
            Relative tolerance for numerical precision when checking if the point is
            inside the particle. The point is considered inside if it is within `rtol`
            of the particle.

        Returns
        -------
        bool
            `True` if the point is inside the particle, `False` otherwise.
        """
        dx = point[0] - self.centers[i, 0]
        dy = point[1] - self.centers[i, 1]
        dz = point[2] - self.centers[i, 2]
        return (dx**2 + dy**2 + dz**2) <= ((1.0 + rtol) * self.radii[i]) ** 2

    def point_inside_any_particle(
        self,
        point: NDArray,
        rtol: float = 0.0,
    ) -> tuple[bool, int]:
        r"""
        Check if a point is inside any particle.

        Parameters
        ----------
        point : NDArray
            Point coordinates.
        rtol : float
            Relative tolerance for numerical precision when checking if the point is
            inside the particle. The point is considered inside if it is within `rtol`
            of the particle.

        Returns
        -------
        tuple[bool, int]:
            `True` if the point is inside any particle, `False` otherwise. The index of
            the particle that the point is inside, or -1 if the point is not inside any
            particle.
        """
        for i in range(self.Nt):
            if self.point_inside_particle(i, point, rtol):
                return (True, i)
        return (False, -1)

    def particles_overlap(
        self,
        i: int,
        j: int,
    ) -> bool:
        r"""
        Check if two particles overlap.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.

        Returns
        -------
        bool
            `True` if the particles overlap, `False` otherwise.
        """
        dx = self.centers[i, 0] - self.centers[j, 0]
        dy = self.centers[i, 1] - self.centers[j, 1]
        dz = self.centers[i, 2] - self.centers[j, 2]
        return (dx**2 + dy**2 + dz**2) <= (self.radii[i] + self.radii[j]) ** 2

    def clearance_radius(
        self,
        point: NDArray,
    ) -> float:
        r"""
        Compute the radius of the largest sphere centered at `point` that does not
        overlap any particle.

        Parameters
        ----------
        point : NDArray
            Point coordinates.

        Returns
        -------
        float
            Clearance radius at `point`.
        """
        R = self.Lbox
        for i in range(self.Nt):
            dx = point[0] - self.centers[i, 0]
            dy = point[1] - self.centers[i, 1]
            dz = point[2] - self.centers[i, 2]
            s = math.sqrt(dx**2 + dy**2 + dz**2) - self.radii[i]
            if s < R:
                R = s
        return R

    def distance_to_nearest_wall(
        self,
        point: NDArray,
    ) -> float:
        r"""
        Compute the distance from a point to the nearest wall of the box.

        Parameters
        ----------
        point : NDArray
            Point coordinates.

        Returns
        -------
        float
            Distance from `point` to the nearest wall of the box.
        """
        Lbox = self.Lbox
        return min(
            point[0],
            Lbox - point[0],
            point[1],
            Lbox - point[1],
            point[2],
            Lbox - point[2],
        )

    def vfs(self) -> NDArray:
        r"""
        Compute the approximate volume fractions of each particle group in the box.

        Note
        ----
        Particles close to the wall will have a fraction of their volume outside the box.
        This 'lost' volume is not accounted for in this calculation. To be done.

        Returns
        -------
        NDArray
            Actual volume fractions of each particle group.
        """
        if self.Nt == 0:
            return np.zeros_like(self.rs)
        else:
            Ns = self.Ns.astype(np.float64)
            return Ns * (4.0 / 3.0 * pi * self.rs**3) / self.Lbox**3

    def rsa(self) -> bool:
        r"""
        Generate a non-overlapping particle ensemble using random sequential addition.

        Particles are placed uniformly at random in a cubic simulation box. Candidate
        positions that overlap previously placed particles are rejected until a valid
        position is found.

        Failure to place all particles is likely at a high total volume fraction,
        which makes it difficult to find non-overlapping positions for all particles.
        In this case, the box will contain as many particles as possible given the
        requested volume fractions and total particle count.

        Returns
        -------
        bool
            `True` if all particles were successfully placed without overlap, `False`
            otherwise.
        """
        # Target particle group values
        rs = self.rs
        vfs = self.vfs_target
        Nt = self.Nt_target

        # Number density of each particle group and total
        ns = vfs / (4.0 / 3.0 * pi * rs**3)
        nt = ns.sum()

        # Tentative particle counts of each group
        Ns = np.rint(Nt * ns / nt).astype(np.int64)
        Nt = min(Ns.sum(), Nt)

        # Box length
        self.Lbox = (Nt / nt) ** (1 / 3)

        # Place particles sequentially without overlap
        abort = False
        max_attempts = 100 * Nt
        Ns_actual = np.zeros_like(Ns)
        groups = np.repeat(np.arange(rs.size), Ns)
        for k in range(Nt):
            i = groups[k]
            self.groups[k] = i
            self.radii[k] = rs[i]
            attempts = 0
            while True:
                self.centers[k, 0] = np.random.uniform(0.0, self.Lbox)
                self.centers[k, 1] = np.random.uniform(0.0, self.Lbox)
                self.centers[k, 2] = np.random.uniform(0.0, self.Lbox)
                attempts += 1
                overlap = False
                for j in range(k):
                    overlap = self.particles_overlap(k, j)
                    if overlap:
                        break
                if not overlap:
                    break
                if attempts > max_attempts:
                    print(
                        f"Failed to find non-overlapping location for particle {k}/{Nt} after {max_attempts} attempts.\n"  # noqa: E501
                        "Total particle volume fraction is probably too high."
                    )
                    abort = True
                    break
            if abort:
                break

            Ns_actual[i] += 1

        self.success_rsa = not abort
        self.Ns = Ns_actual
        self.Nt = Ns_actual.sum()
        self.centers = self.centers[: self.Nt, :].copy()
        self.radii = self.radii[: self.Nt].copy()
        self.groups = self.groups[: self.Nt].copy()

        return self.success_rsa


@nb.experimental.jitclass
class WalkResult:
    r"""
    Result of a random walk of a point particle in a box containing an ensemble of
    non-overlapping spheres.

    Parameters
    ----------
    success : bool
        `True` if the random walker collided with a particle within the maximum number of
        steps, `False` otherwise.
    time : float
        Time taken for the walk.
    trajectory : NDArray
        Random walk trajectory.
    particle_group : int
        Group identifier of the particle that the random walker collided with.
    particle_radius : float
        Radius of the particle that the random walker collided with.
    particle_center : NDArray
        Center coordinates of the particle that the random walker collided with.
    """

    success: bool
    time: float
    trajectory: nb.float64[:, :]
    particle_group: nb.int64
    particle_radius: nb.float64
    particle_center: nb.float64[:]

    def __init__(
        self,
        success: bool,
        time: float,
        trajectory: NDArray,
        particle_group: int,
        particle_radius: float,
        particle_center: NDArray,
    ) -> None:
        self.success = success
        self.time = time
        self.trajectory = trajectory
        self.particle_group = particle_group
        self.particle_radius = particle_radius
        self.particle_center = particle_center


@nb.njit(fastmath=True)
def simulate_walk(
    box: Box,
    D: float = 1.0,
    rtol: float = 1e-3,
    maxsteps: int = 100,
) -> WalkResult:
    r"""
    Simulate a random walk of a point particle in a box containing an ensemble of
    non-overlapping spheres.

    Notes
    -----
    * The algorithm is O(N²), which is not ideal for large particle counts. Can be
      improved by implementing cell lists. To be done.
    * There are "wall effects" because the random walker is not allowed to step outside
      the box and we do not have periodic boundary conditions.

    Parameters
    ----------
    box : Box
        Box containing the particle ensemble.
    D : float
        Diffusion coefficient of the random walker. This only affects the time associated
        with the walk and not the trajectory itself.
    rtol : float
        Relative tolerance for numerical precision when checking if the random walker has
        collided with a particle. The walker is considered to have collided if it is
        within `rtol` of the particle surface.
    maxsteps : int
        Maximum number of steps for the random walk. If the walker does not collide with a
        particle within this number of steps, the walk will be terminated and considered
        unsuccessful.

    Returns
    -------
    WalkResult
        Object containing the random walk trajectory, and information about the particle
        hit (if any) and time taken for the walk.
    """
    # Find a free initial position for the radical
    X = np.empty(3, dtype=np.float64)
    Lbox = box.Lbox
    while True:
        X[0] = np.random.uniform(0.0, Lbox)
        X[1] = np.random.uniform(0.0, Lbox)
        X[2] = np.random.uniform(0.0, Lbox)

        inside = False
        for i in range(box.Nt):
            inside = box.point_inside_particle(i, X)
            if inside:
                break
        if not inside:
            break

    # Preallocate trajectory array with a maximum number of steps
    trajectory = np.empty((maxsteps + 1, 3), dtype=np.float64)
    trajectory[0, :] = X

    # Main random walking loop
    time = 0.0
    success = False
    idx_particle = -1
    Xtry = np.empty(3, dtype=np.float64)
    for step in range(1, maxsteps + 1):
        # Find radius of largest sphere centered about X
        # R = min(box.clearance_radius(X), box.distance_to_nearest_wall(X))
        R = box.clearance_radius(X)

        # Move to random point on the sphere of radius R centered about X
        # Could be improved with period boundary conditions to avoid rejection sampling
        # near the walls. To be done.
        while True:
            Xtry[:] = random_point_sphere(X, R)
            if box.point_inside_box(Xtry):
                break
        X[:] = Xtry
        trajectory[step, :] = X
        time += R**2 / (6.0 * D)

        # Check if point is numerically close to any particle surface
        success, idx_particle = box.point_inside_any_particle(X, rtol)

        if success:
            break

    if success:
        particle_group = box.groups[idx_particle]
        particle_radius = box.radii[idx_particle]
        particle_center = box.centers[idx_particle, :]
    else:
        particle_group = -1
        particle_radius = -1.0
        particle_center = np.zeros(3, dtype=np.float64)

    return WalkResult(
        success,
        time,
        trajectory[: step + 1],
        particle_group,
        particle_radius,
        particle_center,
    )


@nb.njit(fastmath=True)
def random_point_sphere(center: NDArray, radius: float) -> NDArray:
    """
    Generate a random point on the surface of a sphere.

    Parameters
    ----------
    center : NDArray
        Sphere center coordinates.
    radius : float
        Sphere radius.

    Returns
    -------
    NDArray
        Random point on the surface of the sphere.
    """
    z = np.random.uniform(-1.0, 1.0)
    ϕ = np.random.uniform(0, 2.0 * pi)
    xy_radius = math.sqrt(max(0.0, 1.0 - z * z))

    point = np.empty(3, dtype=np.float64)
    point[0] = center[0] + radius * xy_radius * math.cos(ϕ)
    point[1] = center[1] + radius * xy_radius * math.sin(ϕ)
    point[2] = center[2] + radius * z

    return point


# @nb.njit
def simulate_multiple(
    rs: NDArray,
    vfs: NDArray,
    number_boxes: int,
    number_particles_per_box: int,
    number_walks_per_box: int,
    D: float = 1.0,
) -> list[WalkResult]:
    """
    Simulate multiple random walks in across multiple boxes.

    Multiple boxes are generated with the same target particle group parameters but
    different random placements of particles. Multiple random walks are simulated
    in each box.

    Notes
    -----
    I should refactor to return the boxes as well, because certain properrties are easier
    to get from the box.

    Parameters
    ----------
    rs : NDArray
        Particle radii of each group.
    vfs : NDArray
        Target volume fractions of each group.
    number_boxes : int
        Number of boxes to simulate in parallel.
    number_particles_per_box : int
        Number of particles to place in each box.
    number_walks_per_box : int
        Number of random walks to simulate in each box.
    D : float
        Diffusion coefficient of the random walker.

    Returns
    -------
    list[Walk]
        List of random walks simulated across all boxes.
    """
    walks = []
    for i in range(number_boxes):
        # RSA can fail at high volume fractions, so we try multiple times
        box = Box(rs, vfs, number_particles_per_box)
        success_rsa = False
        for _ in range(10):
            success_rsa = box.rsa()
            if success_rsa:
                break
        if not success_rsa:
            print("  Failed to fill box after 10 attempts. Skipping.")
            continue

        for _ in range(number_walks_per_box):
            walk = simulate_walk(box, D=D)
            walks.append(walk)

    return walks
