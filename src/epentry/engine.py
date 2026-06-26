import math
from math import pi

import numba as nb
import numpy as np
from numpy.typing import NDArray


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


@nb.experimental.jitclass
class NBox:
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
    centers : NDArray
        Particle center coordinates.
    cell_size : float
        Size of each cell in the cell list.
    groups : NDArray
        Particle group identifiers.
    head : NDArray
        Head of the linked list for each cell in the cell list.
    Lbox : float
        Simulation box length.
    nc : int
        Number of cells per dimension.
    next : NDArray
        Next particle index for each particle in the cell list.
    Ns : NDArray
        Actual number of particles per group after placement.
    Nt : int
        Actual total number of particles after placement.
    Nt_target : int
        Target total number of particles.
    radii : NDArray
        Particle radii.
    rmin : float
        Minimum particle radius.
    rmax : float
        Maximum particle radius.
    rs : NDArray
        Particle radii.
    success_rsa : bool
        Status of random sequential addition procedure. `False` if not all particles
        could be inserted in the box.
    vfs_target : NDArray
        Target volume fractions.
    """

    rs: nb.float64[:]
    rmin: nb.float64
    rmax: nb.float64
    vfs_target: nb.float64[:]
    Nt_target: nb.int64
    Ns: nb.int64[:]
    Nt: nb.int64
    Lbox: nb.float64
    centers: nb.float64[:, :]
    radii: nb.float64[:]
    groups: nb.int64[:]
    success_rsa: bool
    head: nb.int64[:]
    next: nb.int64[:]
    cell_size: nb.float64
    nc: nb.int64

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
        self.rmin = np.min(self.rs)
        self.rmax = np.max(self.rs)

        # Initialize actual values to zero until particles are placed
        self.Ns = np.zeros(len(self.rs), dtype=np.int64)
        self.Nt = 0
        self.Lbox = 0.0
        self.radii = np.zeros(Nt, dtype=np.float64)
        self.centers = np.zeros((Nt, 3), dtype=np.float64)
        self.groups = np.zeros(Nt, dtype=np.int64)
        self.success_rsa = False
        self.cell_size = 0.0
        self.nc = 0
        self.head = np.empty(0, np.int64)
        self.next = np.empty(0, np.int64)

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


@nb.njit(fastmath=True)
def rsa(box: NBox) -> bool:
    r"""
    Generate a non-overlapping particle ensemble using random sequential addition.

    Particles are placed uniformly at random in a cubic simulation box. Candidate
    positions that overlap previously placed particles are rejected until a valid
    position is found.

    Failure to place all particles is likely at a high total volume fraction,
    which makes it difficult to find non-overlapping positions for all particles.
    In this case, the box will contain as many particles as possible given the
    requested volume fractions and total particle count.

    Parameters
    ----------
    box : Box
        Box object. Updated in-place with the particle ensemble.

    Returns
    -------
    bool
        `True` if all particles were successfully placed without overlap, `False`
        otherwise.
    """
    # Target particle group values
    rs = box.rs
    vfs = box.vfs_target
    Nt = box.Nt_target

    # Number density of each particle group and total
    ns = vfs / (4.0 / 3.0 * pi * rs**3)
    nt = ns.sum()

    # Tentative particle counts of each group
    Ns = np.rint(Nt * ns / nt).astype(np.int64)
    Nt = min(Ns.sum(), Nt)

    # Box length
    box.Lbox = (Nt / nt) ** (1 / 3)

    # Place particles sequentially without overlap
    abort = False
    max_attempts = 100 * Nt  # heuristic
    Ns_actual = np.zeros_like(Ns)
    groups = np.repeat(np.arange(rs.size), Ns)
    for k in range(Nt):
        i = groups[k]
        box.groups[k] = i
        box.radii[k] = rs[i]
        attempts = 0
        while True:
            box.centers[k, 0] = np.random.uniform(0.0, box.Lbox)
            box.centers[k, 1] = np.random.uniform(0.0, box.Lbox)
            box.centers[k, 2] = np.random.uniform(0.0, box.Lbox)
            attempts += 1
            overlap = False
            for j in range(k):
                overlap = particles_overlap(box, k, j)
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

    box.success_rsa = not abort
    box.Ns = Ns_actual
    box.Nt = Ns_actual.sum()
    box.centers = box.centers[: box.Nt, :].copy()
    box.radii = box.radii[: box.Nt].copy()
    box.groups = box.groups[: box.Nt].copy()

    build_cell_list(box)

    return box.success_rsa


@nb.njit(fastmath=True)
def build_cell_list(box: NBox) -> None:
    """Build a cell list for the particles in the box.

    Builds a cell list for the particles in the box to accelerate neighbor searches. The
    cell list is a 3D grid of cubic cells that covers the entire box. Each particle is
    assigned to a cell based on its center coordinates. The head of the linked list for
    each cell is stored in `box.head`, and the next particle index for each particle is
    stored in `box.next`. The cell size is set to twice the maximum particle radius to
    ensure that all potential overlaps are captured.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble. Updated in-place with the cell list
        data structures.
    """
    h = 2.0 * box.rmax
    nc = max(1, int(np.floor(box.Lbox / h)))
    ncells_total = nc**3

    head = -np.ones(ncells_total, dtype=np.int64)
    next = -np.ones(box.Nt, dtype=np.int64)

    for i in range(box.Nt):
        x = box.centers[i, 0]
        y = box.centers[i, 1]
        z = box.centers[i, 2]

        c = cell_index(x, y, z, h, nc)

        next[i] = head[c]
        head[c] = i

    box.cell_size = h
    box.nc = nc
    box.head = head
    box.next = next


@nb.njit(inline="always")
def point_inside_box(box: NBox, point: NDArray) -> bool:
    r"""
    Check if a point is inside the box.

    Parameters
    ----------
    box: NBox
        Box object containing the particle ensemble.
    point : NDArray
        Point coordinates.

    Returns
    -------
    bool
        `True` if the point is inside the box, `False` otherwise.
    """
    return (
        0.0 <= point[0] <= box.Lbox
        and 0.0 <= point[1] <= box.Lbox
        and 0.0 <= point[2] <= box.Lbox
    )


@nb.njit(inline="always")
def distance_to_nearest_wall(box: NBox, point: NDArray) -> float:
    r"""
    Compute the distance from a point to the nearest wall of the box.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
    point : NDArray
        Point coordinates.

    Returns
    -------
    float
        Distance from `point` to the nearest wall of the box.
    """
    Lbox = box.Lbox
    return min(
        point[0],
        Lbox - point[0],
        point[1],
        Lbox - point[1],
        point[2],
        Lbox - point[2],
    )


@nb.njit(inline="always")
def point_inside_particle(box: NBox, i: int, point: NDArray, rtol: float = 0.0) -> bool:
    r"""
    Check if a point is inside a particle.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
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
    dx = point[0] - box.centers[i, 0]
    dy = point[1] - box.centers[i, 1]
    dz = point[2] - box.centers[i, 2]
    return (dx**2 + dy**2 + dz**2) <= ((1.0 + rtol) * box.radii[i]) ** 2


@nb.njit(fastmath=True)
def _point_inside_any_particle(
    box: NBox, point: NDArray, rtol: float = 0.0
) -> tuple[bool, int]:
    r"""
    Check if a point is inside any particle.

    Note
    ----
    This is a naive O(N) implementation that checks all particles. It is kept for
    reference and testing purposes. The `point_inside_any_particle` method implements
    a more efficient O(1) algorithm using a cell list.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
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
    for i in range(box.Nt):
        if point_inside_particle(box, i, point, rtol):
            return (True, i)
    return (False, -1)


@nb.njit(fastmath=True)
def point_inside_any_particle(
    box: NBox, point: NDArray, rtol: float = 0.0
) -> tuple[bool, int]:
    r"""
    Check if a point is inside any particle.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
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
    h = box.cell_size
    nc = box.nc

    ix = min(max(int(point[0] / h), 0), nc - 1)
    iy = min(max(int(point[1] / h), 0), nc - 1)
    iz = min(max(int(point[2] / h), 0), nc - 1)

    for dx in (-1, 0, 1):
        nx = ix + dx

        if nx < 0 or nx >= nc:
            continue

        for dy in (-1, 0, 1):
            ny = iy + dy

            if ny < 0 or ny >= nc:
                continue

            for dz in (-1, 0, 1):
                nz = iz + dz

                if nz < 0 or nz >= nc:
                    continue

                c = nx + nc * (ny + nc * nz)
                p = box.head[c]

                while p != -1:
                    if point_inside_particle(box, p, point, rtol):
                        return True, p
                    p = box.next[p]

    return False, -1


@nb.njit(inline="always")
def particles_overlap(box: NBox, i: int, j: int) -> bool:
    r"""
    Check if two particles overlap.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
    i : int
        Index of the first particle.
    j : int
        Index of the second particle.

    Returns
    -------
    bool
        `True` if the particles overlap, `False` otherwise.
    """
    dx = box.centers[i, 0] - box.centers[j, 0]
    dy = box.centers[i, 1] - box.centers[j, 1]
    dz = box.centers[i, 2] - box.centers[j, 2]
    return (dx**2 + dy**2 + dz**2) <= (box.radii[i] + box.radii[j]) ** 2


@nb.njit(fastmath=True)
def _clearance_radius(box: NBox, point: NDArray) -> float:
    r"""
    Compute the radius of the largest sphere centered at `point` that does not
    overlap any particle.

    Note
    ----
    This is a naive O(N) implementation that checks all particles. It is kept for
    reference and testing purposes. The `clearance_radius` method implements a more
    efficient O(1) algorithm using a cell list.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
    point : NDArray
        Point coordinates.

    Returns
    -------
    float
        Clearance radius at `point`.
    """
    R = box.Lbox
    for i in range(box.Nt):
        dx = point[0] - box.centers[i, 0]
        dy = point[1] - box.centers[i, 1]
        dz = point[2] - box.centers[i, 2]
        s = math.sqrt(dx**2 + dy**2 + dz**2) - box.radii[i]
        if s < R:
            R = s
    return R


@nb.njit(fastmath=True)
def clearance_radius(box: NBox, point: NDArray) -> float:
    r"""
    Compute the radius of the largest sphere centered at `point`
    that does not overlap any particle.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
    point : NDArray
        Point coordinates.

    Returns
    -------
    float
        Clearance radius at `point`.
    """
    h = box.cell_size
    nc = box.nc

    ix = min(max(int(point[0] / h), 0), nc - 1)
    iy = min(max(int(point[1] / h), 0), nc - 1)
    iz = min(max(int(point[2] / h), 0), nc - 1)

    R = box.Lbox

    for shell in range(nc):
        xmin = max(ix - shell, 0)
        xmax = min(ix + shell, nc - 1)

        ymin = max(iy - shell, 0)
        ymax = min(iy + shell, nc - 1)

        zmin = max(iz - shell, 0)
        zmax = min(iz + shell, nc - 1)

        for nx in range(xmin, xmax + 1):
            for ny in range(ymin, ymax + 1):
                for nz in range(zmin, zmax + 1):
                    # only visit cells on the shell boundary
                    if (
                        nx != xmin
                        and nx != xmax
                        and ny != ymin
                        and ny != ymax
                        and nz != zmin
                        and nz != zmax
                    ):
                        continue

                    c = nx + nc * (ny + nc * nz)
                    p = box.head[c]

                    while p != -1:
                        dx = point[0] - box.centers[p, 0]
                        dy = point[1] - box.centers[p, 1]
                        dz = point[2] - box.centers[p, 2]

                        s = math.sqrt(dx * dx + dy * dy + dz * dz) - box.radii[p]
                        if s < R:
                            R = s

                        p = box.next[p]

        # no unseen particle can improve R
        if shell * h - box.rmax > R:
            break

    return R


@nb.njit(inline="always")
def cell_index(x: float, y: float, z: float, h: float, nc: int) -> int:
    """Return the cell index for a point in a cubic grid of cells.

    Parameters
    ----------
    x, y, z : float
        Coordinates of the point.
    h : float
        Cell size.
    nc : int
        Number of cells along each dimension.
    """
    ix = min(max(int(x / h), 0), nc - 1)
    iy = min(max(int(y / h), 0), nc - 1)
    iz = min(max(int(z / h), 0), nc - 1)

    return ix + nc * (iy + nc * iz)


@nb.njit(fastmath=True, inline="always")
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


@nb.njit(fastmath=True)
def simulate_walk(
    box: NBox, D: float, rtol: float = 1e-3, maxsteps: int = 100
) -> WalkResult:
    r"""
    Simulate a random walk of a point particle in a box containing an ensemble of
    non-overlapping spheres.

    Notes
    -----
    * There are "wall effects" because the random walker is not allowed to step outside
      the box and we do not have periodic boundary conditions.

    Parameters
    ----------
    box : NBox
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
    # Check that the box has been filled with particles
    if box.Nt == 0:
        raise ValueError("Box has no particles. Run rsa() to fill the box first.")

    # Find a free initial position for the radical
    X = np.empty(3, dtype=np.float64)
    Lbox = box.Lbox
    while True:
        X[0] = np.random.uniform(0.0, Lbox)
        X[1] = np.random.uniform(0.0, Lbox)
        X[2] = np.random.uniform(0.0, Lbox)

        inside, _ = point_inside_any_particle(box, X)
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
        R = clearance_radius(box, X)
        if R <= box.rmin * rtol / 10:
            break  # If the clearance radius is too small, terminate the walk

        # Move to random point on the sphere of radius R centered about X
        # Could be improved with period boundary conditions to avoid rejection sampling
        # near the walls. To be done.
        while True:
            Xtry[:] = random_point_sphere(X, R)
            if point_inside_box(box, Xtry):
                break
        X[:] = Xtry
        trajectory[step, :] = X
        time += R**2 / (6.0 * D)  # Mean first-passage time

        # Check if point is numerically close to any particle surface
        success, idx_particle = point_inside_any_particle(box, X, rtol)

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


# @nb.njit
def simulate_multiple(
    rs: NDArray,
    vfs: NDArray,
    number_boxes: int,
    number_particles_per_box: int,
    number_walks_per_box: int,
    D: float = 1.0,
) -> list[WalkResult]:
    r"""
    Simulate multiple random walks in across multiple boxes.

    Multiple boxes are generated with the same target particle group parameters but
    different random placements of particles. Multiple random walks are simulated
    in each box.

    Notes
    -----
    I should refactor to return the boxes as well, because certain properrties are
    easier to get from the box.

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
        box = NBox(rs, vfs, number_particles_per_box)
        success_rsa = False
        for _ in range(10):
            success_rsa = rsa(box)
            if success_rsa:
                build_cell_list(box)
                break
        if not success_rsa:
            print("  Failed to fill box after 10 attempts. Skipping.")
            continue

        for _ in range(number_walks_per_box):
            walk = simulate_walk(box, D=D)
            walks.append(walk)

    return walks
