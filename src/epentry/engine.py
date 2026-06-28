import math
from math import pi
from typing import Literal

import numba as nb
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "METHODS",
    "WalkResult",
    "NBox",
    "simulate_multiple",
]


METHOD_RSA = 0
METHOD_BCC = 1
METHOD_EQUIL = 2

METHODS = {
    METHOD_RSA: "Random Sequential Addition",
    METHOD_BCC: "Body-Centered Cubic",
    METHOD_EQUIL: "Equilibrated Hard Spheres",
}


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
    length : float
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
    periodic : bool
        Whether to apply periodic boundary conditions.
    radii : NDArray
        Particle radii.
    rmin : float
        Minimum particle radius.
    rmax : float
        Maximum particle radius.
    rs : NDArray
        Particle radii.
    success : bool
        Status of particle generation procedure.
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
    length: nb.float64
    centers: nb.float64[:, :]
    radii: nb.float64[:]
    groups: nb.int64[:]
    method: nb.int64
    success: bool
    head: nb.int64[:]
    next: nb.int64[:]
    cell_size: nb.float64
    nc: nb.int64
    periodic: bool

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
        self.method = -1
        self.periodic = False
        self.length = 0.0
        self.radii = np.zeros(0, dtype=np.float64)
        self.centers = np.zeros((0, 3), dtype=np.float64)
        self.groups = np.zeros(0, dtype=np.int64)
        self.success = False
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
            return Ns * (4.0 / 3.0 * pi * self.rs**3) / self.length**3


@nb.njit(fastmath=True)
def rsa(box: NBox, periodic: bool = True, cell_list: bool = True) -> bool:
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
    periodic : bool
        Whether to apply periodic boundary conditions.
    cell_list : bool
        Whether to build a cell list for the particle ensemble after placement. This
        is recommended for efficient neighbor searches during random walks.

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
    box.length = (Nt / nt) ** (1 / 3)

    # Set box metadata
    box.method = METHOD_RSA
    box.periodic = periodic

    # Allocate particle arrays
    box.radii = np.zeros(Nt, dtype=np.float64)
    box.centers = np.zeros((Nt, 3), dtype=np.float64)
    box.groups = np.zeros(Nt, dtype=np.int64)

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
            box.centers[k, 0] = np.random.uniform(0.0, box.length)
            box.centers[k, 1] = np.random.uniform(0.0, box.length)
            box.centers[k, 2] = np.random.uniform(0.0, box.length)
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

    # Finalize
    box.success = not abort
    box.Ns = Ns_actual
    box.Nt = Ns_actual.sum()
    box.centers = box.centers[: box.Nt, :].copy()
    box.radii = box.radii[: box.Nt].copy()
    box.groups = box.groups[: box.Nt].copy()

    if cell_list:
        build_cell_list(box)

    return box.success


@nb.njit(fastmath=True)
def bcc(box: NBox, cell_list: bool = True) -> bool:
    r"""
    Generate a body-centered cubic (BCC) particle arrangement.

    Particles are placed on a perfect BCC lattice fitted into a cubic simulation
    box whose side length is derived from the target number density. As many complete
    unit cells as fit along each axis are used; the actual particle count may differ
    slightly from the target.

    Parameters
    ----------
    box : NBox
        Box object. Updated in-place with the particle ensemble.
    cell_list : bool
        Whether to build a cell list after placement.

    Returns
    -------
    bool
        `True` if all particles were successfully placed without overlap, `False`
        otherwise.
    """
    if box.rs.size > 1:
        raise ValueError("Only a single particle group is supported in BCC mode.")

    vf = box.vfs_target[0]
    if vf > math.sqrt(3) * pi / 8:
        raise ValueError("The target volume fraction exceeds the maximum for BCC (0.68).")

    # Number density and box length
    r = box.rs[0]
    nt = vf / (4.0 / 3.0 * pi * r**3)
    a = (2.0 / nt) ** (1.0 / 3.0)
    nc = max(1, int(np.round((box.Nt_target / 2.0) ** (1.0 / 3.0))))
    box.length = nc * a
    Nt = 2 * nc**3

    # Set box metadata
    box.method = METHOD_BCC
    box.periodic = True

    # Allocate particle arrays
    box.radii = np.full(Nt, r, dtype=np.float64)
    box.centers = np.zeros((Nt, 3), dtype=np.float64)
    box.groups = np.zeros(Nt, dtype=np.int64)

    # Place particles on BCC sites
    # Two motif positions per unit cell (in units of 'a'):
    #   corner:      (0,   0,   0  )
    #   body-centre: (0.5, 0.5, 0.5)
    motif = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)

    k = 0
    for ix in range(nc):
        for iy in range(nc):
            for iz in range(nc):
                origin_x = ix * a
                origin_y = iy * a
                origin_z = iz * a
                for m in range(2):
                    box.centers[k, 0] = origin_x + motif[m, 0] * a
                    box.centers[k, 1] = origin_y + motif[m, 1] * a
                    box.centers[k, 2] = origin_z + motif[m, 2] * a
                    k += 1

    # Finalize
    box.success = True  # Bcc always succeeds if inputs are valid
    box.Ns = np.array([Nt], dtype=np.int64)
    box.Nt = Nt

    if cell_list:
        build_cell_list(box)

    return box.success


@nb.njit(fastmath=True)
def equilibrium_distribution(
    box: NBox,
    n_sweeps: int = 200,
    target_accept: float = 0.35,
) -> bool:
    r"""
    Generate an equilibrated ensemble of hard spheres.

    A BCC lattice is first generated, and then an adaptive Monte Carlo procedure is
    used to equilibrate the particle ensemble. The maximum displacement for each particle
    is adjusted to achieve the target acceptance ratio.

    This is method is limited to monodisperse hard spheres, but can generate particle
    configurations with significantly higher volume fractions than RSA.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
    n_sweeps : int
        Number of Monte Carlo sweeps to perform. One sweep consists of one attempted
        move for each particle in the box.
    target_accept : float
        Target acceptance ratio for the adaptive Monte Carlo. The maximum displacement
        for each particle is adjusted to achieve this acceptance ratio.

    Returns
    -------
    bool
        `True` if the equilibration was successful, `False` otherwise.
    """
    # Generate BCC lattice
    bcc(box, cell_list=False)

    # Compute initial gap between nearest neighbors
    r = box.radii[0]
    Nt = box.Nt
    L = box.length
    nc = max(1, int(np.round((Nt / 2.0) ** (1.0 / 3.0))))
    a = L / nc
    gap = math.sqrt(3) / 2 * a - 2.0 * r

    # Adaptive Monte Carlo equilibration
    alpha = 0.2
    block = 10
    delta = 0.2 * gap
    delta_max = 0.5 * gap
    delta_min = 1e-3 * gap
    for _ in range(n_sweeps // block):
        naccept = 0

        for _ in range(block):
            naccept += mc_sweep(box, delta)

        naccept /= block * Nt

        delta *= math.exp(alpha * (naccept - target_accept))
        delta = min(max(delta, delta_min), delta_max)

    box.method = METHOD_EQUIL
    box.success = True
    build_cell_list(box)

    return box.success


@nb.njit(fastmath=True)
def mc_sweep(box: NBox, delta: float) -> int:
    r"""
    Perform one Monte Carlo sweep for hard-sphere particles.

    Parameters
    ----------
    box : NBox
        Box object containing the particle ensemble.
    delta : float
        Maximum displacement for each particle in the sweep.

    Returns
    -------
    int
        Number of accepted moves in the sweep.
    """
    naccepted = 0
    L = box.length
    Nt = box.Nt

    for i in range(Nt):
        # Store old position
        ox = box.centers[i, 0]
        oy = box.centers[i, 1]
        oz = box.centers[i, 2]

        # Propose move
        box.centers[i, 0] = ox + delta * (2.0 * np.random.random() - 1.0)
        box.centers[i, 1] = oy + delta * (2.0 * np.random.random() - 1.0)
        box.centers[i, 2] = oz + delta * (2.0 * np.random.random() - 1.0)

        # Check overlap
        overlap = False
        for j in range(Nt):
            if j == i:
                continue

            if particles_overlap(box, i, j):
                overlap = True
                break

        if overlap:
            # reject → restore
            box.centers[i, 0] = ox
            box.centers[i, 1] = oy
            box.centers[i, 2] = oz

        else:
            # accept → enforce PBC
            if box.periodic:
                box.centers[i, 0] = wrap(box.centers[i, 0], L)
                box.centers[i, 1] = wrap(box.centers[i, 1], L)
                box.centers[i, 2] = wrap(box.centers[i, 2], L)

            naccepted += 1

    return naccepted


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
    hmin = 2.0 * box.rmax
    nc = max(1, int(np.floor(box.length / hmin)))
    ncells_total = nc**3
    h = box.length / nc

    head = -np.ones(ncells_total, dtype=np.int64)
    next = -np.ones(box.Nt, dtype=np.int64)

    for i in range(box.Nt):
        x = box.centers[i, 0]
        y = box.centers[i, 1]
        z = box.centers[i, 2]

        _, _, _, c = cell_index(x, y, z, h, nc, box.periodic)

        next[i] = head[c]
        head[c] = i

    box.cell_size = h
    box.nc = nc
    box.head = head
    box.next = next


@nb.njit(fastmath=True)
def simulate_walk(
    box: NBox, D: float, rtol: float = 1e-3, maxsteps: int = 100
) -> WalkResult:
    r"""
    Simulate a random walk of a point particle in a box containing an ensemble of
    non-overlapping spheres.

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
        raise ValueError(
            "Box has no particles. Run `rsa()` or `bcc()` to fill the box first."
        )

    # Find a free initial position for the radical
    X = np.empty(3, dtype=np.float64)
    L = box.length
    while True:
        X[0] = np.random.uniform(0.0, L)
        X[1] = np.random.uniform(0.0, L)
        X[2] = np.random.uniform(0.0, L)

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
        R = clearance_radius(box, X)
        if R <= box.rmin * rtol / 10:
            break  # If the clearance radius is too small, terminate the walk

        # Move to random point on the sphere of radius R centered about X
        if box.periodic:
            Xtry[:] = random_point_sphere(X, R)
            X[0] = Xtry[0] % L
            X[1] = Xtry[1] % L
            X[2] = Xtry[2] % L
        else:
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


@nb.njit(inline="always")
def apply_mic(dx: float, dy: float, dz: float, L: float) -> tuple[float, float, float]:
    """Apply the Minimum Image Convention (MIC) to a displacement vector.

    Parameters
    ----------
    dx, dy, dz : float
        Displacement vector components.
    L : float
        Box length.

    Returns
    -------
    tuple[float, float, float]
        Displacement vector components after applying the MIC.

    """
    half_L = 0.5 * L

    if dx > half_L:
        dx -= L
    elif dx < -half_L:
        dx += L

    if dy > half_L:
        dy -= L
    elif dy < -half_L:
        dy += L

    if dz > half_L:
        dz -= L
    elif dz < -half_L:
        dz += L

    return dx, dy, dz


@nb.njit(inline="always")
def wrap(x: float, L: float) -> float:
    """Wrap a coordinate into the box using periodic boundary conditions.

    Parameters
    ----------
    x : float
        Coordinate to wrap.
    L : float
        Box length.

    Returns
    -------
    float
        Wrapped coordinate.
    """
    if x < 0.0:
        x += L
    elif x >= L:
        x -= L

    return x


@nb.njit(inline="always")
def cell_index(
    x: float, y: float, z: float, h: float, nc: int, periodic: bool
) -> tuple[int, int, int, int]:
    """
    Compute linear cell index supporting both PBC and non-PBC boxes.

    Parameters
    ----------
    x, y, z : float
        Coordinates of the point.
    h : float
        Cell size.
    nc : int
        Number of cells along each dimension.
    periodic : bool
        Whether to apply periodic boundary conditions.

    Returns
    -------
    tuple[int, int, int, int]
        Cell indices (ix, iy, iz) and linear cell index c.
    """
    ix = int(np.floor(x / h))
    iy = int(np.floor(y / h))
    iz = int(np.floor(z / h))

    if periodic:
        ix = ix % nc
        iy = iy % nc
        iz = iz % nc
    else:
        # Hard clip to box boundaries to catch edge-case precision errors
        if ix < 0:
            ix = 0
        elif ix >= nc:
            ix = nc - 1

        if iy < 0:
            iy = 0
        elif iy >= nc:
            iy = nc - 1

        if iz < 0:
            iz = 0
        elif iz >= nc:
            iz = nc - 1

    c = ix + iy * nc + iz * nc**2
    return ix, iy, iz, c


@nb.njit(inline="always")
def neighbor_cell_1d(i: int, di: int, nc: int, periodic: bool) -> int:
    """Compute neighbor cell index along one dimension, or -1 if out of bounds."""
    ni = i + di
    if periodic:
        return ni % nc
    else:
        if ni < 0 or ni >= nc:
            return -1
        return ni


@nb.njit(inline="always")
def particles_overlap(box: NBox, i: int, j: int) -> bool:
    r"""
    Check if two particles overlap, accounting for periodic boundary conditions
    if enabled in the box configuration.

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

    if box.periodic:
        dx, dy, dz = apply_mic(dx, dy, dz, box.length)

    return (dx**2 + dy**2 + dz**2) <= (box.radii[i] + box.radii[j]) ** 2


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
        0.0 <= point[0] <= box.length
        and 0.0 <= point[1] <= box.length
        and 0.0 <= point[2] <= box.length
    )


@nb.njit(inline="always")
def point_inside_particle(box: NBox, i: int, point: NDArray, rtol: float = 0.0) -> bool:
    r"""
    Check if a point is inside a particle, accounting for periodic boundary conditions
    if enabled in the box configuration.

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

    if box.periodic:
        dx, dy, dz = apply_mic(dx, dy, dz, box.length)

    return (dx**2 + dy**2 + dz**2) <= ((1.0 + rtol) * box.radii[i]) ** 2


@nb.njit(fastmath=True)
def _point_inside_any_particle(
    box: NBox, point: NDArray, rtol: float = 0.0
) -> tuple[bool, int]:
    r"""
    Check if a point is inside any particle, accounting for periodic boundary conditions
    if enabled in the box configuration.

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
    periodic = box.periodic

    ix, iy, iz, _ = cell_index(point[0], point[1], point[2], h, nc, periodic)

    for dx in (-1, 0, 1):
        nx = neighbor_cell_1d(ix, dx, nc, periodic)
        if nx == -1:
            continue

        for dy in (-1, 0, 1):
            ny = neighbor_cell_1d(iy, dy, nc, periodic)
            if ny == -1:
                continue

            for dz in (-1, 0, 1):
                nz = neighbor_cell_1d(iz, dz, nc, periodic)
                if nz == -1:
                    continue

                c = nx + nc * (ny + nc * nz)
                p = box.head[c]

                while p != -1:
                    if point_inside_particle(box, p, point, rtol):
                        return True, p
                    p = box.next[p]

    return False, -1


@nb.njit(fastmath=True)
def _clearance_radius(box: NBox, point: NDArray) -> float:
    r"""
    Compute the radius of the largest sphere centered at `point` that does not
    overlap any particle. Accounts for periodic boundary conditions if enabled in the box
    configuration.

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
    L = box.length
    R = L

    for i in range(box.Nt):
        dx = point[0] - box.centers[i, 0]
        dy = point[1] - box.centers[i, 1]
        dz = point[2] - box.centers[i, 2]

        if box.periodic:
            dx, dy, dz = apply_mic(dx, dy, dz, L)

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
    L = box.length
    periodic = box.periodic

    ix, iy, iz, _ = cell_index(point[0], point[1], point[2], h, nc, periodic)

    R = L
    if periodic:
        max_shells = nc // 2 + 1
    else:
        max_shells = nc

    for shell in range(max_shells):
        # Loop over the 3D grid neighborhood
        for dx in range(-shell, shell + 1):
            nx = neighbor_cell_1d(ix, dx, nc, periodic)
            if nx == -1:
                continue

            for dy in range(-shell, shell + 1):
                ny = neighbor_cell_1d(iy, dy, nc, periodic)
                if ny == -1:
                    continue

                for dz in range(-shell, shell + 1):
                    nz = neighbor_cell_1d(iz, dz, nc, periodic)
                    if nz == -1:
                        continue

                    # Only visit cells on the outer skin of the shell
                    if shell > 0:
                        if (
                            dx > -shell
                            and dx < shell
                            and dy > -shell
                            and dy < shell
                            and dz > -shell
                            and dz < shell
                        ):
                            continue

                    # Linearized cell index matching build_cell_list layout
                    c = nx + nc * (ny + nc * nz)
                    p = box.head[c]

                    # Traversal of the cell's linked list
                    while p != -1:
                        cx = point[0] - box.centers[p, 0]
                        cy = point[1] - box.centers[p, 1]
                        cz = point[2] - box.centers[p, 2]

                        # Find true shortest path across boundaries
                        if periodic:
                            cx, cy, cz = apply_mic(cx, cy, cz, L)

                        s = math.sqrt(cx * cx + cy * cy + cz * cz) - box.radii[p]
                        if s < R:
                            R = s

                        p = box.next[p]

        # Early Exit Criterion
        if shell * h - box.rmax > R:
            break

    return R


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


# @nb.njit
def simulate_multiple(
    rs: NDArray,
    vfs: NDArray,
    number_boxes: int,
    number_particles_per_box: int,
    number_walks_per_box: int,
    D: float = 1.0,
    periodic: bool = True,
    method: Literal["RSA", "BCC", "Equilibrium"] = "RSA",
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
    periodic : bool
        Whether to apply periodic boundary conditions.

    Returns
    -------
    list[Walk]
        List of random walks simulated across all boxes.
    """
    walks = []
    for i in range(number_boxes):
        box = NBox(rs, vfs, number_particles_per_box)

        if method == "RSA":
            # RSA can fail at high volume fractions, so we try multiple times
            success = False
            for _ in range(10):
                success = rsa(box, periodic=periodic, cell_list=False)
                if success:
                    build_cell_list(box)
                    break
            if not success:
                print("  Failed to fill box after 10 attempts. Skipping.")
                continue
        elif method == "BCC":
            bcc(box, cell_list=True)
        elif method == "Equilibrium":
            equilibrium_distribution(box)
        else:
            raise ValueError(f"Unknown method: {method}")

        for _ in range(number_walks_per_box):
            walk = simulate_walk(box, D=D)
            walks.append(walk)

    return walks
