from textwrap import dedent

import pyvista as pv
from matplotlib.figure import Figure
from numpy.typing import NDArray

import epentry.engine as engine
import epentry.view as view
from epentry.engine import NBox, WalkResult

__all__ = ["Box"]


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
    """

    nbox: NBox

    def __init__(
        self,
        rs: NDArray,
        vfs: NDArray,
        Nt: int,
    ) -> None:
        self.nbox = NBox(rs, vfs, Nt)

    def __repr__(self) -> str:
        """Return a string representation of the Box object."""
        return dedent(f"""\
            Box(
                rs          = {self.nbox.rs},
                vfs_target  = {self.nbox.vfs_target},
                vs          = {self.nbox.vfs()},
                Nt_target   = {self.nbox.Nt_target},
                Nt          = {self.nbox.Nt},
                Ns          = {self.nbox.Ns},
                Lbox        = {self.nbox.Lbox},
                success_rsa = {self.nbox.success_rsa},
            )
        """)

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
        return engine.rsa(self.nbox)

    def plot_with_matplotlib(
        self,
        walk: WalkResult | None = None,
        resolution: int = 18,
        alpha: float = 0.5,
    ) -> Figure:
        """
        Plot a 3D visualization of the particles the random walk using `matplotlib`.

        Each particle is rendered as a sphere and colored according to its
        group index. If present, the trajectory is overlaid as a 3D line.

        Parameters
        ----------
        walk : Walk
            Walk object containing the trajectory to overlay. If `None`, only the
            particle ensemble will be plotted.
        resolution : int
            Resolution of the sphere mesh used to render each particle.
            Higher values produce smoother spheres but increase rendering cost.
        alpha : float
            Transparency of particle surfaces in the plot.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the 3D rendering.
        """
        return view.plot_sim_matplotlib(self.nbox, walk, resolution, alpha)

    def plot_with_pyvista(
        self,
        walk: WalkResult | None = None,
        resolution: int = 18,
        alpha: float = 0.5,
    ) -> pv.Plotter:
        """
        Plot a 3D visualization of the particles the random walk using `PyVista`.

        Parameters
        ----------
        walk : Walk
            Walk object containing the trajectory to overlay. If `None`, only the
            particle ensemble will be plotted.
        resolution : int
            Resolution of the sphere mesh used to render each particle.
            Higher values produce smoother spheres but increase rendering cost.
        alpha : float
            Transparency of particle surfaces in the plot.

        Returns
        -------
        pyvista.Plotter
            Plotter object containing the 3D rendering. Call `show()` on the
            returned object to display the plot.
        """
        return view.plot_sim_pyvista(self.nbox, walk, resolution, alpha)

    def simulate_walk(
        self,
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
        * There are "wall effects" because the random walker is not allowed to step
        outside the box and we do not have periodic boundary conditions.

        Parameters
        ----------
        D : float
            Diffusion coefficient of the random walker. This only affects the time
            associated with the walk and not the trajectory itself.
        rtol : float
            Relative tolerance for numerical precision when checking if the random walker
            has collided with a particle. The walker is considered to have collided if it
            is within `rtol` of the particle surface.
        maxsteps : int
            Maximum number of steps for the random walk. If the walker does not collide
            with a particle within this number of steps, the walk will be terminated and
            considered unsuccessful.

        Returns
        -------
        WalkResult
            Object containing the random walk trajectory, and information about the
            particle hit (if any) and time taken for the walk.
        """
        return engine.simulate_walk(self.nbox, D, rtol, maxsteps)
