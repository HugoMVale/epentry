from textwrap import dedent
from typing import Literal

import numpy as np
import pyvista as pv
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

import epentry.engine as engine
import epentry.view as view
from epentry.engine import NBox, WalkResult

__all__ = ["Box"]


class Box:
    """
    Ensemble of particles in a rectangular box.

    Parameters
    ----------
    rs : ArrayLike
        Radii of each particle group.
    vfs : ArrayLike
        Target volume fractions of each particle group.
    Nt : int
        Target total number of particles in the box.

    Examples
    --------
    >>> from epentry import Box
    >>> box = Box(rs=[1.0, 2.0], vfs=[0.1, 0.2], Nt=100)
    >>> box.place_particles(method="RSA")
    >>> box.plot()
    >>> walk_result = box.simulate_walk(D=1.0)
    >>> box.plot(walk_result)
    """

    _nbox: NBox

    def __init__(
        self,
        rs: ArrayLike,
        vfs: ArrayLike,
        Nt: int,
    ) -> None:
        rs = np.asarray(rs, dtype=np.float64)
        vfs = np.asarray(vfs, dtype=np.float64)

        if rs.ndim != 1 or vfs.ndim != 1:
            raise ValueError("rs and vfs must be 1D arrays.")
        if len(rs) != len(vfs):
            raise ValueError(
                f"Length of rs ({len(rs)}) must match length of vfs ({len(vfs)})."
            )
        if np.any(rs <= 0.0):
            raise ValueError("All particle radii must be positive.")
        if np.any(vfs < 0.0) or np.any(vfs > 1.0):
            raise ValueError("All volume fractions must be in the range [0, 1].")
        if vfs.sum() == 0.0 or vfs.sum() > 1.0:
            raise ValueError(
                f"Sum of volume fractions ({vfs.sum()}) must be in the range (0, 1)."
            )
        if not (isinstance(Nt, int) and Nt > 0):
            raise ValueError("Total number of particles must be a positive integer.")

        self._nbox = NBox(rs, vfs, Nt)

    def __repr__(self) -> str:
        """Return a string representation of the Box object."""
        return dedent(f"""\
            Box(
                rs         = {self._nbox.rs},
                vfs_target = {self._nbox.vfs_target},
                vfs        = {self._nbox.vfs()},
                Nt_target  = {self._nbox.Nt_target},
                Nt         = {self._nbox.Nt},
                Ns         = {self._nbox.Ns},
                length     = {self._nbox.length},
                method     = {engine.METHODS.get(self._nbox.method, "None")},
                periodic   = {self._nbox.periodic},
                success    = {self._nbox.success},
            )
        """)

    def place_particles(
        self,
        method: Literal["RSA", "SC", "BCC", "FCC", "Equilibrium"] = "RSA",
        periodic: bool = True,
    ) -> bool:
        """
        Generate a non-overlapping particle ensemble using a given method.

        RSA: Particles are placed uniformly at random in a cubic simulation box. Candidate
        positions that overlap previously placed particles are rejected until a valid
        position is found. Failure to place all particles is likely at a high total volume
        fraction

        SC: Particles are placed on a simple cubic lattice. The lattice spacing is chosen
        to achieve the target volume fraction. This method is only valid for a single
        particle group.

        BCC: Particles are placed on a body-centered cubic lattice. The lattice spacing is
        chosen to achieve the target volume fraction. This method is only valid for a
        single particle group.

        FCC: Particles are placed on a face-centered cubic lattice. The lattice spacing is
        chosen to achieve the target volume fraction. This method is only valid for a
        single particle group.

        Equilibrium: Particles are placed in a BCC lattice and then allowed to relax to
        an equilibrium configuration using a Monte Carlo simulation. This method is only
        valid for a single particle group.

        Parameters
        ----------
        periodic : bool
            Whether to apply periodic boundary conditions. Only relevant for the RSA
            method.

        Returns
        -------
        bool
            `True` if all particles were successfully placed without overlap, `False`
            otherwise.

        """
        if method == "RSA":
            return engine.rsa(self._nbox, periodic)
        elif method == "SC":
            return engine.sc(self._nbox)
        elif method == "BCC":
            return engine.bcc(self._nbox)
        elif method == "FCC":
            return engine.fcc(self._nbox)
        elif method == "Equilibrium":
            return engine.equilibrium_distribution(self._nbox)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Must be 'RSA', 'SC', 'BCC', 'FCC' or 'Equilibrium'."  # noqa: E501
            )

    def plot(
        self,
        walk: WalkResult | None = None,
        backend: Literal["matplotlib", "pyvista"] = "pyvista",
        resolution: int = 18,
        alpha: float = 0.5,
        elevation: float = 20.0,
        azimuth: float = 35.0,
    ) -> Figure | pv.Plotter:
        """
        Plot a 3D visualization of the particles and the random walk.

        Each particle is rendered as a sphere and colored according to its
        group index. If present, the trajectory is overlaid as a 3D line.

        Parameters
        ----------
        walk : WalkResult | None
            Walk object containing the trajectory to overlay. If `None`, only the
            particle ensemble will be plotted.
        backend : Literal["matplotlib", "pyvista"]
            Backend to use for plotting.
        resolution : int
            Resolution of the sphere mesh used to render each particle.
            Higher values produce smoother spheres but increase rendering cost.
        alpha : float
            Transparency of particle surfaces in the plot.
        elevation : float
            Elevation angle in the z plane for the 3D plot view.
        azimuth : float
            Azimuth angle in the x,y plane for the 3D plot view.

        Returns
        -------
        matplotlib.figure.Figure | pyvista.Plotter
            Figure object containing the 3D rendering.
        """
        if backend == "matplotlib":
            return view.plot_with_matplotlib(
                self._nbox, walk, resolution, alpha, elevation, azimuth
            )
        elif backend == "pyvista":
            return view.plot_with_pyvista(
                self._nbox, walk, resolution, alpha, elevation, azimuth
            )
        else:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be 'matplotlib' or 'pyvista'."
            )

    def simulate_walk(
        self,
        D: float = 1.0,
        rtol: float = 1e-3,
        maxsteps: int = 100,
    ) -> WalkResult:
        """
        Simulate a random walk of a point particle in a box containing an ensemble of
        non-overlapping spheres.

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
        return engine.simulate_walk(self._nbox, D, rtol, maxsteps)
