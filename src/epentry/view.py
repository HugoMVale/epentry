from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.figure import Figure

from epentry.engine import NBox, WalkResult

__all__ = ["plot_with_matplotlib", "plot_with_pyvista"]


def plot_with_matplotlib(
    box: NBox,
    walk: WalkResult | None = None,
    resolution: int = 18,
    alpha: float = 0.5,
) -> Figure:
    """
    Plot a 3D visualization of a particle ensemble and random walk using `matplotlib`.

    Each particle is rendered as a sphere and colored according to its
    group index. If present, the trajectory is overlaid as a 3D line.

    Parameters
    ----------
    box : Box
        Box object.
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
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    base_cmap = plt.get_cmap("tab10")
    group_to_color = {
        group: base_cmap(i % base_cmap.N) for i, group in enumerate(np.unique(box.groups))
    }

    u = np.linspace(0, 2 * pi, resolution)
    v = np.linspace(0, pi, resolution)

    for i in range(box.Nt):
        c = box.centers[i]
        r = box.radii[i]
        color = group_to_color[box.groups[i]]

        x = c[0] + r * np.outer(np.cos(u), np.sin(v))
        y = c[1] + r * np.outer(np.sin(u), np.sin(v))
        z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            edgecolor="none",
            alpha=alpha,
            shade=True,
        )

    if walk is not None:
        ax.plot(
            walk.trajectory[:, 0],
            walk.trajectory[:, 1],
            walk.trajectory[:, 2],
            color="black",
            linewidth=2.5,
            label="trajectory",
            zorder=10,
        )

    xlim = (0.0, box.Lbox)
    ax.set_xlim(*xlim)
    ax.set_ylim(*xlim)
    ax.set_zlim(*xlim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=35)

    fig.tight_layout()

    return fig


def plot_with_pyvista(
    box: NBox,
    walk: WalkResult | None = None,
    resolution: int = 18,
    alpha: float = 0.5,
) -> pv.Plotter:
    """
    Plot a 3D visualization of a particle ensemble and random walk using `PyVista`.

    Parameters
    ----------
    box : Box
        Box object.
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
    plotter = pv.Plotter()

    point_cloud = pv.PolyData(box.centers)
    point_cloud["radius"] = box.radii

    unique_groups, group_indices = np.unique(box.groups, return_inverse=True)
    n_groups = len(unique_groups)
    point_cloud["group_idx"] = group_indices

    sphere_template = pv.Sphere(
        radius=1.0, theta_resolution=resolution, phi_resolution=resolution
    )

    spheres_mesh = point_cloud.glyph(scale="radius", geom=sphere_template, orient=False)

    spheres_mesh = spheres_mesh.point_data_to_cell_data()

    plotter.add_mesh(
        spheres_mesh,
        scalars="group_idx",
        cmap=plt.get_cmap("tab10", n_groups),
        clim=[-0.5, n_groups - 0.5],
        opacity=alpha,
        show_scalar_bar=False,
        smooth_shading=True,
    )

    if walk is not None:
        line = pv.MultipleLines(points=walk.trajectory)
        plotter.add_mesh(line, color="black", line_width=4, label="trajectory")

    plotter.camera.elevation = 20
    plotter.camera.azimuth = 35
    plotter.set_background("white")

    Lbox = box.Lbox
    box_outline = pv.Cube(
        center=(Lbox / 2, Lbox / 2, Lbox / 2),
        bounds=(0.0, Lbox, 0.0, Lbox, 0.0, Lbox),
    )
    plotter.add_mesh(box_outline, style="wireframe", color="gray", line_width=1)

    return plotter
