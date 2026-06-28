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

    xlim = (0.0, box.length)
    ax.set_xlim(*xlim)
    ax.set_ylim(*xlim)
    ax.set_zlim(*xlim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=35)

    fig.tight_layout()

    return fig


def plot_with_pyvista(
    box,
    walk=None,
    resolution: int = 18,
    alpha: float = 0.5,
) -> pv.Plotter:
    """
    Plot a 3D visualization of a particle ensemble and random walk using `PyVista`.
    Includes support for PBC via ghost particles, mesh clipping, trajectory unwrapping,
    and distinct visual styling for ghost particles.
    """
    plotter = pv.Plotter()
    Lbox = box.length

    centers = box.centers
    radii = box.radii
    groups = box.groups
    periodic = box.periodic

    # 2. Handle PBC: Generate Ghost Particles & Track Them
    curr_centers, curr_radii, curr_groups = centers, radii, groups
    is_ghost = np.zeros(len(centers), dtype=bool)

    if periodic:
        for dim in range(3):
            new_c_list, new_r_list, new_g_list = [], [], []

            mask_low = (curr_centers[:, dim] - curr_radii) < 0
            if np.any(mask_low):
                c_low = curr_centers[mask_low].copy()
                c_low[:, dim] += Lbox
                new_c_list.append(c_low)
                new_r_list.append(curr_radii[mask_low])
                new_g_list.append(curr_groups[mask_low])

            mask_high = (curr_centers[:, dim] + curr_radii) > Lbox
            if np.any(mask_high):
                c_high = curr_centers[mask_high].copy()
                c_high[:, dim] -= Lbox
                new_c_list.append(c_high)
                new_r_list.append(curr_radii[mask_high])
                new_g_list.append(curr_groups[mask_high])

            if new_c_list:
                new_centers = np.vstack(new_c_list)
                new_radii = np.concatenate(new_r_list)
                new_groups = np.concatenate(new_g_list)
                new_ghosts = np.ones(len(new_centers), dtype=bool)

                # Stack new ghosts into the working arrays
                curr_centers = np.vstack([curr_centers, new_centers])
                curr_radii = np.concatenate([curr_radii, new_radii])
                curr_groups = np.concatenate([curr_groups, new_groups])
                is_ghost = np.concatenate([is_ghost, new_ghosts])

    # Global Group Mapping (Ensures consistent colors across real and ghost meshes)
    unique_groups, all_group_indices = np.unique(curr_groups, return_inverse=True)
    n_groups = len(unique_groups)
    cmap = plt.get_cmap("tab10", n_groups)

    sphere_template = pv.Sphere(
        radius=1.0, theta_resolution=resolution, phi_resolution=resolution
    )
    box_bounds = (0.0, Lbox, 0.0, Lbox, 0.0, Lbox)

    # Process Real Particles
    real_pc = pv.PolyData(curr_centers[~is_ghost])
    real_pc["radius"] = curr_radii[~is_ghost]
    real_pc["group_idx"] = all_group_indices[~is_ghost]

    real_mesh = real_pc.glyph(scale="radius", geom=sphere_template, orient=False)
    if periodic:
        real_mesh = real_mesh.clip_box(bounds=box_bounds, invert=False)
    real_mesh = real_mesh.point_data_to_cell_data()

    plotter.add_mesh(
        real_mesh,
        scalars="group_idx",
        cmap=cmap,
        clim=[-0.5, n_groups - 0.5],
        opacity=alpha,
        show_scalar_bar=False,
        smooth_shading=True,
    )

    # Process Ghost Particles
    has_ghosts = np.any(is_ghost)
    if has_ghosts:
        ghost_pc = pv.PolyData(curr_centers[is_ghost])
        ghost_pc["radius"] = curr_radii[is_ghost]
        ghost_pc["group_idx"] = all_group_indices[is_ghost]

        ghost_mesh = ghost_pc.glyph(scale="radius", geom=sphere_template, orient=False)
        ghost_mesh = ghost_mesh.clip_box(bounds=box_bounds, invert=False)
        ghost_mesh = ghost_mesh.point_data_to_cell_data()

        plotter.add_mesh(
            ghost_mesh,
            scalars="group_idx",
            cmap=cmap,
            clim=[-0.5, n_groups - 0.5],
            opacity=alpha * 0.3,  # Drops opacity heavily for a "ghostly" look
            show_scalar_bar=False,
            smooth_shading=True,
            # style="wireframe"
        )

    # Trajectory with PBC Wrapping
    if walk is not None and walk.trajectory is not None and len(walk.trajectory) > 0:
        trajectory = walk.trajectory
        if periodic:
            diffs = np.abs(np.diff(trajectory, axis=0))
            wrap_indices = np.where(np.any(diffs > Lbox / 2.0, axis=1))[0]
            segments = np.split(trajectory, wrap_indices + 1)

            for i, segment in enumerate(segments):
                if len(segment) > 1:
                    line = pv.MultipleLines(points=segment)
                    label = "trajectory" if i == 0 else None
                    plotter.add_mesh(line, color="black", line_width=4, label=label)
        else:
            line = pv.MultipleLines(points=trajectory)
            plotter.add_mesh(line, color="black", line_width=4, label="trajectory")

        # Highlight the last point of the trajectory with a red sphere
        last_pc = pv.PolyData([trajectory[-1]])
        last_sphere = last_pc.glyph(
            scale=False,
            geom=pv.Sphere(
                radius=box.rmin * 0.01, theta_resolution=18, phi_resolution=18
            ),
            orient=False,
        )

        plotter.add_mesh(
            last_sphere,
            color="red",
            opacity=1.0,
            smooth_shading=True,
        )

    # Environment & Camera
    plotter.add_axes()
    plotter.camera.elevation = 20
    plotter.camera.azimuth = 35
    plotter.set_background("white")

    box_outline = pv.Cube(
        center=(Lbox / 2, Lbox / 2, Lbox / 2),
        bounds=box_bounds,
    )
    plotter.add_mesh(box_outline, style="wireframe", color="gray", line_width=1)

    plotter.reset_camera()

    return plotter
