import itertools

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LightSource, to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from epentry.engine import NBox, WalkResult

__all__ = [
    "plot_with_matplotlib",
    "plot_with_pyvista",
]


def plot_with_matplotlib(
    box: NBox,
    walk: WalkResult | None = None,
    resolution: int = 18,
    alpha: float = 1.0,
    elevation: float = 20.0,
    azimuth: float = 35.0,
) -> Figure:
    """
    Plot a 3D visualization of the particles and the random walk using `matplotlib`.

    Each particle is rendered as a sphere and colored according to its group index. If
    present, the trajectory is overlaid as a 3D line.

    There is no support for periodic boundary conditions in this backend. Ghost particles
    are not generated and the trajectory is not wrapped.

    Parameters
    ----------
    walk : WalkResult | None
        Walk object containing the trajectory to overlay. If `None`, only the
        particle ensemble will be plotted.
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
    matplotlib.figure.Figure
        Figure object containing the 3D rendering.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    base_cmap = plt.get_cmap("tab10")
    unique_groups = np.unique(box.groups)
    group_to_color = {
        group: base_cmap(i % base_cmap.N) for i, group in enumerate(unique_groups)
    }

    # Pre-compute a unit sphere template mesh
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x_template = np.outer(np.cos(u), np.sin(v))
    y_template = np.outer(np.sin(u), np.sin(v))
    z_template = np.outer(np.ones_like(u), np.cos(v))

    # Convert the grid into explicit quad faces
    # For a grid of size (R, R), there are (R-1)*(R-1) quads
    faces_template = []
    for r in range(resolution - 1):
        for c in range(resolution - 1):
            # Define the 4 corners of the quad patch
            p1 = (x_template[r, c], y_template[r, c], z_template[r, c])
            p2 = (x_template[r + 1, c], y_template[r + 1, c], z_template[r + 1, c])
            p3 = (
                x_template[r + 1, c + 1],
                y_template[r + 1, c + 1],
                z_template[r + 1, c + 1],
            )
            p4 = (x_template[r, c + 1], y_template[r, c + 1], z_template[r, c + 1])
            faces_template.append([p1, p2, p3, p4])
    faces_template = np.array(faces_template)  # Shape: (N_faces_per_sphere, 4, 3)

    # Vectorized construction of all sphere faces
    all_polygons = []
    all_facecolors = []

    for i in range(box.Nt):
        c = box.centers[i]
        r = box.radii[i]
        color = group_to_color[box.groups[i]]

        # Scale and shift the unit template faces to this particle's position/radius
        sphere_faces = faces_template * r + c
        all_polygons.append(sphere_faces)

        # Assign the same color to all faces of this sphere
        rgba_color = to_rgba(color, alpha=alpha)
        all_facecolors.extend([rgba_color] * len(faces_template))

    # Flatten the list of all polygons across all spheres
    all_polygons = np.vstack(all_polygons)

    # Define a light source coming from the top-right-front
    light = LightSource(azdeg=315, altdeg=45)

    # Add everything to the axis as a single, ultra-fast collection
    collection = Poly3DCollection(
        all_polygons,
        facecolors=all_facecolors,
        edgecolors=None,
        shade=True,
        lightsource=light,
    )

    # Optional: Enable light shading manually if desired, or let Matplotlib handle it flatly
    # For simple color categorization, unshaded collections look crisp and clean.
    ax.add_collection3d(collection)

    # Trajectory overlay
    if walk is not None and walk.trajectory is not None and len(walk.trajectory) > 0:
        ax.plot(
            walk.trajectory[:, 0],
            walk.trajectory[:, 1],
            walk.trajectory[:, 2],
            color="black",
            linewidth=2.5,
            label="trajectory",
            zorder=10,
        )

    # Box wireframe
    Lbox = box.length
    corners = np.array(
        [
            [0, 0, 0],
            [Lbox, 0, 0],
            [Lbox, Lbox, 0],
            [0, Lbox, 0],
            [0, 0, Lbox],
            [Lbox, 0, Lbox],
            [Lbox, Lbox, Lbox],
            [0, Lbox, Lbox],
        ]
    )
    box_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # verticals
    ]
    for i, j in box_edges:
        ax.plot(
            *zip(corners[i], corners[j]),
            color="gray",
            linewidth=1,
            zorder=1,
        )

    # Limits and view setup
    xlim = (0.0, box.length)
    ax.set_xlim(*xlim)
    ax.set_ylim(*xlim)
    ax.set_zlim(*xlim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elevation, azim=azimuth)
    fig.tight_layout()

    return fig


def plot_with_pyvista(
    box: NBox,
    walk: WalkResult | None = None,
    resolution: int = 18,
    alpha: float = 1.0,
    elevation: float = 20.0,
    azimuth: float = 35.0,
    clip: bool = False,
) -> pv.Plotter:
    """
    Plot a 3D visualization of the particles and the random walk using `pyvista`.

    Each particle is rendered as a sphere and colored according to its group index. If
    present, the trajectory is overlaid as a 3D line.

    Parameters
    ----------
    box: NBox
        The particle ensemble to visualize.
    walk : WalkResult | None
        Walk object containing the trajectory to overlay. If `None`, only the
        particle ensemble will be plotted.
    resolution : int
        Resolution of the sphere mesh used to render each particle.
        Higher values produce smoother spheres but increase rendering cost.
    alpha : float
        Transparency of particle surfaces in the plot.
    elevation : float
        Elevation angle in the z plane for the 3D plot view.
    azimuth : float
        Azimuth angle in the x,y plane for the 3D plot view.
    clip : bool
        Whether to clip particles to the box boundaries.

    Returns
    -------
    pyvista.Plotter
        Figure object containing the 3D rendering.
    """
    plotter = pv.Plotter()
    Lbox = box.length

    centers = box.centers
    radii = box.radii
    groups = box.groups
    periodic = box.periodic

    # Handle PBC: Generate Ghost Particles & Track Them
    curr_centers, curr_radii, curr_groups = centers, radii, groups
    is_ghost = np.zeros(len(centers), dtype=bool)

    if periodic:
        offsets = np.array(
            [off for off in itertools.product((-1, 0, 1), repeat=3) if off != (0, 0, 0)]
        )  # shape (26, 3)

        ghost_c_list, ghost_r_list, ghost_g_list = [], [], []
        for offset in offsets:
            shifted = centers + offset * Lbox
            clamped = np.clip(shifted, 0.0, Lbox)
            dist = np.linalg.norm(shifted - clamped, axis=1)

            mask = dist < radii
            if np.any(mask):
                ghost_c_list.append(shifted[mask])
                ghost_r_list.append(radii[mask])
                ghost_g_list.append(groups[mask])

        if ghost_c_list:
            new_centers = np.vstack(ghost_c_list)
            new_radii = np.concatenate(ghost_r_list)
            new_groups = np.concatenate(ghost_g_list)

            curr_centers = np.vstack([centers, new_centers])
            curr_radii = np.concatenate([radii, new_radii])
            curr_groups = np.concatenate([groups, new_groups])
            is_ghost = np.concatenate(
                [
                    np.zeros(len(centers), dtype=bool),
                    np.ones(len(new_centers), dtype=bool),
                ]
            )

    # Global Group Mapping (Ensures consistent colors across real and ghost meshes)
    unique_groups, all_group_indices = np.unique(curr_groups, return_inverse=True)
    n_groups = len(unique_groups)
    if n_groups <= 10:
        cmap = plt.get_cmap("tab10", n_groups)
    elif n_groups <= 20:
        cmap = plt.get_cmap("tab20", n_groups)
    else:
        cmap = plt.get_cmap("hsv", n_groups)

    sphere_template = pv.Sphere(
        radius=1.0, theta_resolution=resolution, phi_resolution=resolution
    )
    box_bounds = (0.0, Lbox, 0.0, Lbox, 0.0, Lbox)

    # Process Real and Ghost Particles
    for ghost_flag, opacity_mult in [(False, 1.0), (True, 0.3)]:
        mask = is_ghost if ghost_flag else ~is_ghost
        if ghost_flag and not np.any(mask):
            continue

        if periodic and clip:
            mesh = _clip_spheres_to_box(
                curr_centers[mask],
                curr_radii[mask],
                all_group_indices[mask],
                sphere_template,
                box_bounds,
            )
            if mesh.n_points == 0:
                continue
        else:
            pc = pv.PolyData(curr_centers[mask])
            pc["radius"] = curr_radii[mask]
            pc["group_idx"] = all_group_indices[mask]

            mesh = pc.glyph(scale="radius", geom=sphere_template, orient=False)
            mesh = mesh.point_data_to_cell_data()

        plotter.add_mesh(
            mesh,
            scalars="group_idx",
            cmap=cmap,
            clim=[-0.5, n_groups - 0.5],
            opacity=alpha * opacity_mult,
            show_scalar_bar=False,
            smooth_shading=True,
        )

    # Trajectory with PBC Wrapping
    if walk is not None and walk.trajectory is not None and len(walk.trajectory) > 0:
        trajectory = walk.trajectory

        if periodic:
            # 1. Split into continuous pieces where jumps occur
            diffs = np.diff(trajectory, axis=0)
            wrap_indices = np.where(np.any(np.abs(diffs) > Lbox / 2.0, axis=1))[0]
            segments = np.split(trajectory, wrap_indices + 1)

            # 2. Combine segments using PyVista's native multi-dataset tools
            # This completely avoids complex raw VTK index math
            lines_to_add = []
            dots_to_add = []

            for segment in segments:
                if len(segment) > 1:
                    lines_to_add.append(pv.MultipleLines(points=segment))
                elif len(segment) == 1:
                    dots_to_add.append(pv.PolyData(segment))

            # Merge lines into a single compound dataset for rapid rendering
            if lines_to_add:
                combined_lines = (
                    pv.merge(lines_to_add) if len(lines_to_add) > 1 else lines_to_add[0]
                )
                plotter.add_mesh(
                    combined_lines, color="black", line_width=4, label="trajectory"
                )

            if dots_to_add:
                combined_dots = (
                    pv.merge(dots_to_add) if len(dots_to_add) > 1 else dots_to_add[0]
                )
                plotter.add_mesh(
                    combined_dots,
                    color="black",
                    point_size=8,
                    render_points_as_spheres=True,
                )
        else:
            line = pv.MultipleLines(points=trajectory)
            plotter.add_mesh(line, color="black", line_width=4, label="trajectory")

        # Highlight the last point of the trajectory with a red sphere
        last_sphere = pv.Sphere(
            radius=Lbox * 5e-3,
            center=trajectory[-1],
            theta_resolution=18,
            phi_resolution=18,
        )
        plotter.add_mesh(last_sphere, color="red", opacity=1.0, smooth_shading=True)

    # Environment & Camera
    plotter.add_axes()
    plotter.camera.elevation = elevation
    plotter.camera.azimuth = azimuth
    plotter.set_background("white")

    box_outline = pv.Cube(
        center=(Lbox / 2, Lbox / 2, Lbox / 2),
        bounds=box_bounds,
    )
    plotter.add_mesh(box_outline, style="wireframe", color="gray", line_width=1)

    plotter.reset_camera()

    return plotter


def _plane_disc_cap(
    center: np.ndarray,
    radius: float,
    axis: int,
    coord: float,
    keep_ge: bool,
    other_active: list[tuple[int, float, bool]],
    n_seg: int = 48,
) -> pv.PolyData | None:
    """Build the exact flat disc cap where a sphere intersects a plane.

    A sphere cut by a plane is always a circle, so the cap can be built analytically
    instead of relying on VTK's hole-filling heuristics (which behave inconsistently
    across axes/particles). If the sphere also crosses other planes at the same time
    (a corner particle), the disc is trimmed against those planes too via
    Sutherland-Hodgman polygon clipping, since the visible cap on one face can be a
    lens/wedge rather than a full circle.
    """
    dist = center[axis] - coord
    r2 = radius**2 - dist**2
    if r2 <= 0:
        return None
    rp = np.sqrt(r2)
    c_disc = center.copy()
    c_disc[axis] = coord

    other_axes = [a for a in (0, 1, 2) if a != axis]
    u_axis, v_axis = other_axes
    u = np.zeros(3)
    u[u_axis] = 1.0
    v = np.zeros(3)
    v[v_axis] = 1.0

    t = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    poly = [c_disc + rp * np.cos(ti) * u + rp * np.sin(ti) * v for ti in t]

    for oaxis, ocoord, okeep_ge in other_active:
        clipped = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            d1 = (p1[oaxis] - ocoord) if okeep_ge else (ocoord - p1[oaxis])
            d2 = (p2[oaxis] - ocoord) if okeep_ge else (ocoord - p2[oaxis])
            in1, in2 = d1 >= 0, d2 >= 0
            if in1:
                clipped.append(p1)
            if in1 != in2:
                tt = d1 / (d1 - d2)
                clipped.append(p1 + tt * (p2 - p1))
        poly = clipped
        if len(poly) < 3:
            return None

    poly = np.array(poly)
    centroid = poly.mean(axis=0)
    pts = np.vstack([centroid, poly])
    n = len(poly)
    faces = np.hstack([[3, 0, i + 1, (i + 1) % n + 1] for i in range(n)])

    mesh = pv.PolyData(pts, faces)
    normal = np.zeros(3)
    normal[axis] = -1.0 if keep_ge else 1.0
    mesh.point_data["Normals"] = np.tile(normal, (mesh.n_points, 1))
    return mesh


def _clip_spheres_to_box(
    centers: np.ndarray,
    radii: np.ndarray,
    group_indices: np.ndarray,
    sphere_template: pv.PolyData,
    bounds: tuple[float, float, float, float, float, float],
) -> pv.PolyData:
    """Clip each glyphed sphere to an axis-aligned box, capping the cut with exact
    analytic discs, and reattach the (constant per-sphere) group scalar since
    clipping/capping does not propagate point/cell data.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    pieces = []
    for center, radius, gidx in zip(centers, radii, group_indices):
        active = []
        if center[0] - radius < xmin:
            active.append((0, xmin, True))
        if center[0] + radius > xmax:
            active.append((0, xmax, False))
        if center[1] - radius < ymin:
            active.append((1, ymin, True))
        if center[1] + radius > ymax:
            active.append((1, ymax, False))
        if center[2] - radius < zmin:
            active.append((2, zmin, True))
        if center[2] + radius > zmax:
            active.append((2, zmax, False))

        if not active:
            sphere = sphere_template.copy()
            sphere.points = sphere.points * radius + center
            sphere = sphere.compute_normals(
                auto_orient_normals=True, consistent_normals=True
            )
            sphere.cell_data["group_idx"] = np.full(sphere.n_cells, gidx)
            pieces.append(sphere)
            continue

        sphere = sphere_template.copy()
        sphere.points = sphere.points * radius + center
        shell = sphere.clip_box(bounds, invert=False).extract_surface(
            algorithm="dataset_surface"
        )
        if shell.n_points == 0:
            continue
        shell = shell.compute_normals(auto_orient_normals=True, consistent_normals=True)
        shell.cell_data["group_idx"] = np.full(shell.n_cells, gidx)
        pieces.append(shell)

        for axis, coord, keep_ge in active:
            others = [a for a in active if a[0] != axis]
            cap = _plane_disc_cap(center, radius, axis, coord, keep_ge, others)
            if cap is None:
                continue
            cap.cell_data["group_idx"] = np.full(cap.n_cells, gidx)
            pieces.append(cap)

    return pv.merge(pieces) if pieces else pv.PolyData()
