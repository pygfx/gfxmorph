"""
Example morphing.
"""

import os
import json

import trimesh
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la
from gfxmorph.maybe_pygfx import smooth_sphere_geometry, DynamicMeshGeometry
from gfxmorph import DynamicMesh, MeshUndoTracker
from gfxmorph.meshfuncs import vertex_get_neighbours
from gfxmorph.utils import logger


INSTRUCTIONS = """
* Morph by pulling on the mesh slices.
* Morph in the direction of the normal by clicking in 3D view and pulling up/down.
* Hold shift to pan/zoom/rotate.
* Click with control/command in the 3D view to select that point in the slice views.
* Scroll with the radius-sphere visisible to change its size.
"""

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

MESH_COLOR = 0, 0.7, 0.8, 1
MESH_COLOR_MORPH = 0.7, 0.7, 0.2, 1


# %% Morph logic


class DynamicMeshGeometryForMorphing(DynamicMeshGeometry):
    # This is a subclass of both gfx.Geometry and MeshChangeTracker.
    # This illustrates how we can relatively easily associate additional
    # buffers with the mesh.

    def new_vertices_buffer(self, mesh):
        super().new_vertices_buffer(mesh)
        # self.sizes = gfx.Buffer(np.zeros((self.positions.nitems,), np.float32))
        self.colors = gfx.Buffer(np.zeros((self.positions.nitems, 4), np.float32))
        self.colors.data[:] = MESH_COLOR


def softlimit(i, limit):
    """Make i be withing <-limit, +limit>, but using soft slopes."""
    f = np.exp(-np.abs(i) / limit)
    return -limit * (f - 1) * np.sign(i)


def gaussian_weights(f):
    return np.exp(-0.5 * f * f)  # Gaussian kernel (sigma 1, mu 0)


class Morpher:
    """Container class for everything related to the mesh and morphing it."""

    def __init__(self):
        # The object that stores & manages the data
        self.m = DynamicMesh(None, None)

        # Tracker that implement a redo stack
        self.undo_tracker = MeshUndoTracker()
        self.m.track_changes(self.undo_tracker)

        # Tracker that maps the mesh to a gfx Geometry (with live updates)
        self.geometry = DynamicMeshGeometryForMorphing()
        self.m.track_changes(self.geometry)

        self.state = None
        self.ref_edge_length = 0.1
        self.radius = 1

        self._create_world_objects()

    def calibrate_scale(self):
        edge_positions = self.m.positions[self.m.edges]
        edge_lengths = np.linalg.norm(
            edge_positions[:, 0, :] - edge_positions[:, 1, :], axis=1
        )
        mean_edge_length = edge_lengths.mean()

        object_size = np.linalg.norm(
            self.m.positions.max(axis=0) - self.m.positions.min(axis=0)
        )

        self.ref_edge_length = mean_edge_length * 0.9
        self.radius = object_size / 15

    def _create_world_objects(self):
        # The front, to show the mesh itself.
        self.wob_front = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color_mode="vertex", color="#6ff", flat_shading=False, side="FRONT"
            ),
        )

        # The back, to show holes and protrusions.
        self.wob_back = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#900", flat_shading=False, side="BACK"
            ),
        )

        # The wireframe, to show edges.
        self.wob_wire = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#777",
                opacity=0.1,
                wireframe=True,
                wireframe_thickness=2,
                side="FRONT",
            ),
        )

        # Helper to show points being grabbed
        self.wob_points = gfx.Points(
            self.geometry,
            gfx.PointsMaterial(color="yellow", vertex_sizes=True),
        )

        # A gizmo to show the direction of movement.
        self.wob_gizmo = gfx.Mesh(
            gfx.cylinder_geometry(0.05, 0.05, 0.5, radial_segments=16),
            gfx.MeshPhongMaterial(color="yellow"),
        )
        self.wob_gizmo.visible = False

        # Radius helper
        self.wob_radius = gfx.Mesh(
            smooth_sphere_geometry(subdivisions=2),
            gfx.MeshPhongMaterial(color="yellow", opacity=0.2, side="front"),
        )
        self.wob_radius.visible = False

        self.world_objects = gfx.Group()
        self.world_objects.add(
            self.wob_front,
            self.wob_back,
            self.wob_wire,
            # self.wob_points,
            self.wob_gizmo,
            self.wob_radius,
        )

    def cancel(self):
        self.undo_tracker.cancel(self.m)

    def commit(self):
        self.undo_tracker.commit()

    def undo(self):
        self.undo_tracker.undo(self.m)

    def redo(self):
        self.undo_tracker.redo(self.m)

    def highlight(self, highlight):
        if highlight:
            # self.wob_radius.visible = True
            self.wob_wire.material.opacity = 0.2
        else:
            self.wob_radius.visible = False
            self.wob_wire.material.opacity = 0.1

    def show_morph_grab(self, fi, coord):
        # Get pos
        coordvec = np.array(coord).reshape(3, 1)
        vii = self.m.faces[fi]
        assert np.sum(coordvec) > 0, f"unexpected pick coord: {coord}"
        pos = (self.m.positions[vii] * coordvec).sum(axis=0) / np.sum(coordvec)
        # Adjust world objects
        self.wob_radius.local.position = pos
        self.wob_radius.local.scale = self.radius
        self.wob_radius.visible = True

    def start_morph_from_vertex(self, xy, vi):
        """Initiate a drag, based on a vertex index.

        This method may feel less precise than using ``start_morph_from_face``,
        but is included for cases where picking capabilities are limited.
        """
        assert isinstance(vi, int)

        vii = [vi]
        distances = [0]
        pos = self.m.positions[vi].copy()
        normal = self.m.normals[vi].copy()

        return self._start_morph(xy, vii, distances, pos, normal)

    def start_morph_from_face(self, xy, fi, coord, in_dir_of_normal=False):
        """Initiate a drag, based on a face index and face coordinates.

        This allows precise grabbing of the mesh. The face coordinates
        are barycentric coordinates; for each vertex it indicates how
        close the grab-point is to that vertex, with 1 being on the
        vertex and 0 being somewhere on the edge between the other two
        vertices. A value of (0.5, 0.5, 0.5) represents the center of
        the face.
        """
        assert isinstance(fi, int)
        assert isinstance(coord, (tuple, list)) and len(coord) == 3

        vii = self.m.faces[fi]
        coord_vec = np.array(coord).reshape(3, 1)
        pos = (self.m.positions[vii] * coord_vec).sum(axis=0) / np.sum(coord)

        if in_dir_of_normal:
            normal = (self.m.normals[vii] * coord_vec).sum(axis=0)
            normal = normal / np.linalg.norm(normal)
        else:
            normal = None

        distances = np.linalg.norm(self.m.positions[vii] - pos, axis=1)
        return self._start_morph(xy, vii, distances, pos, normal)

    def _start_morph(self, xy, vii, ref_distances, pos, normal):
        # Select vertices
        self._select_vertices(vii, ref_distances)
        if not self.state:
            return

        # Add more state
        more_state = {
            "action": "morph",
            "pos": pos,
            "normal": normal,
            "xy": xy,
        }
        self.state.update(more_state)

        # Bring gizmo into place
        if normal is not None:
            self.wob_gizmo.visible = True
            self.wob_gizmo.local.scale = self.radius * 2
            self.wob_gizmo.world.position = pos
            self.wob_gizmo.local.rotation = la.quat_from_vecs((0, 0, 1), normal)

    def _select_vertices(self, vii, ref_distances):
        # Cancel any pending changes to the mesh. If we were already dragging,
        # that operation is cancelled. If other code made uncomitted changes,
        # these are discarted too (code should have comitted).
        self.cancel()
        self.finish()

        # Select vertices.
        search_distance = self.radius * 3  # 3 x std
        method = "auto"  # method = "edge" if self.radius > 1.5 else "smooth2"
        indices, geodesic_distances = self.m.select_vertices_over_surface(
            vii, ref_distances, search_distance, method
        )
        positions = self.m.positions[indices]

        # Pre-calculate deformation weights
        weights = gaussian_weights(geodesic_distances / self.radius).reshape(-1, 1)

        # If for some (future) reason, the selection is empty, cancel
        if len(indices) == 0:
            return

        # Update data for points that highlight the selection
        first, last = indices.min(), indices.max()
        self.geometry.colors.data[indices] = MESH_COLOR_MORPH
        self.geometry.colors.update_range(first, last - first + 1)
        # self.geometry.sizes.data[indices] = 7  # 2 + 20*weights.flatten()
        # self.geometry.sizes.update_range(first, last - first + 1)

        # Store state
        self.state = {
            "action": "",
            "indices": indices,
            "positions": positions,
            "weights": weights,
        }

    def start_smooth(self, xy, fi, coord):
        """Start a smooth action."""

        assert isinstance(fi, int)
        assert isinstance(coord, (tuple, list)) and len(coord) == 3

        vii = self.m.faces[fi]
        coord_vec = np.array(coord).reshape(3, 1)
        pos = (self.m.positions[vii] * coord_vec).sum(axis=0) / np.sum(coord)
        ref_distances = np.linalg.norm(self.m.positions[vii] - pos, axis=1)

        # Select vertices
        self._select_vertices(vii, ref_distances)
        if not self.state:
            return

        # Add more state
        more_state = {
            "action": "brush_smooth",
            "xy": xy,
        }
        self.state.update(more_state)

    def move(self, xy):
        if not self.state:
            return
        elif self.state["action"] == "morph":
            self.move_morph(xy)
        elif self.state["action"] == "brush_smooth":
            self.move_smooth(xy)

    def move_morph(self, xy):
        if self.state is None or self.state["action"] != "morph":
            return

        # Don't show radius during the drag
        self.wob_radius.visible = False

        # Get delta movement, and express in world coordinates
        dxy = np.array(xy) - self.state["xy"]
        if self.state["normal"] is not None:
            delta = self.state["normal"] * (dxy[1] / 10)
        elif "xdirection" in self.state and "ydirection" in self.state:
            delta = (
                self.state["xdirection"] * dxy[0] + self.state["ydirection"] * dxy[1]
            )
        else:
            print("zz")
            return

        # Limit the displacement, so it can never be pulled beyond the vertices participarting in the morph.
        # We limit it to 2 sigma. Vertices up to 3 sigma are displaced.
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0:  # avoid zerodivision
            delta_norm_limited = softlimit(delta_norm, 2 * self.radius)
            delta *= delta_norm_limited / delta_norm
        delta.shape = (1, 3)

        # Apply delta to gizmo
        self.wob_gizmo.world.position = self.state["pos"] + delta

        # Apply delta
        self.m.update_vertices(
            self.state["indices"],
            self.state["positions"] + delta * self.state["weights"],
        )

    def move_smooth(self, xy):
        if self.state is None or self.state["action"] != "brush_smooth":
            return

        # Get delta movement, and express in world coordinates.
        # Do a smooth "tick" when the mouse has moved 10 px.
        moved_pixel_dist = np.linalg.norm(np.array(xy) - self.state["xy"])
        if moved_pixel_dist > 10:
            self._smooth_some()
            self.state["xy"] = xy

    def _smooth_some(self, smooth_factor=0.5):
        # We only smooth each vertex with its direct neighbours, but
        # when these little smooth operations are applied recursively,
        # we end up with a pretty Gaussian smooth. Selecting multiple
        # neighbouring vertices (for each vertex), and applying an
        # actual Gaussian kernel is problematic, because we may select
        # zero vertices at low scales (woops nothing happens), or many
        # at high scales (woops performance).
        smooth_factor = max(0.0, min(1.0, float(smooth_factor)))

        faces = self.m.faces
        positions = self.m.positions
        vertex2faces = self.m.vertex2faces

        s_indices = self.state["indices"]
        s_weights = self.state["weights"]

        new_positions = np.zeros((len(s_indices), 3), np.float32)
        for i in range(len(s_indices)):
            vi = s_indices[i]
            w = s_weights[i]
            p = positions[vi]
            vii = list(vertex_get_neighbours(faces, vertex2faces, vi))
            p_delta = (positions[vii] - p).sum(axis=0) / len(vii)
            new_positions[i] = p + p_delta * (w * smooth_factor)

        # Apply new positions
        self.m.update_vertices(s_indices, new_positions)

    def finish(self):
        """Stop the morph or smooth action and commit the result."""
        self.wob_gizmo.visible = False
        if self.state:
            # Update
            indices = self.state["indices"]
            first, last = indices.min(), indices.max()
            self.geometry.colors.data[indices] = MESH_COLOR
            self.geometry.colors.update_range(first, last - first + 1)
            # self.geometry.sizes.data[indices] = 0
            # self.geometry.sizes.update_range(first, last - first + 1)
            # Commit or cancel
            if self.m.is_manifold:
                self._smooth_some(0.1)
                self.commit()
            else:
                self.cancel()
            # Post-processing
            if self.state["action"] == "morph" and self.m.is_manifold:
                self.m.resample_selection(
                    self.state["indices"], self.state["weights"], self.ref_edge_length
                )
                if self.m.is_manifold:
                    self.undo_tracker.commit_amend()
                else:
                    # Ideally this never happens, but it's a failsafe. At time of writing, it actually happens sometimes.
                    self.cancel()
                    logger.warn(
                        "Discarding resampling step because it made the mesh non-manifold"
                    )
            # todo: sometimes faces are missing or weird faces occur, due to the faces data not being synced correctlty
            # -> I've looked into why this happens but have not been able to find the cause.
            # -> Let's revisit when we implement a more efficient way to do the updates.
            self.geometry.indices.update_range()
            self.state = None


morpher = Morpher()


# %% Setup the viz


renderer = gfx.WgpuRenderer(WgpuCanvas(size=(1000, 700)))

# View 0: xy
viewport0 = gfx.Viewport(renderer)
camera0 = gfx.OrthographicCamera(8, 8)
controller0 = gfx.PanZoomController(camera0, register_events=viewport0)

# View 1: xz
viewport1 = gfx.Viewport(renderer)
camera1 = gfx.OrthographicCamera(8, 8)
controller1 = gfx.PanZoomController(camera1, register_events=viewport1)

# View 2: yz
viewport2 = gfx.Viewport(renderer)
camera2 = gfx.OrthographicCamera(8, 8)
controller2 = gfx.PanZoomController(camera2, register_events=viewport2)

# View 3: 3D
viewport3 = gfx.Viewport(renderer)
camera3 = gfx.PerspectiveCamera()
controller3 = gfx.OrbitController(camera3, register_events=viewport3)

# Setup the scenes
scenes = [gfx.Scene() for i in range(4)]
for i in range(4):
    scenes[i].add(gfx.Background(None, gfx.BackgroundMaterial(0.4, 0.6)))
    scenes[i].add(gfx.AmbientLight())

mesh0 = gfx.Mesh(
    morpher.geometry, gfx.MeshSliceMaterial(thickness=10, color_mode="vertex")
)
mesh1 = gfx.Mesh(
    morpher.geometry, gfx.MeshSliceMaterial(thickness=10, color_mode="vertex")
)
mesh2 = gfx.Mesh(
    morpher.geometry, gfx.MeshSliceMaterial(thickness=10, color_mode="vertex")
)

mesh0.xdirection, mesh0.ydirection = np.array([1, 0, 0]), np.array([0, -1, 0])
mesh1.xdirection, mesh1.ydirection = np.array([-1, 0, 0]), np.array([0, 0, -1])
mesh2.xdirection, mesh2.ydirection = np.array([0, 1, 0]), np.array([0, 0, -1])


scenes[0].add(mesh0)
scenes[1].add(mesh1)
scenes[2].add(mesh2)
scenes[3].add(morpher.world_objects)
scenes[3].add(camera3.add(gfx.DirectionalLight()))

# %% Functions to modify the mesh


def show_scene():
    scene = scenes[3]
    camera0.show_object(scene, view_dir=(0, 0, -1), up=(0, 1, 0))
    camera1.show_object(scene, view_dir=(0, -1, 0), up=(0, 0, 1))
    camera2.show_object(scene, view_dir=(-1, 0, 0), up=(0, 0, 1))
    camera3.show_object(scene)
    x, y, z, _ = scene.get_bounding_sphere()
    look_at(x, y, z)
    renderer.request_draw()


def look_at(x, y, z):
    mesh0.material.plane = 0, 0, -1, z  # xy
    mesh1.material.plane = 0, -1, 0, y  # xz
    mesh2.material.plane = -1, 0, 0, x  # yz


def add_sphere(dx=0, dy=0, dz=0):
    geo = smooth_sphere_geometry(subdivisions=1)
    positions, faces = geo.positions.data, geo.indices.data
    positions += (dx, dy, dz)

    morpher.m.add_mesh(positions, faces)
    morpher.commit()
    show_scene()


def add_bone(name):
    filename = os.path.join(DATA_DIR, name)
    if not os.path.isfile(filename):
        raise RuntimeError(f"Invalid bone dataset '{name}'.")

    geo = gfx.geometry_from_trimesh(trimesh.load(filename))
    # meshes = gfx.load_scene(filename)
    # geo = meshes[0].geometry
    morpher.m.add_mesh(geo.positions.data, geo.indices.data)
    morpher.commit()
    show_scene()


# %% Create key and mouse bindings

print(__doc__)


# Create controller, also bind it to shift, so we can always hit shift and use the camera
for contr in [controller0, controller1, controller2, controller3]:
    for k in list(contr.controls.keys()):
        contr.controls["shift+" + k] = contr.controls[k]


@renderer.add_event_handler("key_down")
def on_key(e):
    if e.key == "1":
        print("Adding a sphere.")
        add_sphere()

    elif e.key == "m":
        print("Metadata:")
        print(json.dumps(morpher.m.metadata, indent=2))
    elif e.key.lower() == "z" and ("Control" in e.modifiers or "Meta" in e.modifiers):
        if "Shift" in e.modifiers:
            morpher.redo()
        else:
            morpher.undo()
        renderer.request_draw()


@morpher.wob_front.add_event_handler(
    "pointer_down",
    "pointer_up",
    "pointer_move",
    "pointer_enter",
    "pointer_leave",
    "wheel",
)
def on_mouse_3d(e):
    if "Shift" in e.modifiers:
        # Don't react when shift is down, so that the controller can work
        morpher.highlight(None)
        renderer.request_draw()
    elif e.type == "pointer_down" and e.button == 1:
        face_index = e.pick_info["face_index"]
        face_coord = e.pick_info["face_coord"]
        if "Control" in e.modifiers or "Meta" in e.modifiers:
            vii = morpher.m.faces[face_index]
            coord_vec = np.array(face_coord).reshape(3, 1)
            pos = (morpher.m.positions[vii] * coord_vec).sum(axis=0) / np.sum(
                face_coord
            )
            look_at(*pos)
        else:
            morpher.start_morph_from_face((e.x, e.y), face_index, face_coord, True)
            renderer.request_draw()
            e.target.set_pointer_capture(e.pointer_id, e.root)
    elif e.type == "pointer_down" and e.button == 2:
        face_index = e.pick_info["face_index"]
        face_coord = e.pick_info["face_coord"]
        morpher.start_smooth((e.x, e.y), face_index, face_coord)
        renderer.request_draw()
        e.target.set_pointer_capture(e.pointer_id, e.root)
    elif e.type == "pointer_up":
        morpher.finish()
        renderer.request_draw()
    elif e.type == "pointer_move":
        if morpher.state:
            morpher.move((e.x, e.y))
        else:
            face_index = e.pick_info["face_index"]
            face_coord = e.pick_info["face_coord"]
            morpher.show_morph_grab(face_index, face_coord)
        renderer.request_draw()
    elif e.type == "pointer_enter":
        morpher.highlight(True)
        renderer.request_draw()
    elif e.type == "pointer_leave":
        morpher.highlight(False)
        renderer.request_draw()
    elif e.type == "wheel":
        if not morpher.state:
            morpher.radius *= 2 ** (e.dy / 500)
            face_index = e.pick_info["face_index"]
            face_coord = e.pick_info["face_coord"]
            morpher.show_morph_grab(face_index, face_coord)
            renderer.request_draw()
            e.cancel()


def get_directions_for_mesh(mesh):
    # Get the corresponding camera
    if mesh == mesh0:
        cam = camera0
        size = viewport0.logical_size
    elif mesh == mesh1:
        cam = camera1
        size = viewport1.logical_size
    elif mesh == mesh2:
        cam = camera2
        size = viewport2.logical_size
    # Determine vector that maps xy mouse movement to movement in world space
    movement_scale = max(cam.width / size[0], cam.height / size[1])
    return mesh.xdirection * movement_scale, mesh.ydirection * movement_scale


def on_mouse_2d(e):
    if "Shift" in e.modifiers:
        # Don't react when shift is down, so that the controller can work
        morpher.highlight(None)
        renderer.request_draw()
    elif e.type == "pointer_down" and e.button == 1:
        face_index = e.pick_info["face_index"]
        face_coord = e.pick_info["face_coord"]
        morpher.start_morph_from_face((e.x, e.y), face_index, face_coord)
        dir = get_directions_for_mesh(e.target)
        morpher.state["xdirection"], morpher.state["ydirection"] = dir
        renderer.request_draw()
        e.target.set_pointer_capture(e.pointer_id, e.root)
    elif e.type == "pointer_down" and e.button == 2:
        face_index = e.pick_info["face_index"]
        face_coord = e.pick_info["face_coord"]
        morpher.start_smooth((e.x, e.y), face_index, face_coord)
        renderer.request_draw()
        e.target.set_pointer_capture(e.pointer_id, e.root)
    elif e.type == "pointer_up":
        morpher.finish()
        renderer.request_draw()
    elif e.type == "pointer_move":
        if morpher.state:
            morpher.move((e.x, e.y))
        else:
            face_index = e.pick_info["face_index"]
            face_coord = e.pick_info["face_coord"]
            morpher.show_morph_grab(face_index, face_coord)
        renderer.request_draw()
    elif e.type == "pointer_enter":
        morpher.highlight(True)
        renderer.request_draw()
    elif e.type == "pointer_leave":
        morpher.highlight(False)
        renderer.request_draw()
    elif e.type == "wheel":
        if not morpher.state:
            morpher.radius *= 2 ** (e.dy / 500)
            face_index = e.pick_info["face_index"]
            face_coord = e.pick_info["face_coord"]
            morpher.show_morph_grab(face_index, face_coord)
            renderer.request_draw()
            e.cancel()


for m in [mesh0, mesh1, mesh2]:
    m.add_event_handler(
        on_mouse_2d,
        "pointer_down",
        "pointer_up",
        "pointer_move",
        "pointer_enter",
        "pointer_leave",
        "wheel",
    )


# %% Run


@renderer.add_event_handler("resize")
def layout(event=None):
    w, h = renderer.logical_size
    w2, h2 = (w - 30) / 2, (h - 30) / 2
    viewport0.rect = 10, 10, w2, h2
    viewport1.rect = w / 2 + 5, 10, w2, h2
    viewport2.rect = 10, h / 2 + 5, w2, h2
    viewport3.rect = w / 2 + 5, h / 2 + 5, w2, h2


def animate():
    viewport0.render(scenes[0], camera0)
    viewport1.render(scenes[1], camera1)
    viewport2.render(scenes[2], camera2)
    viewport3.render(scenes[3], camera3)
    renderer.flush()


layout()


if __name__ == "__main__":
    print(INSTRUCTIONS)

    # add_sphere(0, 0, 0)
    # add_sphere(3, 0, 0)
    add_bone("coxae.stl")

    morpher.calibrate_scale()

    renderer.request_draw(animate)
    run()
