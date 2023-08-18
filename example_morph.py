"""
Example morphing.
"""

import json

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la
from gfxmorph.maybe_pygfx import smooth_sphere_geometry, DynamicMeshGeometry
from gfxmorph import DynamicMesh, MeshUndoTracker
from gfxmorph.meshfuncs import vertex_get_neighbours


# %% Morph logic


class DynamicMeshGeometryWithSizes(DynamicMeshGeometry):
    # This is a subclass of both gfx.Geometry and MeshChangeTracker.
    # This illustrates how we can relatively easily associate additional
    # buffers with the mesh. In this case we add a buffer for points
    # sizes to show the selected vertices. If we'd instead use colors
    # to show selected vertices, that'd work in much the same way.

    def new_vertices_buffer(self, mesh):
        super().new_vertices_buffer(mesh)
        self.sizes = gfx.Buffer(np.zeros((self.positions.nitems,), np.float32))


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
        self.geometry = DynamicMeshGeometryWithSizes()
        self.m.track_changes(self.geometry)

        self.state = None
        self.radius = 0.3

        self._create_world_objects()

    def _create_world_objects(self):
        # The front, to show the mesh itself.
        self.wob_front = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#6ff", flat_shading=False, side="FRONT"
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
                opacity=0.05,
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
            self.wob_points,
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
            self.wob_wire.material.opacity = 0.075
        else:
            self.wob_radius.visible = False
            self.wob_wire.material.opacity = 0.05

    def show_morph_grab(self, fi, coord):
        # Get pos
        coordvec = np.array(coord).reshape(3, 1)
        vii = self.m.faces[fi]
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

    def start_morph_from_face(self, xy, fi, coord):
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

        normal = (self.m.normals[vii] * coord_vec).sum(axis=0)
        normal = normal / np.linalg.norm(normal)
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
        self.wob_gizmo.visible = True
        self.wob_gizmo.world.position = pos
        self.wob_gizmo.local.rotation = la.quat_from_vecs((0, 0, 1), normal)

    def _select_vertices(self, vii, ref_distances):
        # Cancel any pending changes to the mesh. If we were already dragging,
        # that operation is cancelled. If other code made uncomitted changes,
        # these are discarted too (code should have comitted).
        self.cancel()
        self.finish()

        # Select vertices
        search_distance = self.radius * 3  # 3 x std
        indices, geodesic_distances = self.m.select_vertices_over_surface(
            vii, ref_distances, search_distance
        )
        positions = self.m.positions[indices]

        # Pre-calculate deformation weights
        weights = gaussian_weights(geodesic_distances / self.radius).reshape(-1, 1)

        # If for some (future) reason, the selection is empty, cancel
        if len(indices) == 0:
            return

        # Update data for points that highlight the selection
        self.geometry.sizes.data[indices] = 7  # 2 + 20*weights.flatten()
        self.geometry.sizes.update_range(indices.min(), indices.max() + 1)

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
        delta = self.state["normal"] * (dxy[1] / 100)

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
            if self.state["action"] == "morph":
                self._smooth_some(0.25)
                # todo: resample, smooth after resample?
            indices = self.state["indices"]
            self.geometry.sizes.data[indices] = 0
            self.geometry.sizes.update_range(indices.min(), indices.max() + 1)
            self.state = None
            self.commit()


morpher = Morpher()


# %% Setup the viz


renderer = gfx.WgpuRenderer(WgpuCanvas())

camera = gfx.PerspectiveCamera()
camera.show_object((0, 0, 0, 8))

scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial(0.4, 0.6)))
scene.add(camera.add(gfx.DirectionalLight()), gfx.AmbientLight())
scene.add(morpher.world_objects)


# %% Functions to modify the mesh


def add_sphere(dx=0, dy=0, dz=0):
    geo = smooth_sphere_geometry(subdivisions=1)
    positions, faces = geo.positions.data, geo.indices.data
    positions += (dx, dy, dz)

    morpher.m.add_mesh(positions, faces)
    morpher.commit()

    camera.show_object(scene)
    renderer.request_draw()


# %% Create key and mouse bindings

print(__doc__)


# Create controller, also bind it to shift, so we can always hit shift and use the camera
controller = gfx.OrbitController(camera, register_events=renderer)
for k in list(controller.controls.keys()):
    controller.controls["shift+" + k] = controller.controls[k]


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
def on_mouse(e):
    if "Shift" in e.modifiers:
        # Don't react when shift is down, so that the controller can work
        morpher.highlight(None)
        renderer.request_draw()
    elif e.type == "pointer_down" and e.button == 1:
        face_index = e.pick_info["face_index"]
        face_coord = e.pick_info["face_coord"]
        morpher.start_morph_from_face((e.x, e.y), face_index, face_coord)
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


# %% Run

add_sphere()
add_sphere(3, 0, 0)


def animate():
    renderer.render(scene, camera)


renderer.request_draw(animate)

run()
