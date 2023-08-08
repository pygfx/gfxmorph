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


# %% Setup the mesh


class DynamicMeshGeometryWithSizes(DynamicMeshGeometry):
    def new_vertices_buffer(self, mesh):
        super().new_vertices_buffer(mesh)
        # Add sizes so we can show selected vertices.
        # If we'd instead use colors to show selected vertices, that'd
        # work in much the same way.
        self.sizes = gfx.Buffer(np.zeros((self.positions.nitems,), np.float32))


class MeshContainer:
    """Just a container to keep all stuff related to the mesh together."""

    def __init__(self):
        # The object that stores & manages the data
        self.dynamic_mesh = DynamicMesh(None, None)

        # Tracker that implement a redo stack
        self.undo_tracker = MeshUndoTracker()
        self.dynamic_mesh.track_changes(self.undo_tracker)

        # Tracker that maps the mesh to a gfx Geometry (with live updates)
        self.geometry = DynamicMeshGeometryWithSizes()
        self.dynamic_mesh.track_changes(self.geometry)

        # Two world objects, one for the front and one for the back (so we can clearly see holes)
        self.ob1 = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#6ff", flat_shading=False, side="FRONT"
            ),
        )
        self.ob2 = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#900", flat_shading=False, side="BACK"
            ),
        )
        self.ob3 = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#777",
                opacity=0.05,
                wireframe=True,
                wireframe_thickness=2,
                side="FRONT",
            ),
        )

    def cancel(self):
        self.undo_tracker.cancel(self.dynamic_mesh)

    def commit(self):
        self.undo_tracker.commit()

    def undo(self):
        self.undo_tracker.undo(self.dynamic_mesh)

    def redo(self):
        self.undo_tracker.redo(self.dynamic_mesh)


mesh = MeshContainer()


# %% Setup the viz


renderer = gfx.WgpuRenderer(WgpuCanvas())

camera = gfx.PerspectiveCamera()
camera.show_object((0, 0, 0, 8))

scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial(0.4, 0.6)))
scene.add(camera.add(gfx.DirectionalLight()), gfx.AmbientLight())
scene.add(mesh.ob1, mesh.ob2, mesh.ob3)

# Helper to show points being grabbed
vertex_highlighter = gfx.Points(
    mesh.geometry,
    gfx.PointsMaterial(color="yellow", vertex_sizes=True),
)
scene.add(vertex_highlighter)

# Gizmo helper
gizmo = gfx.Mesh(
    gfx.cylinder_geometry(0.05, 0.05, 0.5, radial_segments=16),
    gfx.MeshPhongMaterial(color="yellow"),
)
gizmo.visible = False
scene.add(gizmo)

# Radius helper
radius_helper = gfx.Mesh(
    smooth_sphere_geometry(subdivisions=2),
    gfx.MeshPhongMaterial(color="yellow", opacity=0.2, side="front"),
)
radius_helper.visible = False
scene.add(radius_helper)


# %% Functions to modify the mesh


def add_sphere(dx=0, dy=0, dz=0):
    geo = smooth_sphere_geometry(subdivisions=1)
    positions, faces = geo.positions.data, geo.indices.data
    positions += (dx, dy, dz)

    mesh.dynamic_mesh.add_mesh(positions, faces)
    mesh.commit()

    camera.show_object(scene)
    renderer.request_draw()


# %% Morph logic


def softlimit(i, limit):
    """Make i be withing <-limit, +limit>, but using soft slopes."""
    f = np.exp(-np.abs(i) / limit)
    return -limit * (f - 1) * np.sign(i)


def gaussian_weights(f):
    return np.exp(-0.5 * f * f)  # Gaussian kernel (sigma 1, mu 0)


class Morpher:
    def __init__(self, dynamic_mesh, gizmo):
        self.m = dynamic_mesh
        self.gizmo = gizmo
        self.state = None
        self.radius = 0.3

    def start_from_vertex(self, xy, vi):
        """Initiate a drag, based on a vertex index.

        This method may feel less precise than using ``start_from_face``,
        but is included for cases where picking capabilities are limited.
        """
        assert isinstance(vi, int)

        vii = [vi]
        distances = [0]
        pos = self.m.positions[vi].copy()
        normal = self.m.normals[vi].copy()

        return self._start(xy, vii, distances, pos, normal)

    def start_from_face(self, xy, fi, coord):
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

        return self._start(xy, vii, distances, pos, normal)

    def _start(self, xy, vii, distances, pos, normal):
        # Cancel any pending changes to the mesh. If we were already dragging,
        # that operation is cancelled. If other code made uncomitted changes,
        # these are discarted too (code should have comitted).
        mesh.cancel()
        self.stop()

        # Bring gizmo into place
        self.gizmo.visible = True
        self.gizmo.world.position = pos
        self.gizmo.local.rotation = la.quat_from_vecs((0, 0, 1), normal)

        # Select vertices
        search_distance = self.radius * 3  # 3 x std
        indices = self.m.select_vertices_over_surface(vii, distances, search_distance)
        positions = self.m.positions[indices]

        # Pre-calculate deformation weights
        abs_distances = np.linalg.norm(positions - pos, axis=1)
        weights = gaussian_weights(abs_distances / self.radius).reshape(-1, 1)

        # If for some (future) reason, the selection is empty, cancel
        if len(indices) == 0:
            return

        # Update data for highlighter
        mesh.geometry.sizes.data[indices] = 7  # 2 + 20*weights.flatten()
        mesh.geometry.sizes.update_range(indices.min(), indices.max() + 1)

        # Store state
        self.state = {
            "indices": indices,
            "positions": positions,
            "weights": weights,
            "xy": xy,
            "pos": pos,
            "normal": normal,
        }

    def move(self, xy):
        """Process a mouse movement."""
        if self.state is None:
            return

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
        self.gizmo.world.position = self.state["pos"] + delta

        # Apply delta
        self.m.update_vertices(
            self.state["indices"],
            self.state["positions"] + delta * self.state["weights"],
        )

    def stop(self):
        """Stop the morph drag and commit the result."""
        # AK: I thought of also introducing a cancel method, but it
        # feels more natural to just finish the drag and then undo it.
        self.gizmo.visible = False
        if self.state:
            indices = self.state["indices"]
            mesh.geometry.sizes.data[indices] = 0
            mesh.geometry.sizes.update_range(indices.min(), indices.max() + 1)
            self.state = None
            mesh.commit()


morpher = Morpher(mesh.dynamic_mesh, gizmo)

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
        print(json.dumps(mesh.dynamic_mesh.metadata, indent=2))
    elif e.key.lower() == "z" and ("Control" in e.modifiers or "Meta" in e.modifiers):
        if "Shift" in e.modifiers:
            mesh.redo()
        else:
            mesh.undo()
        renderer.request_draw()


@mesh.ob1.add_event_handler(
    "pointer_down",
    "pointer_up",
    "pointer_move",
    "pointer_enter",
    "pointer_leave",
    "wheel",
)
def on_mouse(e):
    if "Shift" in e.modifiers:
        radius_helper.visible = False
        renderer.request_draw()
    elif e.type == "pointer_down" and e.button == 1:
        mesh.ob1.set_pointer_capture(e.pointer_id, e.root)
        face_index = e.pick_info["face_index"]
        face_coord = e.pick_info["face_coord"]
        if False:
            # Select closest vertex
            vii = mesh.dynamic_mesh.faces[face_index]
            vi = int(vii[np.argmax(face_coord)])
            morpher.start_from_vertex((e.x, e.y), vi)
        else:
            # Calculate position in between vertices
            morpher.start_from_face((e.x, e.y), face_index, face_coord)
        renderer.request_draw()
    elif e.type == "pointer_up" and e.button == 1:
        morpher.stop()
        renderer.request_draw()
    elif e.type == "pointer_move":
        if morpher.state:
            morpher.move((e.x, e.y))
            radius_helper.visible = False
            renderer.request_draw()
        else:
            face_index = e.pick_info["face_index"]
            face_coord = e.pick_info["face_coord"]
            vii = mesh.dynamic_mesh.faces[face_index]
            coord_vec = np.array(face_coord).reshape(3, 1)
            pos = (mesh.dynamic_mesh.positions[vii] * coord_vec).sum(axis=0) / np.sum(
                coord_vec
            )
            radius_helper.local.position = pos
            radius_helper.local.scale = morpher.radius
            radius_helper.visible = True
            renderer.request_draw()
    elif e.type == "pointer_enter":
        radius_helper.visible = True
        mesh.ob3.material.opacity = 0.075
        renderer.request_draw()
    elif e.type == "pointer_leave":
        radius_helper.visible = False
        mesh.ob3.material.opacity = 0.05
        renderer.request_draw()
    elif e.type == "wheel":
        morpher.radius *= 2 ** (e.dy / 500)
        radius_helper.local.scale = morpher.radius
        renderer.request_draw()
        e.cancel()


# %% Run

add_sphere()
add_sphere(3, 0, 0)


def animate():
    renderer.render(scene, camera)


renderer.request_draw(animate)

run()
