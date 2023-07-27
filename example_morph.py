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


# --- Setup the mesh


class MeshContainer:
    """Just a container to keep all stuff related to the mesh together."""

    def __init__(self):
        # The object that stores & manages the data
        self.dynamic_mesh = DynamicMesh(None, None)

        # Tracker that implement a redo stack
        self.undo_tracker = MeshUndoTracker()
        self.dynamic_mesh.track_changes(self.undo_tracker)

        # Tracker that maps the mesh to a gfx Geometry (with live updates)
        self.geometry = DynamicMeshGeometry()
        self.dynamic_mesh.track_changes(self.geometry)

        # Two world objects, one for the front and one for the back (so we can clearly see holes)
        self.ob1 = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#6ff", flat_shading=True, side="FRONT"
            ),
        )
        self.ob2 = gfx.Mesh(
            self.geometry,
            gfx.materials.MeshPhongMaterial(
                color="#900", flat_shading=True, side="BACK"
            ),
        )

    # todo: Lets make the undo_tracker a bit more powerful so that this stuff is not needed
    #     # Include a simple undo stack
    #     self._undo_stack = []
    #     self._redo_stack = []
    #
    # def save_state(self):
    #     self._undo_stack.append(self.undo_tracker.get_version())
    #

    def commit(self):
        self.undo_tracker.commit()

    def undo(self):
        self.undo_tracker.undo(self.dynamic_mesh)

    def redo(self):
        self.undo_tracker.redo(self.dynamic_mesh)


mesh = MeshContainer()


# --- Setup the viz


renderer = gfx.WgpuRenderer(WgpuCanvas())

camera = gfx.PerspectiveCamera()
camera.show_object((0, 0, 0, 8))

scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial(0.4, 0.6)))
scene.add(camera.add(gfx.DirectionalLight()), gfx.AmbientLight())
scene.add(mesh.ob1, mesh.ob2)

# A Little helper
gizmo = gfx.Mesh(
    gfx.cylinder_geometry(0.05, 0.05, 0.5, radial_segments=16),
    gfx.MeshPhongMaterial(color="yellow"),
)
scene.add(gizmo)


# --- Functions to modify the mesh


def add_sphere(dx=0, dy=0, dz=0):
    geo = smooth_sphere_geometry(subdivisions=1)
    positions, faces = geo.positions.data, geo.indices.data
    positions += (dx, dy, dz)

    mesh.dynamic_mesh.add_mesh(positions, faces)
    mesh.commit()

    camera.show_object(scene)
    renderer.request_draw()


# --- Morph logic


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
        assert isinstance(vi, int)

        vii = [vi]
        distances = [0]
        pos = self.m.positions[vi].copy()
        normal = self.m.normals[vi].copy()

        return self._start(xy, vii, distances, pos, normal)

    def start_from_face(self, xy, fi, coord):
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
        if self.state:
            pass  # todo: probably need to reset state

        # Bring gizmo into place
        self.gizmo.visible = True
        self.gizmo.world.position = pos
        self.gizmo.local.rotation = la.quat_from_vecs((0, 0, 1), normal)

        # Select vertices
        search_distance = self.radius * 3  # 3 x std
        indices = self.m.select_vertices_over_surface(vii, distances, search_distance)
        positions = self.m.positions[indices]

        # todo: indicate the selected vertices somehow, maybe with dots?

        # If for some (future) reason, the selection is empty, cancel
        if len(indices) == 0:
            return

        # Store state
        self.state = {
            "indices": indices,
            "positions": positions,
            "xy": xy,
            "pos": pos,
            "normal": normal,
        }

    def move(self, xy):
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
        indices = self.state["indices"]
        positions = self.state["positions"]
        distances = np.linalg.norm(positions - self.state["pos"], axis=1)
        weights = gaussian_weights(distances / self.radius).reshape(-1, 1)
        self.m.update_vertices(indices, positions + delta * weights)

    def stop(self):
        self.gizmo.visible = False
        self.state = None
        mesh.commit()


morpher = Morpher(mesh.dynamic_mesh, gizmo)

# --- Create key bindings

print(__doc__)


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


@mesh.ob1.add_event_handler("pointer_down", "pointer_up", "pointer_move")
def on_mouse(e):
    if e.type == "pointer_down" and e.button == 1:
        if e.pick_info["world_object"] is mesh.ob1:
            mesh.ob1.set_pointer_capture(e.pointer_id, e.root)
            renderer.request_draw()
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
    elif e.type == "pointer_up" and e.button == 1:
        morpher.stop()
    elif e.type == "pointer_move":
        morpher.move((e.x, e.y))
        renderer.request_draw()


# --- Run

add_sphere()
add_sphere(3, 0, 0)


# Important to load the controller after binding the morph events, otherwise
# we cannot cancel events.
controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)


renderer.request_draw(animate)

run()
