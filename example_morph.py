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


class Morpher:
    def __init__(self, dynamic_mesh, gizmo):
        self.m = dynamic_mesh
        self.gizmo = gizmo
        self.state = None

    def start(self, x, y, vi):
        pos = self.m.positions[vi].copy()
        norm = self.m.normals[vi].copy()

        # Bring gizmo into place
        self.gizmo.visible = True
        self.gizmo.world.position = pos
        self.gizmo.local.rotation = la.quat_from_vecs((0, 0, 1), norm)

        # Select base positions. Also "change" the positions to seed the undo stack.
        indices = np.array([vi], np.int32)
        positions = self.m.positions[indices]

        # Store state
        self.state = {
            "indices": indices,
            "positions": positions,
            "xy": (x, y),
            "pos": pos,
            "norm": norm,
        }

    def move(self, x, y):
        if self.state is None:
            return

        dxy = np.array([x, y]) - self.state["xy"]
        dpos = self.state["norm"] * (dxy[1] / 100)

        indices = self.state["indices"]
        positions = self.state["positions"]

        self.m.update_vertices(indices, positions + dpos)
        # mesh.undo_tracker.merge_last()

        self.gizmo.world.position = self.state["pos"] + dpos

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
            vii = mesh.dynamic_mesh.faces[face_index]
            if True:
                # Select closest vertex
                vi = vii[np.argmax(face_coord)]
                morpher.start(e.x, e.y, vi)
            else:
                # Calculate position in between vertices
                raise NotImplementedError()
    elif e.type == "pointer_up" and e.button == 1:
        morpher.stop()
    elif e.type == "pointer_move":
        morpher.move(e.x, e.y)
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
