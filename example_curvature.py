"""
Little script to demonstrate positioning a new point in between two points,
and taking curvature into account.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


n1 = np.array([1, 1, 0], np.float32)
n2 = np.array([1, 1, 0], np.float32)
n1 /= np.linalg.norm(n1)
n2 /= np.linalg.norm(n2)

p1 = np.array([10, 0, 0], np.float32)
p2 = np.array([30, 0, 0], np.float32)

dist = np.linalg.norm(p2 - p1)
dirr = p2 - p1
dirr /= dist

ort = np.cross(n1, dirr) + np.cross(n2, dirr)
ort /= np.linalg.norm(ort)
dir1 = -np.cross(n1, ort)
dir2 = +np.cross(n2, ort)
dir1 /= np.linalg.norm(dir1)
dir2 /= np.linalg.norm(dir2)

n = 64
t2 = np.linspace(0, 1, n, dtype=np.float32).reshape(n, 1)
t1 = 1 - t2

# Define position of points in between p1 and p2
curve = t1 * t2
line = t1 * p1 + t2 * p2  # linear
# line = t1 * p1 + t2 * p2 + t1 * curve * n1 * dist + t2 * curve * n2 * dist  # normals
line = (
    t1 * p1 + t2 * p2 + curve * t1 * dir1 * dist + curve * t2 * dir2 * dist
)  # curvature

line_ob = gfx.Line(
    gfx.Geometry(positions=line),
    gfx.LineMaterial(color="cyan"),
)

points_ob = gfx.Points(
    gfx.Geometry(positions=[p1, p2, line[n // 2]]),
    gfx.PointsMaterial(color="magenta", size=10),
)

normal_ob = gfx.Line(
    gfx.Geometry(positions=[p1, p1 + n1, p2, p2 + n2]),
    gfx.LineSegmentMaterial(color="red"),
)


scene = gfx.Scene()
scene.add(line_ob, normal_ob, points_ob)

renderer = gfx.WgpuRenderer(WgpuCanvas())
camera = gfx.OrthographicCamera()
camera.show_object(scene)
controller = gfx.PanZoomController(camera, register_events=renderer)

renderer.request_draw(lambda: renderer.render(scene, camera))
controller

run()
