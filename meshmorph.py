# ## Developer notes
#
# Our internal arrays are larger than needed - we have free slots. This
# is because we must be able to dynamically add and remove vertices and
# faces. Further, we have one reserved vertex slot at index zero,
# because we must be able to mark faces as unused and still point to a
# valid vertex index.
#
# In the faces array, empty faces are denoted by setting its values to 0.
# In the vertices array, empty vertices are denoted by setting its values to NaN.
#
# We always make a copy of the given data:
# - so we control the dtype.
# - we will change the values, avoid surprises by modifying given arrays.
# - we need the first vertex to be empty.
# - we may want to initialize with some extra size.
#
# Vertex indices are denoted with vi, face indices with fi.

import time
import warnings  # todo: use warnings or logger?
import queue

import numpy as np
import maybe_pylinalg
import maybe_pygfx

# We assume meshes with triangles (not quads) for now
VERTICES_PER_FACE = 3
VERTEX_OFFSET = 0


# %% Functional stuff


def mesh_is_edge_manifold_and_closed(faces):
    """Check whether the mesh is edge-manifold, and whether it is closed.

    This is a (probably) much faster way to check that the mesh is edge-manifold.
    It does not guarantee proper manifoldness though, since there can still be faces
    that are attached via a single vertex.
    """

    # Select edges
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
    edges = edges.reshape(-1, 2)
    edges.sort(axis=1)  # note, sorting!

    # This line is the performance bottleneck. It is not worth
    # combining this method with e.g. check_oriented, because this
    # line needs to be applied to different data, so the gain would
    # be about zero.
    edges_blob = np.frombuffer(edges, dtype="V8")  # performance trick
    _, edge_counts = np.unique(edges_blob, return_counts=True)

    # The mesh is edge-manifold if edges are shared at most by 2 faces.
    is_edge_manifold = bool(edge_counts.max() <= 2)

    # The mesh is closed if it has no edges incident to just once face.
    # The following is equivalent to np.all(edge_counts == 2)
    is_closed = is_edge_manifold and bool(edge_counts.min() == 2)

    return is_edge_manifold, is_closed


def mesh_is_oriented(faces):
    """Check whether  the mesh is oriented. Also implies edge-manifoldness."""

    # Select edges. Note no sorting!
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
    edges = edges.reshape(-1, 2)

    # The magic line
    edges_blob = np.frombuffer(edges, dtype="V8")  # performance trick
    _, edge_counts = np.unique(edges_blob, return_counts=True)

    # If neighbouring faces have consistent winding, their edges
    # are in opposing directions, so the unsorted edges should have
    # no duplicates. Note that ths implies edge manifoldness,
    # because if an edge is incident to more than 2 faces, one of
    # the edges orientations would appear twice. The reverse is not
    # necessarily true though: a mesh can be edge-manifold but not
    # oriented.
    is_oriented = edge_counts.max() == 1

    return is_oriented


def mesh_volume(vertices, faces):
    """Calculate the volume of the mesh.

    It is assumed that the mesh is closed. If not, the result does
    not mean much. If the volume is negative, it is inside out.
    """
    # This code is surprisingly fast, over 10x faster than the other
    # checks. I also checked out skcg's computed_interior_volume,
    # which uses Gauss' theorem of calculus, but that is
    # considerably (~8x) slower.
    return maybe_pylinalg.volume_of_closed_mesh(vertices, faces)


def mesh_surface_area(vertices, faces):
    # see skcg computed_surface_area
    # Or simply calculate area of each triangle (vectorized)
    raise NotImplementedError()


# def _calculate_vertex2faces(self):
#     """Do a full reset of the vertex2faces backward mapping array."""
#     faces = self._faces
#
#     # todo: what offers the best performance (at the right moment)?
#     # - A python list with lists?
#     # - A python list with arrays (would have to do an extra step here)?
#     # - An array with lists?
#     # - An array with arrays?
#
#     self._vertex2faces = vertex2faces = [[] for i in range(self._nvertices_slots)]
#
#     # self._vertex2faces = vertex2faces = np.empty(self._nvertices_slots, dtype=list)
#     # for vi in range(1, self._nvertices_slots):
#     #     vertex2faces[vi] = []
#
#     for fi in range(self._nfaces_slots):
#         face = faces[fi]
#         if face[0] < VERTEX_OFFSET:
#             continue  # empty slot
#         # Loop unrolling helps performance a bit
#         vertex2faces[face[0]].append(fi)
#         vertex2faces[face[1]].append(fi)
#         vertex2faces[face[2]].append(fi)


class ReverseFaceMap:
    def __init__(self, faces):
        faces = np.asarray(faces)
        nfaces_slots = len(faces)
        self.faces = faces  # meh

        # fidx = np.arange(len(faces) * 3, dtype="i4") // 3
        # ff = faces.reshape(-1)
        #
        # remap = np.argsort(ff)
        # fidxr = fidx[remap]
        # ffr = ff[remap]
        # shifts = np.flatnonzero(ffr[1:] - ffr[:-1]) + 1

        fidx = np.arange(nfaces_slots * 3, dtype="i4")
        fidx //= 3
        ff = faces.reshape(-1)
        remap = np.argsort(ff)
        fidxr = fidx[remap]
        ffr = ff[remap]
        shifts = ffr[1:] - ffr[:-1]
        shifts_idx = np.flatnonzero(shifts)
        # these are the SPLIT points
        shifts_idx += 1
        # vert_idx is the vertex index starting at each split point
        vert_idx = np.concatenate([[ffr[0]], ffr[shifts_idx]])

        nverts = faces.max()
        padding = np.empty((nverts + 1 - len(shifts_idx),), np.int32)
        padding.fill(len(fidxr) + 1)
        shifts_idx = np.concatenate([[0], shifts_idx, padding])

        self.fidxr = fidxr
        self.shifts_idx = shifts_idx
        self.vert_idx = vert_idx

        # --

        # self.vertex2faces = vertex2faces = [[] for i in range(faces.max()+1)]
        #
        # # self.vertex2faces = vertex2faces = np.empty(self._nvertices_slots, dtype=list)
        # # for vi in range(1, self._nvertices_slots):
        # #     vertex2faces[vi] = []
        #
        # for fi in range(len(faces)):
        #     face = faces[fi]
        #     vertex2faces[face[0]].append(fi)
        #     vertex2faces[face[1]].append(fi)
        #     vertex2faces[face[2]].append(fi)

    def __getitem__(self, vi):
        # return [i for i in self.fidxr[self.shifts[vi+1]:self.shifts[vi+2]]]

        # return self.vertex2faces[vi]

        shifts_idx = self.shifts_idx
        fidxr = self.fidxr
        # vert_idx = self.vert_idx

        # # ignore zero
        # if vi == 0:
        #     return []
        # vidx = vi
        # vidx = np.searchsorted(vert_idx, vi)
        # vertex unused
        # if vert_idx[vidx] != vi:
        #     return []
        # special case start/end ranges
        # if vidx == 0:
        #     return fidxr[:shifts_idx[vidx]]
        # if vidx == vert_idx.size - 1:
        #     return fidxr[shifts_idx[vidx-1]:]
        # return fidxr[shifts_idx[vidx-1]:shifts_idx[vidx]]
        return fidxr[shifts_idx[vi] : shifts_idx[vi + 1]]


def get_neighbour_faces(faces, vertex2faces, fi, *, via_edges_only=False):
    """Get a list of face indices that neighbour the given face index."""

    if via_edges_only:
        neighbour_faces1 = set()
        neighbour_faces2 = set()
        for vi in faces[fi]:
            for fi2 in vertex2faces[vi]:
                if fi2 in neighbour_faces1:
                    neighbour_faces2.add(fi2)
                else:
                    neighbour_faces1.add(fi2)
        neighbour_faces = neighbour_faces2
    else:
        neighbour_faces = set()
        for vi in faces[fi]:
            neighbour_faces.update(vertex2faces[vi])

    neighbour_faces.discard(fi)
    return neighbour_faces


def mesh_component_labels(faces, vertex2faces, *, via_edges_only=True):
    """Split the mesh in one or more connected components."""

    # Performance notes:
    # * Using a deque for the front increases performance a tiny bit.
    # * Using set logic makes it a bit slower e.g.
    #   `new_neighbour_faces = neighbour_faces.intersection(faces_to_check)`

    # vertex2faces = ReverseFaceMap(faces)
    faces_to_check = set(range(len(faces)))

    # List of components, with each component being a list of face indices.
    component_index = -1
    component_labels = np.empty((len(faces),), np.int32)
    component_labels.fill(-1)

    while len(faces_to_check) > 0:
        # Create new front - once for each connected component in the mesh
        component_index += 1
        fi_next = faces_to_check.pop()
        front = queue.deque()
        front.append(fi_next)

        # Walk along the front until we find no more neighbours
        while front:
            fi_check = front.popleft()
            component_labels[fi_check] = component_index
            neighbour_faces = get_neighbour_faces(
                faces, vertex2faces, fi_check, via_edges_only=via_edges_only
            )
            for fi in neighbour_faces:
                if fi in faces_to_check:
                    faces_to_check.remove(fi)
                    front.append(fi)

    return component_labels


def check_only_connected_by_edges(faces, vertex2faces):
    """Check whether the mesh is only connected by edges.

    This helps checking for manifoldness. The second condition is:
    the faces incident to a vertex form a closed or an open fan.
    Another way to say this is that a group of connected faces can
    only connect by sharing an edge. It cannot connect to another
    group 1) via an edge that already has 2 faces, or 2) via only
    a vertex. The case (1) is covered by every edge having either
    1 or 2 faces. So we only have to check for the second failure.

    We do this by splitting components using vertex-connectedness,
    and then splitting each component using edge-connectedness. If
    a component has sub-components, then these are connected via a
    vertex, which violates thes rule (that faces around a vertex
    must be a fan or half-open fan).
    """

    # todo: cache these components somewhere, maybe we should force the user to have one instance per component?
    component_labels1 = mesh_component_labels(faces, vertex2faces, via_edges_only=False)

    component_labels2 = mesh_component_labels(faces, vertex2faces, via_edges_only=True)

    return component_labels1.max() == component_labels2.max()


class DynamicMeshData:
    """An object that holds mesh data that can be modified in-place.
    It has buffers that are oversized so the vertex and face array can
    grow. When the buffer is full, a larger buffer is allocated. The
    arrays are contiguous views onto the buffers. Modifications are
    managed to keep the arrays without holes.
    """

    def __init__(self):
        initial_size = 0

        # Create the buffers
        self._faces_buf = np.zeros((initial_size, 3), np.int32)
        self._positions_buf = np.zeros((initial_size, 3), np.float32)
        self._normals_buf = np.zeros((initial_size, 3), np.float32)
        self._colors_buf = np.zeros((initial_size, 4), np.float32)
        # todo: Maybe face colors are more convenient?

        # Create array views
        self._faces = self._faces_buf[:0]
        self._positions = self._positions_buf[:0]
        self._normals = self._normals_buf[:0]
        self._colors = self._colors_buf[:0]

        # Reverse map
        self._vertex2faces = []  # vi -> {fi's}

        self.version_verts = 0
        self.version_faces = 0

    @property
    def faces(self):
        # todo: return a readonly version
        return self._faces

    @property
    def vertices(self):
        # todo: return a readonly version
        # todo: vertices or positions? technically normals and colors (etc) also apply to a vertex
        return self._positions

    def _allocate_faces_buffer(self, n):
        # Sanity check
        nfaces = len(self._faces)
        assert n >= nfaces
        # Allocate new array.
        self._faces_buf = np.zeros((n, VERTICES_PER_FACE), np.int32)
        # Copy the data and reset view
        self._faces_buf[:nfaces, :] = self._faces
        self._faces = self._faces_buf[:nfaces]

    def _allocate_vertices_buffer(self, n):
        # Sanity check
        nverts = len(self._positions)
        assert n >= nverts
        # Allocate new arrays.
        self._positions_buf = np.empty((n, 3), np.float32)
        self._normals_buf = np.empty((n, 3), np.float32)
        self._colors_buf = np.empty((n, 4), np.float32)
        # Copy the data
        self._positions_buf[:nverts, :] = self._positions
        self._normals_buf[:nverts, :] = self._normals
        self._colors_buf[:nverts, :] = self._colors
        # Reset views
        self._positions = self._positions_buf[:nverts]
        self._normals = self._normals_buf[:nverts]
        self._colors = self._colors_buf[:nverts]
        # Resize reverse index
        self._vertex2faces.extend(set() for i in range(n - nverts))

    def _ensure_free_faces(self, n, resize_factor=1.5):
        """Make sure that there are at least n free slots for faces.
        If not, increase the total size of the array by resize_factor.
        """
        n = int(n)
        assert n >= 1
        assert resize_factor > 1
        free_faces = len(self._faces_buf) - len(self._faces)

        if free_faces < n:
            new_size = max(
                len(self._faces) + n, int(len(self._faces_buf) * resize_factor)
            )
            # print(f"Re-allocating faces array to {new_size} elements.")
            self._allocate_faces_buffer(new_size)

    def _ensure_free_vertices(self, n, resize_factor=1.5):
        """Make sure that there are at least n free slots for vertices.
        If not, increase the total size of the array by resize_factor.
        """
        n = int(n)
        assert n >= 1
        assert resize_factor > 1
        free_vertices = len(self._positions_buf) - len(self._positions)

        # todo: also reduce slots if below a certain treshold?
        # To do this without remapping any indices (which we want to avoid, e.g. to keep undo working)
        # we can only make the array as small as the largest index in use. But we can
        # reorganize the array a bit (just a few moves) on each resample(), so that the arrays
        # stay more or less tightly packed.

        if free_vertices < n:
            new_size = max(
                len(self._positions) + n, int(len(self._positions_buf) * resize_factor)
            )
            # print(f"Re-allocating vertex array to {new_size} elements.")
            self._allocate_vertices_buffer(new_size)

    def _get_indices_to_delete(self, indices, what):
        if isinstance(indices, int):
            return [indices]
        elif isinstance(indices, list):
            return indices
        elif isinstance(indices, np.ndarray):
            if indices.ndim == 1 and indices.dtype.kind == "i":
                return indices
            # note: we could allow deleting faces/vertices via a view, but .. maybe not?
            # elif indices.ndim == 2 and faces.shape[1] == 3 and faces.base is self._faces_buf:
            #     addr0 = self._faces_buf.__array_interface__['data'][0]
            #     addr1 = faces.__array_interface__['data'][0]

        # Else
        raise TypeError(
            "The {what} to delete must be given as int, list, or 1D int array."
        )

    def delete_faces(self, face_indices):
        to_delete = set(self._get_indices_to_delete(face_indices, "faces"))

        nfaces1 = len(self._faces_buf)
        nfaces2 = nfaces1 - len(to_delete)

        to_maybe_move = set(range(nfaces2, nfaces1))  # these are for filling the holes
        to_just_drop = to_maybe_move & to_delete  # but some of these may be at the end

        indices1 = list(to_delete - to_just_drop)
        indices2 = list(to_maybe_move - to_just_drop)
        assert len(indices1) == len(indices2)

        # Update reverse map
        vertex2faces = self._vertex2faces
        for fi in to_just_drop:
            face = self._faces[fi]
            vertex2faces[face[0]].discard(fi)
            vertex2faces[face[1]].discard(fi)
            vertex2faces[face[2]].discard(fi)
        for fi1, fi2 in zip(indices1, indices2):
            face = self._faces[fi1]
            for i in range(3):
                fii = vertex2faces[face[i]]
                fii.discard(fi1)
                fii.add(fi2)

        # Move vertices from the end into the slots of the deleted vertices
        self._faces[indices1] = self._faces[indices2]
        # Adjust the array lengths (reset views)
        self._faces_buf = self._faces_buf[:nfaces2]
        # Tidy up
        self._faces_buf[nfaces2:nfaces1] = 0

        self.version_faces += 1

    def delete_vertices(self, vertex_indices):
        # Note: defragmenting when deleting vertices is somewhat expensive
        # because we also need to update the faces. It'd likely be cheaper
        # to let the vertex buffers contain holes, and fill these up as vertices
        # are added. However, the current implementation also has advantages
        # and I like that it works the same as for the faces. Some observations:
        #
        # - With a contiguous vertex array it is easy to check if faces are valid.
        # - No nan checks anywhere.
        # - Getting free slots for vertices is straightforward without
        #   the need for additional structures like a set of free vertices.
        # - The vertices and faces can at any moment be copied and be sound. No export needed.

        to_delete = set(self._get_indices_to_delete(vertex_indices, "vertices"))

        nverts1 = len(self._positions)
        nverts2 = nverts1 - len(to_delete)

        to_maybe_move = set(range(nverts2, nverts1))
        to_just_drop = to_maybe_move & to_delete

        indices1 = list(to_delete - to_just_drop)
        indices2 = list(to_maybe_move - to_just_drop)
        assert len(indices1) == len(indices2)

        # Move vertices from the end into the slots of the deleted vertices
        self._positions[indices1] = self._positions[indices2]
        self._normals[indices1] = self._normals[indices2]
        self._colors[indices1] = self._colors[indices2]

        # Adjust the array lengths (reset views)
        self._positions = self._positions_buf[:nverts2]
        self._normals = self._normals_buf[:nverts2]
        self._colors = self._colors_buf[:nverts2]

        # Zero out the free slots, just to tidy up
        self._positions_buf[nverts2:nverts1] = 0
        self._normals_buf[nverts2:nverts1] = 0
        self._colors_buf[nverts2:nverts1] = 0

        # Update the faces that refer to the moved indices
        for vi1, vi2 in zip(indices1, indices2):
            self._faces[self._faces == vi1] = vi2

        # Update reverse map
        vertex2faces = self._vertex2faces
        for vi in to_delete:
            assert len(vertex2faces[vi]), "Trying to delete an in-use vertex."
        for vi1, vi2 in zip(indices1, indices2):
            vertex2faces[vi2] = vertex2faces[vi1]

        self.version_verts += 1

    def add_faces(self, new_faces):
        # Check incoming array
        faces = np.asarray(new_faces, np.int32)
        if not (
            isinstance(faces, np.ndarray)
            and faces.ndim == 2
            and faces.shape[1] == VERTICES_PER_FACE
        ):
            raise TypeError("Faces must be a Nx3 array")
        # We want to be able to assume that there is at least one face, and 3 valid vertices
        # (or 4 faces and 4 vertices for a closed mesh)
        if len(faces) == 0:
            raise ValueError("Cannot add zero faces.")
        # Check sanity of the faces
        if faces.min() < 0 or faces.max() >= len(self._positions):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        n = len(faces)
        n1 = len(self._faces)
        n2 = n1 + n

        self._ensure_free_faces(n)
        self._faces = self._faces_buf[:n2]
        self._faces[n1:] = faces

        # Update reverse map
        vertex2faces = self._vertex2faces
        for i in range(len(faces)):
            fi = i + n1
            face = faces[i]
            vertex2faces[face[0]].add(fi)
            vertex2faces[face[1]].add(fi)
            vertex2faces[face[2]].add(fi)

        self.version_faces += 1

    def add_vertices(self, new_positions):
        # Check incoming array
        positions = np.asarray(new_positions, np.float32)
        if not (
            isinstance(positions, np.ndarray)
            and positions.ndim == 2
            and positions.shape[1] == 3
        ):
            raise TypeError("Vertices must be a Nx3 array")
        if len(positions) == 0:
            raise ValueError("Cannot add zero vertices.")

        n = len(positions)
        n1 = len(self._positions)
        n2 = n1 + n

        self._ensure_free_vertices(n)
        self._positions = self._positions_buf[:n2]
        self._normals = self._normals_buf[:n2]
        self._colors = self._colors_buf[:n2]

        self._positions[n1:] = new_positions
        self._normals[n1:] = 0.0
        self._colors[n1:] = 0.7, 0.7, 0.7, 1.0

        # Update reverse map
        vertex2faces = self._vertex2faces
        for vi in range(n1, n2):
            if len(vertex2faces[vi]) > 0:
                print("WARNING: found faces referenced on stub vertices.")

        self.version_verts += 1

    # def allocate_faces(self, n):
    #     """Add n new faces to the mesh. Return an nx3 array with the new faces,
    #     which is a view on the faces buffer (so one can write to it).
    #     """
    #     n = int(n)
    #     n1 = len(self._faces)
    #     n2 = n1 + n
    #
    #     self._ensure_free_faces(n)
    #     self._faces = self._faces_buf[:n2]
    #     return self._faces[n1:]
    #
    # def allocate_vertices(self, n):
    #     """Add n new vertices to the mesh. Return an nx3 array with the new positions,
    #     which is a view on the positions buffer (so one can write to it).
    #     """
    #     n = int(n)
    #     n1 = len(self._positions)
    #     n2 = n1 + n
    #
    #     self._ensure_free_vertices(n)
    #     self._positions = self._positions_buf[:n2]
    #     self._normals = self._normals_buf[:n2]
    #     self._colors = self._colors_buf[:n2]
    #     return self._positions[n1:]


# todo: better name
class AbstractMesh:
    """Representation of a mesh, with utilities to modify it.
    The idea is that this can be subclassed to hook it up in a visualization
    system (like pygfx), e.g. process updates in a granular way.
    """

    def __init__(self, vertices, faces):
        self._data = DynamicMeshData()
        self._components = ()

        self._props = {}
        self._props_faces = (
            "is_edge_manifold",
            "is_closed",
            "is_oriented",
            "component_labels",
            "is_only_connected_by_edges",
        )
        self._props_verts = ()
        self._props_verts_and_faces = "volume", "surface"

        # Delegate initialization
        self.add_mesh(vertices, faces)

    def _check_prop(self, name):
        assert name in self._props_faces or name in self._props_verts
        if self._props.get("version_faces", 0) != self._data.version_faces:
            self._props["version_faces"] = self._data.version_faces
            for x in self._props_faces + self._props_verts_and_faces:
                self._props.pop(x, None)
        if self._props.get("version_verts", 0) != self._data.version_verts:
            self._props["version_verts"] = self._data.version_verts
            for x in self._props_verts + self._props_verts_and_faces:
                self._props.pop(x, None)
        return name in self._props

    @property
    def faces(self):
        return self._data.faces

    @property
    def vertices(self):
        return self._data.vertices

    @property
    def component_labels(self):
        """A tuple of connected components that this mesh consists of."""
        if not self._check_prop("component_labels"):
            component_labels = mesh_component_labels(
                self._data.faces, self._data._vertex2faces
            )
            self._props["component_labels"] = component_labels
        return self._props["component_labels"]

    @property
    def is_connected(self):
        """Whether the mesh is a single connected component."""
        # todo: what should this value be if it consists of two parts connected by a single vertex?
        return len(components) <= 1

    @property
    def is_edge_manifold(self):
        """Whether the mesh is edge_manifold.

        This is another way of saying that each edge is part of either
        1 or 2 faces. It is one of the two condition for a mesh to be
        manifold. You could also call it weak-manifold, I suppose.
        """
        if not self._check_prop("is_edge_manifold"):
            is_edge_manifold, is_closed = mesh_is_edge_manifold_and_closed(
                self._data.faces
            )
            self._props["is_edge_manifold"] = is_edge_manifold
            self._props["is_closed"] = is_closed
        return self._props["is_edge_manifold"]

    @property
    def is_manifold(self):
        """Whether the mesh is manifold.

        The mesh being manifold means that:

        * Each edge is part of 1 or 2 faces.
        * The faces incident to a vertex form a closed or an open fan.
        """
        if not self.is_edge_manifold:
            return False
        if not self._check_prop("is_only_connected_by_edges"):
            is_only_connected_by_edges = check_only_connected_by_edges(
                self._data.faces, self._data._vertex2faces
            )
            self._props["is_only_connected_by_edges"] = is_only_connected_by_edges
        return self._props["is_only_connected_by_edges"]

    @property
    def is_closed(self):
        """Whether the mesh is closed.

        If this tool is used in a GUI, you can use this flag to show a
        warning symbol. If this value is False, a warning with more
        info is shown in the console.
        """
        if not self._check_prop("is_closed"):
            is_edge_manifold, is_closed = mesh_is_edge_manifold_and_closed(
                self._data.faces
            )
            self._props["is_edge_manifold"] = is_edge_manifold
            self._props["is_closed"] = is_closed
        return self._props["is_closed"]

    @property
    def is_oriented(self):
        """Whether the mesh is orientable.

        The mesh being orientable means that the face orientation (i.e. winding) is
        consistent - each two neighbouring faces have the same orientation.
        """
        # todo: I think that a mesh can be edge_manifold, connected, but not manifold.
        if not self._check_prop("is_oriented"):
            is_oriented = mesh_is_oriented(self._data.faces)
            self._props["is_oriented"] = is_oriented
        return self._props["is_oriented"]

    @property
    def is_volumetric(self):
        """Whether the mesh has a positive volume.

        If not, the mesh is probably inside-out. CCW winding is assumed.
        """
        # todo: ...

    @property
    def edges(self):
        """All edges of this mesh as pairs of vertex indices

        Returns
        -------
        ndarray, [n_faces, 3, 2]
            pairs of vertex-indices specifying an edge.
            the ith edge is the edge opposite from the ith vertex of the face

        """
        # todo: only use valid faces, maybe per component?
        array = self.faces[:, [[1, 2], [2, 0], [0, 1]]]
        array.setflags(write=False)
        return array

    @property
    def metadata(self):
        """A dict with metadata about the mesh."""
        arrays = self._faces, self._vertices, self._colors, self._normals
        nb = sum([a.nbytes for a in arrays if a is not None])

        return {
            "is_closed": self.is_closed,
            "components": self._component_count,
            "vertices": self._nvertices,
            "faces": self._nfaces,
            "free_vertices": len(self._free_vertices),
            "free_faces": len(self._free_faces),
            "approx_mem": f"{nb/2**20:0.2f} MiB"
            if nb > 2**20
            else f"{nb/2**10:0.2f} KiB",
        }

    # %%

    def add_mesh(self, vertices, faces):
        """Add vertex and face data.

        The data is copied and the internal data structure.
        """
        faces = np.asarray(faces, np.int32)

        # The DynamicMeshData class also does some checks, but it will
        # only check if incoming faces match any vertex, not just the
        # ones we add here, so we perform that check here.
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        vertex_index_offset = len(self._data.vertices)
        self._data.add_vertices(vertices)
        self._data.add_faces(faces + vertex_index_offset)

    # todo: questions:
    # Do we want to keep track of components?
    # Is so, how would we keep track?
    # If you add faces / vertices, would you add them to a specific component?
    #  -> but what if you add stuff making it no longer connected??
    #  -> and what if you add a face that connects two components?
    #  -> this is a more general question: when you modify the mesh, it may be (temporary) non-manifold, un-closed, etc.
    #  -> so even if you mark an added face as belonging to a specific component, we cannot really rely on it?
    # Why would we want to keep track of components? How would we use it?
    # In an UI you could show components in a list. Also e.g. highlight a specific component when selected, or make others transparent.

    # %% could probably a functional API ?

    def repair_manifold(self):
        # Remove collapsed faces

        valid_faces = self._faces[:, 0] > 0
        collapsed_faces = np.array([len(set(f)) != len(f) for f in self._faces], bool)
        (faces2remove,) = np.where(valid_faces & collapsed_faces)

        if len(faces2remove):
            self._faces[faces2remove] = (
                VERTEX_OFFSET - 1,
                VERTEX_OFFSET - 1,
                VERTEX_OFFSET - 1,
            )
            self._free_faces.update(faces2remove)
            self._mesh_props = {}

        # Remove duplicate faces

        while True:
            unique_faces, index, counts = np.unique(
                np.sort(self._faces, axis=1),
                axis=0,
                return_index=True,
                return_counts=True,
            )
            counts[unique_faces[:, 0] < VERTEX_OFFSET] = 0
            if counts.max() == 1:
                break
            faces2remove = index[counts > 1]

            if len(faces2remove):
                self._faces[faces2remove] = 0, 0, 0
                self._free_faces.update(faces2remove)
                self._mesh_props = {}

        # todo: Remove non-manifold faces?
        # Remove vertex-connected faces
        components = self._split_components(None, via_edges_only=False)
        is_only_connected_by_edges = True

        final_components = []
        vertices2dedupe_per_component = []

        for component in components:
            subcomponents = self._split_components(component, via_edges_only=True)
            index_offset = len(final_components)
            final_components.extend(subcomponents)
            while len(vertices2dedupe_per_component) < len(final_components):
                vertices2dedupe_per_component.append(set())
            if len(subcomponents) > 1:
                for i1 in range(len(subcomponents)):
                    for i2 in range(i1 + 1, len(subcomponents)):
                        vii1 = set(self._faces[subcomponents[i1]].flat)
                        vii2 = set(self._faces[subcomponents[i2]].flat)
                        reused = vii1 & vii2
                        vertices2dedupe_per_component[index_offset + i2].update(reused)

        self._component_labels = -1 * np.ones((len(self._faces),), np.int32)
        for i in range(len(final_components)):
            self._component_labels[final_components[i]] = i

        for i in range(len(final_components)):
            component = final_components[i]
            dedupe = vertices2dedupe_per_component[i]
            for vi1 in dedupe:
                vi2 = self._get_free_vertices(1)[0]
                # todo: must only be applied for this ith component. Waiting to implement that because I want to try having faces views first.
                self._faces[self._faces == vi1] = vi2
                self._mesh_props = {}

    def repair_closed(self):
        pass
        # todo: we could detect duplicate vertices, and use it to stitch the faces together.

    def clean_small_components(self, min_faces=4):
        pass

    def repair_oriented(self):
        """ """

        # This implementation walks over the surface using a front. The
        # algorithm is similar to the one for splitting the mesh in
        # connected components, except it does more work at the deepest
        # nesting.
        #
        # It starts out from one face, and reverses the neighboring
        # faces if they don't match the winding of the current face.
        # And so on. Faces that have been been processed cannot be
        # reversed again. So the fix operates as a wave that flows over
        # the mesh, with the first face defining the reference winding.
        #
        # The repair can only fail if the mesh is not manifold or when
        # it is not orientable (i.e. a Mobius strip or Klein bottle).
        #
        # A vectorized implementation might also be possible, but I'm
        # not sure if a closed form solution exists, so it might need
        # to work iteratively? In any case, since this is a repair
        # operation, performance is of less importance.

        faces = self._faces

        # Collect faces to check
        (valid_face_indices,) = np.where(self._faces[:, 0] >= VERTEX_OFFSET)
        faces_to_check = set(valid_face_indices)

        component_count = 0
        reversed_faces = []

        while len(faces_to_check) > 0:
            # Create new front - once for each connected component in the mesh
            component_count += 1
            vi_next = faces_to_check.pop()
            front = queue.deque()
            front.append(vi_next)

            # Walk along the front
            while len(front) > 0:
                fi_check = front.popleft()
                vi1, vi2, vi3 = faces[fi_check]
                for fi in self.get_neighbour_faces(fi_check, via_edges_only=True):
                    if fi in faces_to_check:
                        faces_to_check.remove(fi)
                        front.append(fi)

                        vj1, vj2, vj3 = faces[fi]
                        matching_vertices = {vj1, vj2, vj3} & {vi1, vi2, vi3}
                        if len(matching_vertices) == 2:
                            if vi3 not in matching_vertices:
                                # vi1 in matching_vertices and vi2 in matching_vertices
                                if (
                                    (vi1 == vj1 and vi2 == vj2)
                                    or (vi1 == vj2 and vi2 == vj3)
                                    or (vi1 == vj3 and vi2 == vj1)
                                ):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = (
                                        faces[fi, 2],
                                        faces[fi, 1],
                                    )
                            elif vi1 not in matching_vertices:
                                # vi2 in matching_vertices and vi3 in matching_vertices
                                if fi in faces_to_check:
                                    if (
                                        (vi2 == vj1 and vi3 == vj2)
                                        or (vi2 == vj2 and vi3 == vj3)
                                        or (vi2 == vj3 and vi3 == vj1)
                                    ):
                                        reversed_faces.append(fi)
                                        faces[fi, 1], faces[fi, 2] = (
                                            faces[fi, 2],
                                            faces[fi, 1],
                                        )
                            elif vi2 not in matching_vertices:
                                # vi3 in matching_vertices and vi1 in matching_vertices
                                if fi in faces_to_check:
                                    if (
                                        (vi3 == vj1 and vi1 == vj2)
                                        or (vi3 == vj2 and vi1 == vj3)
                                        or (vi3 == vj3 and vi1 == vj1)
                                    ):
                                        reversed_faces.append(fi)
                                        faces[fi, 1], faces[fi, 2] = (
                                            faces[fi, 2],
                                            faces[fi, 1],
                                        )

            return len(reversed_faces)

    def repair_volumetric(self):
        if self.get_volume() < 0:
            tmp = self._faces[:, 1].copy()
            self._faces[:, 1] = self._faces[:, 2]
            self._faces[:, 2] = tmp

    def check_manifold_closed_deprecated(self):
        """Does a proper check that the mesh is manifold, and also whether its closed."""

        # This algorithm checks the faces around each vertex. This
        # allows checking all conditions for manifoldness, and also
        # whether the mesh is closed. It's not even that slow. But using
        # another combination of check-methods is faster to cover all
        # mesh-properties. Leaving this for a bit, beacause we may use
        # it to do repairs.

        # todo: clean this up

        fliebertjes = 0
        flobbertes = 0
        isopen = 0

        # todo: this selection could be done beforehand, of if we split components first, we may already have lists available?
        for vi1 in range(len(self._vertices)):
            if np.isnan(self._vertices[vi1][0]):
                continue

            # all_vertex_indices = []
            # for fi in self._vertex2faces[vi1]:
            #     vii = set(self._faces[fi])
            #     vii.remove(vi1)
            #     all_vertex_indices.extend(vii)
            #
            # # todo: the below check is not enough to detect all of the above, but it can detect fans, and if it does that considerably
            # # faster then we could do that first, and only to the slower check if its not a fan.
            # # That way we may get reasonable fast code for closed meshes, while being somewhat slower along the edges of open meshes.
            # unique_ids, counts = np.unique(all_vertex_indices, return_counts=True)
            # if (counts==2).all():
            #     continue  # a fan, all is well!
            #
            # if (counts > 2).any():
            #     fliebertjes += 1
            #     continue  # fail, next!
            #
            # if (counts==1).any():
            #     isopen += 1
            #     if (counts==1).sum() > 2:
            #         flobbertes += 1
            #
            # continue

            # Fom the current vertex (vi1), we select all incident faces
            # (all faces that use the vertex) and from that select all
            # neighbouring vertices.
            #
            # Our goal is to check whether these faces form a closed or an open fan.
            # There may not be fliebertjes (a face attached by an edge that is already
            # incident to 2 other faces), or flobbertjes (a face that is attached by only
            # a vertex.
            #
            #
            #  v2     /|v3
            #  |\    / |
            #  | \  /  |
            #  |  vi1  |
            #  |/   \  |
            #  v1    \ |
            #         \|v4
            #

            # vertex_indices_per_face = []
            vertex_counts = {}
            for fi in self._vertex2faces[vi1]:
                vii = set(self._faces[fi])
                vii.remove(vi1)
                for vi in vii:
                    # vertex_counts[vi] = vertex_counts.get(vi, 0) + 1
                    vertex_counts.setdefault(vi, []).append(fi)
                # vertex_indices_per_face.append(vii)

            # for vii in vertex_indices_per_face:
            #     xx = sum(vertex_counts[vi] for vi in vii)
            #     if xx == 0:
            #         self._is_manifold = False
            #     elif xx > 2:
            #         self.

            for vi2, fii in vertex_counts.items():
                if len(fii) == 2:
                    pass  # ok!
                elif len(fii) > 2:
                    fliebertjes += 1  # Bad!
                elif len(fii) == 1:
                    isopen += 1
                    # Might be an edge, or could be a flobbertje
                    fi = fii[0]
                    vii = set(self._faces[fi])
                    if len(vii) != 3:
                        flobbertes += 1
                        continue
                        # todo: a degenerate triangle, should we remove that from top instead?
                    vii.remove(vi1)
                    vii.remove(vi2)
                    vi = vii.pop()
                    if len(vertex_counts[vi]) == 1:
                        if len(vertex_counts) > 2:
                            # If this is the outer vertex on a triangle on the egde, we don't
                            # need to check this further
                            flobbertes += 1

        is_manifold = fliebertjes == 0 and flobbertes == 0
        is_closed = isopen == 0

        return is_manifold, is_closed

    def _fix_stuff_deprecated(self):
        """Check that the mesh is closed.

        This method also autocorrects the winding of indidual faces,
        as well as of a whole sub-mesh (connected component) so that their
        volumes are positive.

        This method reports its findings by showing a warning and by setting
        a private attribute (which can e.g. be checked in tests and debugging).
        """

        # This alogotitm is a port of what we did in Arbiter to check the mesh.
        # This code can:
        # * Check edge-manifoldness, closed, orientability.
        # * Repair winding, repair meshes being inside-out.
        # Leaving for a bit, because we may use some of it for doing repairs.

        # todo: clean this up
        # todo: the _check_manifold_nr2 checks manifoldness and closeness, so in here we can restrict to the winding (detecting and fixing) and detecting connected components.
        # todo: if holes are detected, we can check the vertices at these locations for duplicates, and we can merge those duplicates to create a closed mesh from a "stitched" one.
        faces = self._faces

        # Collect faces to check
        (valid_face_indices,) = np.where(self._faces[:, 0] > 0)
        faces_to_check = set(valid_face_indices)

        # Keep track of issues
        component_count = 0
        total_reversed_count = 0
        unclosed_case1 = []
        unclosed_case2 = []

        self._is_oriented = self._is_manifold = True

        while len(faces_to_check) > 0:
            # Create new front - once for each connected component in the mesh
            component_count += 1
            vi_next = faces_to_check.pop()
            faces_in_this_component = {vi_next}
            reversed_faces = []
            front = queue.deque()
            front.append(vi_next)

            # Walk along the front
            while len(front) > 0:
                fi_check = front.popleft()
                vi1, vi2, vi3 = faces[fi_check]
                neighbour_per_edge = [0, 0, 0]
                for fi in self.get_neighbour_faces(fi_check):
                    vj1, vj2, vj3 = faces[fi]
                    matching_vertices = {vj1, vj2, vj3} & {vi1, vi2, vi3}
                    if len(matching_vertices) == 3:
                        self._is_manifold = False
                    elif len(matching_vertices) == 2:
                        # todo: we now know that we have two matches, so we can write these three if-s better, I think
                        if vi1 in matching_vertices and vi2 in matching_vertices:
                            neighbour_per_edge[0] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in faces_in_this_component:
                                    front.append(fi)
                                    faces_in_this_component.add(fi)
                                if (
                                    (vi1 == vj1 and vi2 == vj2)
                                    or (vi1 == vj2 and vi2 == vj3)
                                    or (vi1 == vj3 and vi2 == vj1)
                                ):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = (
                                        faces[fi, 2],
                                        faces[fi, 1],
                                    )
                        elif vi2 in matching_vertices and vi3 in matching_vertices:
                            neighbour_per_edge[1] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in faces_in_this_component:
                                    front.append(fi)
                                    faces_in_this_component.add(fi)
                                if (
                                    (vi2 == vj1 and vi3 == vj2)
                                    or (vi2 == vj2 and vi3 == vj3)
                                    or (vi2 == vj3 and vi3 == vj1)
                                ):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = (
                                        faces[fi, 2],
                                        faces[fi, 1],
                                    )
                        elif vi3 in matching_vertices and vi1 in matching_vertices:
                            neighbour_per_edge[2] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in faces_in_this_component:
                                    front.append(fi)
                                    faces_in_this_component.add(fi)
                                if (
                                    (vi3 == vj1 and vi1 == vj2)
                                    or (vi3 == vj2 and vi1 == vj3)
                                    or (vi3 == vj3 and vi1 == vj1)
                                ):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = (
                                        faces[fi, 2],
                                        faces[fi, 1],
                                    )

                # Now that we checked all neighbours, check if we have a neighbour on each edge.
                # If this is the case for all faces, we know that the mesh is closed. The mesh
                # can still have weird crossovers or parts sticking out though (e.g. a Klein bottle).
                if not (
                    neighbour_per_edge[0] == 1
                    and neighbour_per_edge[1] == 1
                    and neighbour_per_edge[2] == 1
                ):
                    if (
                        neighbour_per_edge[0] == 0
                        or neighbour_per_edge[1] == 0
                        or neighbour_per_edge[2] == 0
                    ):
                        unclosed_case1.append((fi_check, neighbour_per_edge))
                    else:
                        self._is_manifold = False
                        unclosed_case2.append((fi_check, neighbour_per_edge))

            self._is_oriented = not reversed_faces

            # We've now found one connected component (there may be more)
            faces_in_this_component = list(faces_in_this_component)
            cfaces = self._faces[faces_in_this_component]
            volume = maybe_pylinalg.volume_of_closed_mesh(self._vertices, cfaces)

            if volume < 0:
                # Reverse the whole component
                cfaces[:, 1], cfaces[:, 2] = cfaces[:, 2].copy(), cfaces[:, 1].copy()
                self._faces[faces_in_this_component] = cfaces
                total_reversed_count += len(faces_in_this_component) - len(
                    reversed_faces
                )
                self._dirty_faces.update(faces_in_this_component)
            else:
                # Mark reversed faces as changed for GPU
                total_reversed_count += len(reversed_faces)
                self._dirty_faces.update(reversed_faces)

        self._component_count = component_count

        # Report on winding - all is well, we corrected it!
        self._n_reversed_faces = total_reversed_count
        # if total_reversed_count > 0:
        #     warnings.warn(
        #         f"Auto-corrected the winding of {total_reversed_count} faces."
        #     )

        # Report on the mesh not being closed - this is a problem!
        # todo: how bad is this problem again? Should we abort or still allow e.g. morphing the mesh?
        self._original_unclosed_faces = {fi for fi, _ in unclosed_case1} | {
            fi for fi, _ in unclosed_case2
        }
        if unclosed_case1:
            lines = [f"    - {fi} ({nbe})" for fi, nbe in unclosed_case1]
            n = len(unclosed_case1)
            # warnings.warn(
            #     f"There is a hole in the mesh at {n} faces:\n" + "\n".join(lines)
            # )
        if unclosed_case2:
            lines = [f"    - {fi} ({nbe})" for fi, nbe in unclosed_case2]
            n = len(unclosed_case2)
            # warnings.warn(
            #     f"Too many neighbour faces at {n} faces:\n" + "\n".join(lines)
            # )

    def validate_internals(self):
        raise NotImplementedError()
        # todo: xx

    def get_volume(self):
        """Calculate the volume of the mesh.

        It is assumed that the mesh is closed. If not, the result does
        not mean much. If the volume is negative, it is inside out.
        """
        return sum(c.get_volume() for c in self._components)

    def get_surface_area(self):
        # see skcg computed_surface_area
        # Or simply calculate area of each triangle (vectorized)
        return sum(c.get_surface_area() for c in self._components)

    def split(self):
        """Return a list of Mesh objects, one for each connected component."""
        # I don't think we need this for our purpose, but this class is capable
        # of doing something like this, so it could be a nice util.
        raise NotImplementedError()

    def merge(self, other):
        raise NotImplementedError()

    # %% Walk over the surface

    def get_neighbour_vertices(self, vi):
        """Get a list of vertex indices that neighbour the given vertex index."""
        faces = self._data.faces
        vertices = self._data.vertices
        face_indices = self._vertex2faces[vi]

        neighbour_vertices = set(faces[face_indices].flat)
        neighbour_vertices.discard(vi)

        # todo: is it worth converting to array?
        return neighbour_vertices
        # return np.array(neighbour_vertices, np.int32)

    def get_neighbour_faces(self, fi, *, via_edges_only=False):
        """Get a list of face indices that neighbour the given face index."""

        vertex2faces = self._vertex2faces

        if via_edges_only:
            neighbour_faces1 = set()
            neighbour_faces2 = set()
            for vi in self._faces[fi]:
                for fi2 in vertex2faces[vi]:
                    if fi2 in neighbour_faces1:
                        neighbour_faces2.add(fi2)
                    else:
                        neighbour_faces1.add(fi2)
            neighbour_faces = neighbour_faces2
        else:
            neighbour_faces = set()
            for vi in self._faces[fi]:
                neighbour_faces.update(vertex2faces[vi])

        neighbour_faces.discard(fi)
        return neighbour_faces

    def get_closest_vertex(self, ref_pos):
        """Get the vertex index closest to the given 3D point, and its distance."""
        ref_pos = np.asarray(ref_pos, np.float32)
        if ref_pos.shape != (3,):
            raise ValueError("ref_pos must be a position (3 values).")

        distances = np.linalg.norm(self.vertices - ref_pos, axis=1)
        vi = np.nanargmin(distances)
        return vi, distances[vi]

    def select_vertices_over_surface(self, ref_vertex, max_distance):
        """Given a reference vertex, select more nearby vertices (over the surface).
        Returns a dict mapping vertex indices to dicts containing {pos, color, sdist, adist}.
        """
        vertices = self.vertices
        vi0 = int(ref_vertex)
        p0 = vertices[vi0]
        # selected_vertices = {vi: dict(pos=[x1, y1, z1], color=color, sdist=0, adist=0)}
        selected_vertices = {vi0}
        vertices2check = [(vi0, 0)]
        while len(vertices2check) > 0:
            vi1, cumdist = vertices2check.pop(0)
            p1 = vertices[vi1]
            for vi2 in self.get_neighbour_vertices(vi1):
                if vi2 not in selected_vertices:
                    p2 = vertices[vi2]
                    sdist = cumdist + np.linalg.norm(p2 - p1)
                    if sdist < max_distance:
                        adist = np.linalg.norm(p2 - p0)
                        # selected_vertices[vi2] = dict(pos=[xn, yn, zn], color=color, sdist=sdist, adist=adist)
                        selected_vertices.add(vi2)
                        vertices2check.append((vi2, sdist))
        return selected_vertices


if __name__ == "__main__":
    import pygfx as gfx

    # The geometries with sharp corners like cubes and all hedrons are not closed
    # because we deliberately don't share vertices, so that the normals don't interpolate
    # and make the edges look weird.
    #
    # Some other geometries, like the sphere and torus knot, also seems open thought, let's fix that!

    def get_tetrahedron():
        """A closed tetrahedron as simple list objects, so we can easily add stuff to create corrupt meshes."""
        vertices = [
            [-1, 0, -1 / 2**0.5],
            [+1, 0, -1 / 2**0.5],
            [0, -1, 1 / 2**0.5],
            [0, +1, 1 / 2**0.5],
        ]
        faces = [
            [2, 0, 1],
            [0, 2, 3],
            [1, 0, 3],
            [2, 1, 3],
        ]
        return vertices, faces, True

    # geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
    # geo = gfx.sphere_geometry(1)
    # geo = gfx.geometries.tetrahedron_geometry()
    geo = maybe_pygfx.smooth_sphere_geometry(1)
    # m = AbstractMesh(geo.positions.data, geo.indices.data)
    m = AbstractMesh(*get_tetrahedron()[:2])

    # m._check_manifold_nr1()

    # positions = np.array(
    #     [
    #         [1, 1, 1],
    #         [-1, -1, 1],
    #         [-1, 1, -1],
    #         [1, -1, -1],
    #     ],
    #     dtype=np.float32,
    # )
    #
    # indices = np.array(
    #     [
    #         [2, 0, 1],
    #         [0, 2, 3],
    #         [1, 0, 3],
    #         [2, 1, 3],
    #     ],
    #     dtype=np.int32,
    # )
    # m = AbstractMesh(positions, indices)
