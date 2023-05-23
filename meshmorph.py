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

import numpy as np


# We assume meshes with triangles (not quads) for now
vertices_per_face = 3


class AbstractMesh:
    """
    """
    def __init__(self, vertices, faces, oversize_factor=1.5):

        # In this method we first initialize all used attributes. This
        # serves mostly as internal docs.

        # The counts of active faces and vertices.
        self._nvertices = 0
        self._nfaces = 0

        # The number of slots for the face and vertex arrays.
        self._nvertices_slots = 0  # >= _nvertices + 1
        self._nfaces_slots = 0  # >= _nfaces

        # Array placeholders.
        # Normals are used to determine surface curvature. Colors are
        # used as a feedback mechanism to show what vertices participate
        # in a morph.
        self._vertices = None  # Nx3 float32 arrray
        self._normals = None  # Nx3 float32 array
        self._colors = None  # Nx4 float32 array
        self._faces = None  # Nx3 int32 array

        # We keep track of free slots using sets.
        self._free_vertices = set()
        self._free_faces = set()

        # The reverse mapping. Is a list because the "rows" have variable size.
        # This is probably the most important data structure of this module ...
        self._vertex2faces = []  # list mapping vi -> [fi1, fi2, fi3, ...]

        # We keep track of what has changed, so we can update efficiently to the GPU
        self._dirty_vertices = set()
        self._dirty_colors = set()
        self._dirty_faces = set()

        # todo: other attributes on the original that I'm not sure of ...
        self._meta = {}
        self._gl_objects = None
        self._version_count = 0

        # Delegate initialization
        self.set_data(vertices, faces, oversize_factor)


    def set_data(self, vertices, faces, oversize_factor=1.5):
        """Set the (new) vertex and face data.
        """

        # Check incoming arrays
        if not (isinstance(vertices, np.ndarray) and vertices.ndim == 2 and vertices.shape[1] == 3):
            raise TypeError("Vertices must be a Nx3 array")
        if not (isinstance(faces, np.ndarray) and faces.ndim == 2 and vertices.shape[1] == vertices_per_face):
            raise TypeError("Faces must be a Nx3 array")

        oversize_factor = max(1, oversize_factor or 1.5)

        # Set counts
        self._nvertices = vertices.shape[0]
        self._nfaces = faces.shape[0]
        self._nvertices_slots = int(oversize_factor * self._nvertices) + 1
        self._nfaces_slots = int(oversize_factor * self._nfaces)

        # Check-check, double-check
        assert self._nvertices_slots >= self._nvertices + 1
        assert self._nfaces_slots >= self._nfaces

        # Allocate arrays.
        self._vertices = np.empty((self._nvertices_slots, 3), np.float32)
        self._normals = np.empty((self._nvertices_slots, 3), np.float32)
        self._colors = np.empty((self._nvertices_slots, 3), np.float32)
        self._faces = np.empty((self._nfaces_slots, vertices_per_face), np.int32)

        # Copy the data
        self._vertices[1:self._nvertices+1,:] = vertices
        self._faces[0:self._nfaces,:] = faces + 1

        # Nullify the free slots
        self._vertices[0,:] = 0
        self._vertices[self._nvertices+1:,:] = np.NaN
        self._faces[self._nfaces:,:] = 0

        # Collect the free slots
        self._free_vertices = set(range(self._nvertices+1, self._nvertices_slots))
        self._free_faces = set(range(self._nfaces, self._nfaces_slots))

        # Update self._vertex2faces (the reverse mapping)
        self._calculate_vertex2faces()

        # Mark for full upload
        self._dirty_vertices.add("reset")
        self._dirty_colors.add("reset")
        self._dirty_faces.add("reset")

        # Check/ correct winding and make sure it's a solid
        self.ensure_closed()

        # todo:
        # self.data_update()
        # self.metadata_update()

    def _calculate_vertex2faces(self):
        """Do a full reset of the vertex2faces backward mapping array."""
        faces = self._faces

        # todo: what offers the best performance (at the right moment)?
        # - A python list with lists?
        # - A python list with arrays (would have to do an extra step here)?
        # - An array with lists?
        # - An array with arrays?

        self._vertex2faces = vertex2faces = [[] for i in range(self._nvertices_slots)]

        # self._vertex2faces = vertex2faces = np.empty(self._nvertices_slots, dtype=list)
        # for vi in range(1, self._nvertices_slots):
        #     vertex2faces[vi] = []

        for fi in range(self._nfaces_slots):
            face = faces[fi]
            if face[0] == 0:
                continue  # empty slot
            # Loop unrolling helps performance a bit
            vertex2faces[face[0]].append(fi)
            vertex2faces[face[1]].append(fi)
            vertex2faces[face[2]].append(fi)

    def ensure_closed(self):
        """ Ensure that the mesh is closed, that all faces have the
        same winding, and that the winding is correct (by checking that
        the volume is positive).

        Returns the number of faces that were changed to correct winding.
        Raises an error if the mesh is not a solid geometry (surely
        there is a proper mathematical term for this?).
        """

        faces = self._faces

        # Collect faces to check
        valid_face_indices, = np.where(faces[:,0 ] >0)
        faces_to_check = set(valid_face_indices)

        reversed_faces = []

        while len(faces_to_check) > 0:

            # Create new front - once for each connected component in the mesh
            vi_next = faces_to_check.pop()
            was_in_front = {vi_next}
            front = [vi_next]

            # Walk along the front
            while len(front) > 0:
                fi_check = front.pop(0)
                vi1, vi2, vi3 = faces[fi_check]
                neighbour_per_edge = [0, 0, 0]
                for fi in self.get_neighbour_faces(fi_check):
                    vj1, vj2, vj3 = faces[fi]
                    matching_vertices = {vj1, vj2, vj3} & {vi1, vi2, vi3}
                    if len(matching_vertices) >= 2:
                        if vi1 in matching_vertices and vi2 in matching_vertices:
                            neighbour_per_edge[0] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in was_in_front:
                                    front.append(fi)
                                    was_in_front.add(fi)
                                if ((vi1 == vj1 and vi2 == vj2) or (vi1 == vj2 and vi2 == vj3) or
                                                                   (vi1 == vj3 and vi2 == vj1)):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = faces[fi, 2], faces[fi, 1]
                        elif vi2 in matching_vertices and vi3 in matching_vertices:
                            neighbour_per_edge[1] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in was_in_front:
                                    front.append(fi)
                                    was_in_front.add(fi)
                                if ((vi2 == vj1 and vi3 == vj2) or (vi2 == vj2 and vi3 == vj3) or
                                                                   (vi2 == vj3 and vi3 == vj1)):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = faces[fi, 2], faces[fi, 1]
                        elif vi3 in matching_vertices and vi1 in matching_vertices:
                            neighbour_per_edge[2] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in was_in_front:
                                    front.append(fi)
                                    was_in_front.add(fi)
                                if ((vi3 == vj1 and vi1 == vj2) or (vi3 == vj2 and vi1 == vj3) or
                                                                   (vi3 == vj3 and vi1 == vj1)):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = faces[fi, 2], faces[fi, 1]

                # Now that we checked all neighbours, check if we have a neighbour on each edge.
                # If this is the case for all faces, we know that the mesh is closed. The mesh
                # can still have weird crossovers or parts sticking out though (e.g. a Klein bottle).
                if not (neighbour_per_edge[0] == 1 and neighbour_per_edge[1] == 1 and neighbour_per_edge[2] == 1):
                    if neighbour_per_edge[0] == 0 or neighbour_per_edge[1] == 0 or neighbour_per_edge[2] == 0:
                        msg = f"There is a hole in the mesh at face {fi_check} {neighbour_per_edge}"
                    else:
                        msg = f"Too many neighbour faces for face {fi_check} {neighbour_per_edge}"
                    print(msg)
                    # raise ValueError(msg)

        # todo: the below check should really be done to each connected component within the mesh.
        # For now we assume that the mesh is one single object.

        cur_vol = self.get_volume()
        if cur_vol < 0:
            # Reverse all
            faces[:,1], faces[:,2] = faces[:,2], faces[:,1]
            # How many did we really change
            nfaces_changed = self._nfaces - len(reversed_faces)
            if nfaces_changed >= 0.75 * self._nfaces:
                print(f"Mesh seems to have wrong winding - reversed!")
            # Mark reversed faces as changed for GPU
            self._dirty_faces.add("reset")
            return nfaces_changed
        else:
            # Mark reversed faces as changed for GPU
            for fi in reversed_faces:
                self._dirty_faces.add(fi)
                for i in range(3):
                    self._dirty_vertices.update(faces[fi].flat)  # also for normals
            return len(reversed_faces)

    def get_volume(self):
        """ Calculate the volume of the mesh. You probably want to run ensure_closed()
        on untrusted data when using this.
        """
        def volume_of_triangle(vi1, vi2, vi3, offset=0):
            # https://stackoverflow.com/a/1568551
            # todo: double-check this on e.g. a sphere, create two spheres, volume should double
            p1 = vertices[vi1]
            p2 = vertices[vi2]
            p3 = vertices[vi3]

            p1x, p1y, p1z = p1[:, 0], p1[:, 1], p1[:, 2]
            p2x, p2y, p2z = p2[:, 0], p2[:, 1], p2[:, 2]
            p3x, p3y, p3z = p3[:, 0], p3[:, 1], p3[:, 2]
            v321 = p3x * p2y * p1z
            v231 = p2x * p3y * p1z
            v312 = p3x * p1y * p2z
            v132 = p1x * p3y * p2z
            v213 = p2x * p1y * p3z
            v123 = p1x * p2y * p3z
            # return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)  # CCW
            return - (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)  # CW

        vertices = self._vertices
        faces = self._faces

        valid_face_indices, = np.where(faces[:,0 ] >0)
        valid_faces = faces[valid_face_indices]

        vol1 = volume_of_triangle(valid_faces[:,0], valid_faces[:,1], valid_faces[:,2]).sum()

        # Check integrity
        # vol2 = volume_of_triangle(valid_faces[:,0], valid_faces[:,1], valid_faces[:,2], 10).sum()
        # err_per = (abs(vol1) - abs(vol2)) / max(abs(vol1) + abs(vol2), 0.000000001)
        # if err_per > 0.1:
        #     raise RuntimeError("Cannot calculate volume, the mesh looks not to be closed!")
        # elif err_per > 0.001:
        #     print("WARNING: maybe the mesh is not completely closed?")

        return vol1

    def get_neighbour_vertices(self, vi):
        """ Get a list of vertex indices that neighbour the given vertex index.
        """
        faces = self._faces
        vertices = self._vertices
        face_indices = self._vertex2faces[vi]

        neighbour_vertices = set(faces[face_indices].flat)
        neighbour_vertices.discard(vi)

        # todo: is it worth converting to array?
        return neighbour_vertices
        # return np.array(neighbour_vertices, np.int32)

    def get_neighbour_faces(self, fi):
        """ Get a list of face indices that neighbour the given face index.
        """
        faces = self._faces
        vertices = self._vertices
        vertex2faces = self._vertex2faces

        neighbour_faces = set()
        for vi in faces[fi]:
            neighbour_faces.update(vertex2faces[vi])
        neighbour_faces.discard(fi)

        # todo: is it worth converting to array?
        return neighbour_faces
        # return np.array(neighbour_faces, np.int32)


if __name__ == "__main__":
    import pygfx as gfx

    # The geometries with sharp corners like cubes and all hedrons are not closed
    # because we deliberately don't share vertices, so that the normals don't interpolate
    # and make the edges look weird.
    #
    # Some other geometries, like the sphere and torus knot, also seems open thought, let's fix that!

    # geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
    # geo = gfx.sphere_geometry(1)
    # geo = gfx.geometries.tetrahedron_geometry()
    # m = AbstractMesh(geo.positions.data, geo.indices.data)

    positions = np.array(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [2, 0, 1],
            [0, 2, 3],
            [1, 0, 3],
            [2, 1, 3],
        ],
        dtype=np.int32,
    )
    m = AbstractMesh(positions, indices)

