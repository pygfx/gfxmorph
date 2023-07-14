import queue

import numpy as np

from . import maybe_pylinalg


def make_vertex2faces(faces):
    """Create a simple map to map vertex indices to a list of face indices."""
    faces = np.asarray(faces, np.int32)
    nverts = faces.max() + 1

    vertex2faces = [[] for _ in range(nverts)]
    for fi in range(len(faces)):
        face = faces[fi]
        vertex2faces[face[0]].append(fi)
        vertex2faces[face[1]].append(fi)
        vertex2faces[face[2]].append(fi)
    return vertex2faces


def vertex_get_neighbours(faces, vertex2faces, vi):
    """Get a set of vertex indices that neighbour the given vertex index.

    Connectedness is via the edges.
    """
    neighbour_vertices = set()
    for fi in vertex2faces[vi]:
        neighbour_vertices.update(faces[fi])
    return neighbour_vertices


def face_get_neighbours1(faces, vertex2faces, fi):
    """Get a set of face indices that neighbour the given face index.

    Connectedness is either via an edge or via a vertex.
    """
    neighbour_faces = set()
    for vi in faces[fi]:
        neighbour_faces.update(vertex2faces[vi])
    neighbour_faces.remove(fi)
    return neighbour_faces


def face_get_neighbours2(faces, vertex2faces, fi):
    """Get two sets of face indices that neighbour the given face index.

    The first comprises of both vertex- and edge connections, the second
    only consists of faces connected via an edge.
    """
    neighbour_faces1 = set()
    neighbour_faces2 = set()
    for vi in faces[fi]:
        for fi2 in vertex2faces[vi]:
            if fi2 == fi:
                pass
            elif fi2 in neighbour_faces1:
                neighbour_faces2.add(fi2)
            else:
                neighbour_faces1.add(fi2)
    return neighbour_faces1, neighbour_faces2


def vertex_get_incident_face_groups(
    faces, vertex2faces, vi_check, *, face_adjacency=None
):
    """Get the groups of faces incident to the given vertex.

    If there are zero groups, the vertex has no incident faces. If there
    is exactly one group, the faces incident to the given vertex form
    a (closed or open) fan. If there is more than one group, the mesh
    is not manifold (and can be repaired by duplicating this vertex for
    each group).
    """

    #
    #   Diagram 1           Diagram 2
    #   _________                ____
    #  |\       /|              |   /|
    #  | \  D  / |              |D / |
    #  |  \   /  |              | /  |
    #  |   \ /   |              |/ C |
    #  | B  O  C |              O----|
    #  |   / \   |              |\ B |
    #  |  /   \  |              | \  |
    #  | /  A  \ |              |A \ |
    #  |/_______\|              |___\|
    #
    #
    #   Diagram 3           Diagram 4
    #   _________                ____
    #  |\       /|              |   /|
    #  | \  D  / | _      _     |D / |
    #  |  \   / _|- |    | -._  | /  |
    #  |   \ /.- | E|    |E   -.|/ C |
    #  | B  O----|--|    |------O----|
    #  |   / \ C |              |\ B |
    #  |  /   \  |              | \  |
    #  | /  A  \ |              |A \ |
    #  |/_______\|              |___\|
    #
    #
    # In the two diagrams above, the vertex indicated by the big O is
    # the reference vertex. On the left (diagram 1 and 3) we see a
    # closed fan, and on the right (diagram 2 and 4) an open fan. In
    # the top diagrams all is well, but in the bottom diagrams (3 and
    # 4) there is an additional face E attached to the vertex, breaking
    # the vertex-manifold condition. Note that it does not matter
    # whether E is a lose vertex, part of a strip, or part of an
    # (otherwise) manifold and closed component. Note also that E can
    # even be a face on the same component that faces a-d are part of.
    # That component can still be edge-manifold, closed, and oriented.

    # Note that the algorithm below does not detect duplicate faces or
    # edges with 3 incident faces. Therefore, to be be vertex-manifold,
    # a mesh must *also* be edge-manifold.

    faces_to_check = set(vertex2faces[vi_check])
    groups = []

    while faces_to_check:
        group = []
        groups.append(group)
        fi_next = faces_to_check.pop()
        front = queue.deque()
        front.append(fi_next)
        while front:
            fi_check = front.popleft()
            group.append(fi_check)
            if face_adjacency is not None:
                neighbour_faces2 = face_adjacency[fi_check]
            else:
                _, neighbour_faces2 = face_get_neighbours2(
                    faces, vertex2faces, fi_check
                )
            for fi in neighbour_faces2:
                if fi in faces_to_check:
                    faces_to_check.remove(fi)
                    front.append(fi)
    return groups


def mesh_is_edge_manifold_and_closed(faces):
    """Check whether the mesh is edge-manifold, and whether it is closed.

    This implementation is based on vectorized numpy code and therefore very fast.
    """

    # Special case
    if len(faces) == 0:
        return True, True

    # Select edges
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
    edges = edges.reshape(-1, 2)
    edges.sort(axis=1)  # note, sorting!

    # This line is the performance bottleneck. It is not worth
    # combining this method with e.g. check_oriented, because this
    # line needs to be applied to different data, so the gain would
    # be about zero.
    edges_blob = np.frombuffer(edges, dtype="V8")  # performance trick
    unique_blob, edge_counts = np.unique(edges_blob, return_counts=True)

    # The mesh is edge-manifold if edges are shared at most by 2 faces.
    is_edge_manifold = bool(edge_counts.max() <= 2)

    # The mesh is closed if it has no edges incident to just once face.
    # The following is equivalent to np.all(edge_counts == 2)
    is_closed = is_edge_manifold and bool(edge_counts.min() == 2)

    return is_edge_manifold, is_closed


def mesh_get_non_manifold_edges(faces):
    """Detect non-manifold edges.

    These are returned as a dict ``(vi1, vi2) -> [fi1, fi2, ..]``.
    It maps edges (pairs of vertex indices) to a list face indices incident
    to that edge. I.e. to repair the edge, the faces incidense to each
    edge can be removed. Afterwards, the nonmanifold vertices can be repaired,
    followed by repairing the holes.

    If the returned dictionary is empty, the mesh is edge-manifold. In
    other words, for each edge there are either one or two incident
    faces.
    """

    # Special case
    if len(faces) == 0:
        return {}

    edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
    edges = edges.reshape(-1, 2)
    edges.sort(axis=1)  # note, sorting!

    edges_blob = np.frombuffer(edges, dtype="V8")  # performance trick
    unique_blob, edge_counts = np.unique(edges_blob, return_counts=True)

    # Code above is same as first part in mesh_is_edge_manifold_and_closed()

    # Collect faces for each corrupt unique edge. This happens one by
    # one. Maybe this can be vectorized, but it only hurts performance
    # for corrupt meshes, so not a big deal.
    nonmanifold_edges = {}
    corrupt_indices = np.where(edge_counts > 2)[0]
    for i in corrupt_indices:
        eii = np.where(edges_blob == unique_blob[i])[0]
        edge = tuple(edges[eii[0]])  # Eeach ei in eii refers to the same edge
        fii = list(set(fi for fi in eii // 3))
        nonmanifold_edges[edge] = fii

    return nonmanifold_edges


def mesh_is_oriented(faces):
    """Check whether  the mesh is oriented. Also implies edge-manifoldness.

    This implementation is based on vectorized numpy code and therefore very fast.
    """
    # Special case
    if len(faces) == 0:
        return True

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


def mesh_get_volume(vertices, faces):
    """Calculate the volume of the mesh.

    It is assumed that the mesh is oriented and closed. If not, the result does
    not mean much. If the volume is negative, it is inside out.

    This implementation is based on vectorized numpy code and therefore very fast.
    """
    # Special case
    if len(faces) == 0:
        return 0

    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32)

    # This code is surprisingly fast, over 10x faster than the other
    # checks. I also checked out skcg's computed_interior_volume,
    # which uses Gauss' theorem of calculus, but that is
    # considerably (~8x) slower.
    return maybe_pylinalg.volume_of_closed_mesh(vertices, faces)


def mesh_get_surface_area(vertices, faces):
    """Calculate the surface area of the mesh."""
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32)

    r1 = vertices[faces[:, 0], :]
    r2 = vertices[faces[:, 1], :]
    r3 = vertices[faces[:, 2], :]
    face_normals = np.cross((r2 - r1), (r3 - r1))  # assumes CCW
    faces_areas = 0.5 * np.linalg.norm(face_normals, axis=1)
    return faces_areas.sum()


def mesh_get_component_labels(faces, vertex2faces, *, via_edges_only=True):
    """Split the mesh in one or more connected components.

    Returns a 1D array that contains component indices for all faces.
    """

    # Performance notes:
    # * Using a deque for the front increases performance a tiny bit.
    # * Using set logic makes rather then control flow does not seem to matter much.
    # * The labels that we're interested in we set directly in an array
    #   so we avoid the step to go from set -> list -> array labels.

    faces_to_check = set(range(len(faces)))

    # Array to store component labels. (Using list vs array does not seem to affect performance.)
    component_labels = np.empty((len(faces),), np.int32)
    component_labels.fill(-1)
    component_index = -1

    while len(faces_to_check) > 0:
        # Create new front - once for each connected component in the mesh
        component_index += 1
        fi_next = faces_to_check.pop()
        front = queue.deque()
        front.append(fi_next)

        while front:
            fi_check = front.popleft()
            component_labels[fi_check] = component_index

            if via_edges_only:
                _, neighbour_faces = face_get_neighbours2(faces, vertex2faces, fi_check)
            else:
                neighbour_faces = face_get_neighbours1(faces, vertex2faces, fi_check)

            for fi in neighbour_faces:
                if fi in faces_to_check:
                    faces_to_check.remove(fi)
                    front.append(fi)

    return component_labels


def mesh_get_non_manifold_vertices(faces, vertex2faces):
    """Detect non-manifold vertices.

    These are returned as a dict ``vi -> [[fi1, fi2, ..], [fi3, fi4, ...]]``.
    It maps vertex indices to a list of face-index-groups, each
    representing a fan attached to the vertex. I.e. to repair the vertex,
    a duplicate vertex must be created for each group (except one).

    If the returned dictionary is empty, and the mesh is edge-manifold,
    the mesh is also vertex-manifold. In other words, for each vertex,
    the faces incident to that vertex form a closed or an open fan.

    """
    # This implementation literally performs a test for each vertex.
    # Since the per-vertex test involves querying neighbours a lot, it
    # is somewhat slow. I've tried a few things to check
    # vertex-manifoldness faster, without success. I'll summerize here
    # for furure reference and maybe help others that walk a similar path:
    #
    # By splitting the mesh in connected components twice, once using
    # vertex-connectedness, and once using edge-connectedness, it's
    # easy to spot non-manifold edges in between components. It's even
    # possible to do this in a single iteration! However, it does not
    # reveal non-manifold vertices within components :/
    #
    # Then I tried a few ways to select suspicious faces/vertices during
    # the algorithm that splits connected components, and then examine
    # each suspicious vertex. Since that algorithm already walks over
    # the surface of the mesh and requires information on the face
    # neighbours, overhead can be combined/reused, basically resulting
    # in getting the split components for free. That part works, but I
    # have not been able to find a way to reliably select suspicious
    # faces/vertices.
    #
    # One approach was to mark indirect neighbours (faces connected
    # with only a vertex) as they were encountered, and unmark them
    # when that face was seen again, but now as a direct neighbour (and
    # via an edge that includes the suspicious vertex). Unfortunately,
    # depending on the implementation details, this approach was either
    # too permissive (missing corrupt vertices), slower than the brute
    # force, or leaving so many false positives that you might as well
    # use the brute force method.
    #
    # I was pretty fond of the idea to score each vertex based on its
    # role in connecting neighbours for each face. For an indirect
    # neighbour it scored negative points, for a direct neighbour it
    # scored positive points. If the net score was negative, it was a
    # suspicious vertex and examined properly. It was tuned so that it
    # did not generates false positives for fans of 6 faces  (by scoring
    # edge-neighbours 1.5 times higher). Interestingly, this method is
    # able to reliably detect non-manifold vertices in a large variety
    # of topologies. Unfortunately there are a few vertex-based
    # fan-to-fan connections for which it fails. The example that
    # explains this best is a fan of 3 faces connected to another fan
    # of 3 faces. From the viewpoint of the face (annotated with 'a'
    # below) this configuration is indiscernible from a closed 6-face
    # fan. This means we cannot detect this case without generating a
    # false positive for a *very* common type of fan.
    #   __
    #  |\ | /|
    #  |_\|/_|
    #  | /|\a|
    #  |/ |_\|
    #
    # Conclusion: if we want a fast vertex-manifold check, we should
    # probably just use Cython ...

    # suspicious_vertices = np.unique(faces)
    suspicious_vertices = set(faces.flat)

    # Calculate face adjecency once beforehand, instead of 3x per face
    face_adjacency = [None for _ in range(len(faces))]
    for fi in range(len(faces)):
        _, neighbour_faces2 = face_get_neighbours2(faces, vertex2faces, fi)
        face_adjacency[fi] = neighbour_faces2

    nonmanifold_vertices = {}
    for vi in suspicious_vertices:
        groups = vertex_get_incident_face_groups(
            faces, vertex2faces, vi, face_adjacency=face_adjacency
        )
        if len(groups) > 1:
            nonmanifold_vertices[vi] = groups

    return nonmanifold_vertices


def mesh_get_boundaries(faces):
    """Select the boundaries of the mesh.

    Returns a list of boundaries, where each boundary is a list of
    vertices. Each boundary is a loop (the first and last vertex are
    connected via a boundary edge), and the order of vertices is in the
    appropriate winding direction.

    This function can raise a RuntimeError if it runs into a part of
    the mesh that is non-manifold.
    """
    # Special case
    if len(faces) == 0:
        return []

    faces = np.asarray(faces, np.int32)

    # Select edges
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
    edges = edges.reshape(-1, 2)
    sorted_edges = np.sort(edges, axis=1)

    # Find the unique edges
    edges_blob = np.frombuffer(sorted_edges, dtype="V8")  # performance trick
    unique_blob, unique_index, edge_counts = np.unique(
        edges_blob, return_index=True, return_counts=True
    )
    unique_edges = np.frombuffer(unique_blob, dtype=np.int32).reshape(-1, 2)

    # Find the boundary edges: those that only have one incident face
    (boundary_edge_indices,) = np.where(edge_counts == 1)

    # Build map vi -> eii
    nvertices = faces.max() + 1
    vertex2edges = [[] for _ in range(nvertices)]
    for ei in boundary_edge_indices:
        vi1, vi2 = unique_edges[ei]
        vertex2edges[vi1].append(ei)
        vertex2edges[vi2].append(ei)

    # Sanity check. Note that the mesh can still be non-manifold, as
    # long as it's vertex-manifold along the boundaries, it's fine.
    # Edge-manifoldness is less of an issue, because it means an edge
    # has more than 2 faces, which makes it not a boundary anyway.
    for vi in range(nvertices):
        if len(vertex2edges[vi]) > 2:  # pragma: no cover
            raise RuntimeError("The mesh is not vertex-manifold.")

    # Now group them into boundaries ...
    #
    # This looks a bit like the code we use to walk over the surface
    # of the mesh, e.g. to find connected components. This is similar
    # except we walk over boundaries (edges with just one incident
    # face).

    eii_to_check = set(boundary_edge_indices)
    boundaries = []

    while eii_to_check:
        # Select arbitrary edge as starting point, and add both its edges to
        # a list that we will build up further. The order of these initial
        # vertices determines the direction that we walk in.
        ei = ei_first = eii_to_check.pop()
        original_edge = edges[unique_index[ei]]
        boundary = [original_edge[1]]
        boundaries.append(boundary)

        while True:
            # Get the two vertex indices on the current edge.
            # One is the last added vertex, the other represents the direction
            # we should move in.
            vi1, vi2 = unique_edges[ei]
            vi = vi2 if vi1 == boundary[-1] else vi1
            boundary.append(vi)
            # We look up what boundary-edges are incident to this vertex.
            # One is the previous edge. the other will be the next.
            ei1, ei2 = vertex2edges[vi]
            ei = ei2 if ei1 == ei else ei1
            # We either keep going, or have gone full circle
            if ei in eii_to_check:
                eii_to_check.remove(ei)
            else:
                if not (
                    ei == ei_first and boundary[0] == boundary[-1]
                ):  # pragma: no cover
                    raise RuntimeError(
                        "This should not happen, but if it does, I think the mesh is not manifold."
                    )
                boundary.pop(-1)
                break

        # Sanity check for the algorithm below. It shows the assumptions
        # that we make about the result, but that are (should be) true
        # by the mesh being manifold and how the algorithm works.
        # Can be uncommented when developing.
        # edge_index = unique_index[ei]
        # assert np.all(sorted_edges[edge_index] == unique_edges[ei])
        # original_edge = edges[edge_index]
        # assert boundary[0] == original_edge[1]
        # assert boundary[1] == original_edge[0]

    return boundaries


def mesh_get_consistent_face_orientation(faces, vertex2faces):
    """Make the orientation of the faces consistent.

    Returns a modified copy of the faces which may have some of the
    faces flipped. Faces are flipped by swapping the 2nd and 3d value.
    Note that the winding of the returned array may still not be
    consistent, when the mesh is not manifold or when it is not
    orientable (i.e. a Mobius strip or Klein bottle).
    """

    # This implementation walks over the surface using a front. The
    # algorithm is similar to the one for splitting the mesh in
    # connected components, except it does more work at the deepest
    # nesting.
    #
    # It starts out from one face, and reverses the neighboring
    # faces that don't match the winding of the current face. And
    # so on. Faces that have been been processed cannot be reversed
    # again. So the fix operates as a wave that flows over the mesh,
    # with the first face defining the winding.
    #
    # A closed form solution for this problem does not exist. The skcg
    # lib uses pycosat to find the solution. The below might be slower
    # (being implemented in pure Python), but it's free of dependencies
    # and speed matters less in a repair function, I suppose.

    # Make a copy of the faces, so we can reverse them in-place.
    faces = faces.copy()

    reversed_faces = []
    faces_to_check = set(range(len(faces)))

    while len(faces_to_check) > 0:
        # Create new front - once for each connected component in the mesh
        vi_next = faces_to_check.pop()
        front = queue.deque()
        front.append(vi_next)

        # Walk along the front
        while front:
            fi_check = front.popleft()
            vi1, vi2, vi3 = faces[fi_check]
            _, neighbours = face_get_neighbours2(faces, vertex2faces, fi_check)
            for fi in neighbours:
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

        # If over half the faces were flipped, we flip the hole thing.
        if len(reversed_faces) > 0.5 * len(faces):
            tmp = faces[:, 2].copy()
            faces[:, 2] = faces[:, 1]
            faces[:, 1] = tmp

        return faces


def mesh_stitch_boundaries(vertices, faces, *, atol=1e-5):
    """Stitch touching boundaries together by de-deuplicating vertices.

    Stitching only happens when it does not result in a non-manifold
    mesh. Returns the modified faces array.

    It is up to the caller to remove collapsed faces and any vertices
    that are no longer used.
    """

    # Make a copy of the faces that we can modify
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32).copy()

    # Detect boundaries. A list of vertex indices (vi's)
    boundaries = mesh_get_boundaries(faces)

    # Combine to get a subset of vertices for faster checking.
    boundary_vertices = np.concatenate(boundaries)
    boundary_positions = vertices[boundary_vertices]

    # Keep track of what vi's were already handled as a duplicate
    duplicate_mask = np.zeros((len(vertices),), bool)

    # Create a mask so we can look up the boundary index from a vi
    boundary_mask = np.empty((len(vertices),), np.int32)
    boundary_mask.fill(-1)
    for i, boundary in enumerate(boundaries):
        boundary_mask[boundary] = i

    # The deduplication can cause the mesh to become non-manifold.
    # To prevent this we appy the changes in chunks and validate
    # the change before it is applied. These "chunks" are groups
    # of boundaries that connect. To handle these groups we use a
    # stack-based algorithm simular to those used to traverse over
    # the surface of the mesh.

    boundary_indices_to_check = set(range(len(boundaries)))

    while boundary_indices_to_check:
        boundary_indices_in_this_group = [boundary_indices_to_check.pop()]

        faces_for_this_group = faces.copy()

        while boundary_indices_in_this_group:
            boundary = boundaries[boundary_indices_in_this_group.pop(0)]

            # The next two pieces of code are the heart of the algorithm,
            # most of the rest is to apply the grouping and preventing non-manifoldness.

            # Collect duplicate vertices.
            duplicate_map = {}
            for vi in boundary:
                if not duplicate_mask[vi]:
                    mask3 = np.isclose(
                        boundary_positions, vertices[vi], atol=atol or 0, rtol=0
                    )
                    mask = mask3.sum(axis=1) == 3  # all el's of a vertex must match
                    if mask.sum() > 1:
                        vii = boundary_vertices[mask]
                        duplicate_mask[vii] = True
                        duplicate_map[vi] = vii

            # Now we apply them to a copy of the faces array. Some heavy iterations here ...
            for vi1, vii in duplicate_map.items():
                for vi2 in vii:
                    if vi2 != vi1:
                        faces_for_this_group[faces == vi2] = vi1

            # See if we need to process more boundaries in this group
            group = set()
            for vii in duplicate_map.values():
                group.update(boundary_mask[vii])
            group = group & boundary_indices_to_check
            boundary_indices_to_check.difference_update(group)
            boundary_indices_in_this_group.extend(group)

        # We've now processed one group of boundaries ...

        # Check whether the new face array is manifold, taking collapsed faces into account
        not_collapsed = np.array(
            [len(set(f)) == len(f) for f in faces_for_this_group], bool
        )
        tmp_faces = faces_for_this_group[not_collapsed]
        v2f = make_vertex2faces(tmp_faces)
        is_edge_manifold, _ = mesh_is_edge_manifold_and_closed(tmp_faces)
        non_manifold_verts = mesh_get_non_manifold_vertices(tmp_faces, v2f)
        is_manifold = is_edge_manifold and not non_manifold_verts

        # If it is, apply!
        if is_manifold:
            faces = faces_for_this_group

    return faces
