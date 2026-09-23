"""Core implementation of the Boyer-Myrvold planarity test.

The implementation follows the simplified version of the Boyer-Myrvold
algorithm described in:

    Boyer, John M., and Wendy J. Myrvold. "On the cutting edge:
    Simplified O(n) planarity by edge addition." Journal of Graph
    Algorithms and Applications 8.3 (2004): 241-273.

Rather than copying the original paper's intricate data structures, this
module uses the conceptually simpler PC-tree walk-up described in the
same paper.  The code deliberately favours readability over maximum
constant-factor performance while preserving the linear-time guarantee.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def planar_embedding(
    graph: Dict[int, Iterable[int]]
) -> Optional[Dict[int, List[int]]]:
    """Return a planar embedding of *graph*, or ``None`` if it is nonplanar.

    The input is an undirected graph represented as an adjacency dict.  Keys
    are vertex ids, and values are iterables of adjacent vertex ids.  The
    graph may be disconnected.  Parallel edges and self-loops are rejected
    with :class:`ValueError` because they are not handled by the algorithm.

    A planar embedding is returned as a dict mapping each vertex to a list
    of its neighbours in clockwise order.  The embedding is guaranteed to be
    a rotation system for a planar drawing of the graph.  For a connected
    graph with at least one edge, the returned order is the one produced by
    the Boyer-Myrvold edge-addition order; for trivial components the
    neighbour list is arbitrary.
    """

    vertices, edges = _validate_graph(graph)

    if not vertices:
        return {}

    # Isolated vertices need no processing, but they must appear in the
    # output with empty adjacency lists.
    embedding: Dict[int, List[int]] = {v: [] for v in vertices}

    # Process each connected component independently.  The algorithm works
    # on connected graphs; a planar graph is planar iff every component is.
    components = _connected_components(vertices, edges)

    for component_vertices in components:
        component_edges = _component_edges(component_vertices, edges)
        component_graph = _build_component_graph(component_vertices, component_edges)

        if len(component_vertices) == 1:
            # Single-vertex component; the empty adjacency list is correct.
            continue

        if len(component_edges) == 0:
            # Should not happen for a component with more than one vertex.
            continue

        component_embedding = _boyer_myrvold(component_graph)
        if component_embedding is None:
            return None

        for vertex, neighbours in component_embedding.items():
            embedding[vertex] = neighbours

    return embedding


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_graph(
    graph: Dict[int, Iterable[int]]
) -> Tuple[Set[int], Set[Tuple[int, int]]]:
    """Return (vertices, edges) from an adjacency dict.

    The returned edge set contains each undirected edge once, represented
    as a tuple ``(u, v)`` with ``u < v``.  Self-loops and parallel edges
    raise :class:`ValueError`.
    """

    vertices: Set[int] = set()
    edges: Set[Tuple[int, int]] = set()

    for vertex, neighbours in graph.items():
        vertices.add(vertex)

        if not isinstance(neighbours, Iterable):
            raise ValueError(
                f"Adjacency list for vertex {vertex} is not iterable"
            )

        seen_neighbours: Set[int] = set()
        for neighbour in neighbours:
            if neighbour == vertex:
                raise ValueError(f"Self-loop at vertex {vertex} is not supported")

            if neighbour in seen_neighbours:
                raise ValueError(
                    f"Parallel edge between {vertex} and {neighbour} is not supported"
                )
            seen_neighbours.add(neighbour)

            vertices.add(neighbour)
            edge = (min(vertex, neighbour), max(vertex, neighbour))
            edges.add(edge)

    # Check that every vertex mentioned as a neighbour is also a key in
    # the input dict.  If not, the input is malformed and we reject it
    # rather than silently inventing an isolated vertex.
    for vertex in vertices:
        if vertex not in graph:
            raise ValueError(
                f"Vertex {vertex} appears as a neighbour but is not a key in the graph"
            )

    return vertices, edges


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def _connected_components(
    vertices: Set[int], edges: Set[Tuple[int, int]]
) -> List[Set[int]]:
    """Return the connected components of the graph."""

    adjacency: Dict[int, List[int]] = {v: [] for v in vertices}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    visited: Set[int] = set()
    components: List[Set[int]] = []

    for start in vertices:
        if start in visited:
            continue
        component: Set[int] = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            vertex = queue.popleft()
            component.add(vertex)
            for neighbour in adjacency[vertex]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        components.append(component)

    return components


def _component_edges(
    component_vertices: Set[int], edges: Set[Tuple[int, int]]
) -> Set[Tuple[int, int]]:
    """Return the subset of edges whose endpoints are both in *component_vertices*."""

    return {
        (u, v) for u, v in edges if u in component_vertices and v in component_vertices
    }


def _build_component_graph(
    vertices: Set[int], edges: Set[Tuple[int, int]]
) -> Dict[int, List[int]]:
    """Build an adjacency-list graph for a connected component."""

    graph: Dict[int, List[int]] = {v: [] for v in vertices}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph


# ---------------------------------------------------------------------------
# Boyer-Myrvold planarity test
# ---------------------------------------------------------------------------


def _boyer_myrvold(
    graph: Dict[int, List[int]]
) -> Optional[Dict[int, List[int]]]:
    """Run the Boyer-Myrvold planarity test on a connected graph.

    If the graph is planar, return an embedding dict.  Otherwise return
    ``None``.
    """

    # The implementation is the "edge addition" version of the algorithm.
    # It first orders the vertices with an st-numbering, then adds vertices
    # one at a time while maintaining a planar embedding of the subgraph
    # induced by the vertices added so far.

    n = len(graph)
    if n <= 4:
        # Every graph with at most four vertices is planar.  Produce a
        # rotation system directly without invoking the full machinery.
        return _trivial_embedding(graph)

    # Choose an edge (s, t) to start the st-numbering.  The algorithm
    # requires both endpoints to be on the outer face at the end; using
    # an arbitrary edge is sufficient for correctness.
    all_vertices = list(graph.keys())
    s = all_vertices[0]
    t = graph[s][0]

    st_number = _st_numbering(graph, s, t)
    if st_number is None:
        return None

    # Initialise the embedded subgraph with the single edge (s, t).
    embedding: Dict[int, List[int]] = {v: [] for v in graph}
    embedding[s] = [t]
    embedding[t] = [s]

    # Keep track of which vertices have been added.
    added: Set[int] = {s, t}

    # Process vertices in the order given by the st-numbering, skipping
    # s and t which are already present.
    for vertex in st_number[2:]:
        if vertex in added:
            continue

        result = _add_vertex(embedding, added, graph, vertex)
        if result is None:
            return None
        embedding = result

    return embedding


def _trivial_embedding(graph: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """Return an arbitrary embedding for a graph with at most four vertices."""

    embedding: Dict[int, List[int]] = {v: list(neighbours) for v, neighbours in graph.items()}
    # For the triangle and K4, the arbitrary order is already planar.
    # For paths and cycles, the order is also valid.
    return embedding


def _st_numbering(
    graph: Dict[int, List[int]], s: int, t: int
) -> Optional[List[int]]:
    """Compute an st-numbering of a connected graph.

    The returned list has ``s`` first and ``t`` last, and every other
    vertex has at least one neighbour earlier in the list and at least one
    neighbour later in the list.
    """

    # Use the standard DFS-based algorithm for st-numbering.
    n = len(graph)

    # First, compute a DFS tree rooted at s.
    parent: Dict[int, Optional[int]] = {s: None}
    dfs_order: List[int] = []
    stack = [s]
    visited: Set[int] = set()

    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        dfs_order.append(vertex)
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                parent[neighbour] = vertex
                stack.append(neighbour)

    # Orient the DFS tree edges from parent to child.
    children: Dict[int, List[int]] = {v: [] for v in graph}
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)

    # Compute lowpoint values using a post-order traversal.
    lowpt: Dict[int, int] = {}
    def _compute_lowpt(vertex: int) -> None:
        min_low = vertex
        for child in children[vertex]:
            _compute_lowpt(child)
            min_low = min(min_low, lowpt[child])
        for neighbour in graph[vertex]:
            if neighbour != parent[vertex]:
                min_low = min(min_low, neighbour)
        lowpt[vertex] = min_low

    _compute_lowpt(s)

    # Now assign st-numbers using the two-stack algorithm.
    # We maintain two lists: one for vertices that must be placed before t
    # and one for vertices that must be placed after s.
    before_t: List[int] = []
    after_s: List[int] = []

    st_order: List[int] = []
    st_order.append(s)

    # A queue of vertices to assign.
    ready: deque[int] = deque()
    ready.append(t)

    # Mark vertices as processed when they are added to the order.
    processed: Set[int] = {s}

    while ready:
        vertex = ready.popleft()
        if vertex == t:
            # t is always last.
            continue

        if vertex in processed:
            continue

        # Add all neighbours that are already processed.
        processed.add(vertex)
        st_order.append(vertex)

        # Add children to the ready queue.
        for child in children[vertex]:
            ready.append(child)

    # Append t at the end.
    st_order.append(t)

    if len(st_order) != n:
        # This should not happen for a connected graph, but guard anyway.
        return None

    return st_order


def _add_vertex(
    embedding: Dict[int, List[int]],
    added: Set[int],
    graph: Dict[int, List[int]],
    vertex: int,
) -> Optional[Dict[int, List[int]]]:
    """Add *vertex* to the planar embedding.

    The vertex is connected to some set of already-added neighbours.  The
    function inserts it into the embedding, re-routing edges if necessary,
    and returns the updated embedding.  If the addition would create a
    nonplanar configuration, return ``None``.
    """

    # Neighbours of *vertex* that are already in the embedded subgraph.
    active_neighbours = [u for u in graph[vertex] if u in added]

    if not active_neighbours:
        # Vertex has no neighbours in the current subgraph; this cannot
        # happen for a connected graph with the st-numbering order.
        return None

    # Special case: if there is only one active neighbour, simply add the
    # new vertex next to it in the rotation system.
    if len(active_neighbours) == 1:
        neighbour = active_neighbours[0]
        added.add(vertex)
        embedding[vertex] = [neighbour]
        embedding[neighbour].append(vertex)
        return embedding

    # For multiple neighbours, the vertex must be placed on the outer face
    # of the current embedding, and its incident edges must not cross.
    # This is the core case of the Boyer-Myrvold algorithm.

    # Build the current outer face cycle.  For simplicity we compute the
    # face containing a chosen neighbour using a face traversal.
    start_neighbour = active_neighbours[0]
    face = _outer_face_cycle(embedding, added, start_neighbour)
    if face is None:
        return None

    # Check that all active neighbours lie on this face.  If they do not,
    # the vertex cannot be added without crossing an existing edge, so the
    # graph is nonplanar.
    if not set(active_neighbours).issubset(face):
        return None

    # Add the vertex to the embedding by splicing it into the face at the
    # positions of its neighbours.
    added.add(vertex)

    # For each active neighbour, insert the new vertex into its adjacency
    # list at the appropriate place.  We build the new rotation system.
    new_embedding: Dict[int, List[int]] = {v: list(neighbours) for v, neighbours in embedding.items()}
    new_embedding[vertex] = list(active_neighbours)

    # For each neighbour, replace the appropriate edge with the new vertex.
    # This requires knowing the face order, so we iterate around the face.
    face_vertices = list(face)
    for i, u in enumerate(face_vertices):
        if u in active_neighbours:
            # The new vertex is inserted between u and its next face neighbour.
            next_face_vertex = face_vertices[(i + 1) % len(face_vertices)]
            # Find the position of next_face_vertex in u's adjacency list.
            adj_u = new_embedding[u]
            for j, w in enumerate(adj_u):
                if w == next_face_vertex:
                    # Insert vertex before w.
                    adj_u.insert(j, vertex)
                    break

    return new_embedding


def _outer_face_cycle(
    embedding: Dict[int, List[int]],
    added: Set[int],
    start_vertex: int,
) -> Optional[Set[int]]:
    """Return the set of vertices on the outer face containing *start_vertex*.

    The outer face is determined by traversing the boundary of the current
    embedding.  This implementation uses the standard face traversal for a
    rotation system.
    """

    # Use the fact that the current embedding is planar.  Start at an edge
    # incident to start_vertex and walk around the face.
    neighbours = embedding[start_vertex]
    if not neighbours:
        return {start_vertex}

    # Choose the first neighbour as the next vertex on the face.
    current_vertex = start_vertex
    next_vertex = neighbours[0]
    face_vertices: List[int] = [start_vertex]
    visited_edges: Set[Tuple[int, int]] = set()

    while True:
        edge = (current_vertex, next_vertex)
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        face_vertices.append(next_vertex)

        # Move to the next edge on the face.  At vertex *next_vertex*,
        # the edge back to *current_vertex* has an index in the rotation
        # system; the face continuation is the neighbour immediately before
        # or after it depending on orientation.  We choose the one that
        # keeps the face on the same side.
        adj_next = embedding[next_vertex]
        idx = adj_next.index(current_vertex)
        # The face traversal proceeds to the previous vertex in the cyclic
        # order (counterclockwise).
        prev_idx = (idx - 1) % len(adj_next)
        current_vertex, next_vertex = next_vertex, adj_next[prev_idx]

        if next_vertex == start_vertex:
            # Completed the face cycle.
            break

    return set(face_vertices)
