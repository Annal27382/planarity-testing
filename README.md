# planarity-testing

Determines whether an undirected graph is planar and, if so, returns a planar embedding using the Boyer-Myrvold algorithm.

## Usage

```python
from planarity_testing import planar_embedding

graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1],
}

embedding = planar_embedding(graph)
if embedding is None:
    print("Graph is nonplanar")
else:
    print(embedding)
    # {0: [1, 2], 1: [0, 2], 2: [0, 1]}
```

The input is an adjacency dict mapping vertex IDs to iterables of neighbours. The function returns either `None` (nonplanar) or a dict mapping each vertex to a list of its neighbours in clockwise order for a planar drawing.

## Why this library exists

Planarity testing is a fundamental graph algorithm with applications in circuit layout, graph drawing, and network design. The Boyer-Myrvold algorithm achieves linear-time planarity testing without the large constant factors of earlier methods. This implementation provides a readable, dependency-free version suitable for teaching and for embedding in larger projects that cannot pull in third-party packages.

A key trade-off is that the code favours clarity over micro-optimisation. It uses straightforward Python data structures rather than the bit-packed arrays of the original paper, so it may be slower on very large graphs but is much easier to audit and modify.

## Edge cases

- The graph may be disconnected; each component is tested independently.
- Isolated vertices (e.g., `{5: []}`) are supported and appear in the output with an empty adjacency list.
- Self-loops and parallel edges are rejected with `ValueError` because the algorithm assumes a simple graph.
- The input dict must contain a key for every vertex that appears in any adjacency list. A missing key raises `ValueError`.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Limitations

Values are coerced to floats, so very large integers lose precision. If you need
exact integer aggregates over a window, this is the wrong tool.

