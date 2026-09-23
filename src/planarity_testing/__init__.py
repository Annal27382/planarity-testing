"""Planarity testing for undirected graphs.

This package provides a single public function, :func:`planar_embedding`,
which decides whether an undirected graph is planar and, when it is,
returns a planar embedding.
"""

from .core import planar_embedding

__all__ = ["planar_embedding"]
