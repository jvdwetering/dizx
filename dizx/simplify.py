import itertools
from typing import Optional

from . import Edge
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType
from .basicrules import x_color_change, _add_vertex_between


def to_gh(g: BaseGraph[VT, ET]) -> None:
    """Turns every red node into a green node by changing regular edges into
    hadamard edges"""
    ty = g.types()
    # Copy `vertices` because it can change during the next for-loop
    vertices_copy = [v for v in g.vertices()]
    for v in vertices_copy:
        if ty[v] == VertexType.X:
            x_color_change(g, v)


def is_graph_like(g: BaseGraph[VT, ET]) -> bool:
    """Puts a ZX-diagram in graph-like form"""

    # checks that all spiders are Z-spiders
    for v in g.vertices():
        if g.type(v) not in [VertexType.Z, VertexType.BOUNDARY]:
            return False

    for v1, v2 in itertools.combinations(g.vertices(), 2):
        if not g.connected(v1, v2):
            continue

        # Z-spiders are only connected via Hadamard edges
        if g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z\
                and g.edge_object(g.edge(v1, v2)).is_simple_edge():
            return False

    # no self-loops
    for v in g.vertices():
        if g.connected(v, v):
            return False

    # every I/O is connected to a Z-spider
    bs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    for b in bs:
        if g.vertex_degree(b) != 1 or\
                g.type(list(g.neighbors(b))[0]) != VertexType.Z:
            return False

    # every Z-spider is connected to at most one I/O
    zs = [v for v in g.vertices() if g.type(v) == VertexType.Z]
    for z in zs:
        b_neighbors = [n for n in g.neighbors(z)
                       if g.type(n) == VertexType.BOUNDARY]
        if len(b_neighbors) > 1:
            return False

    return True


def to_graph_like(g: BaseGraph[VT, ET]) -> None:
    """Checks if a ZX-diagram is graph-like"""

    # turn all red spiders into green spiders
    to_gh(g)

    # simplify: remove excess HAD's, fuse along non-HAD edges, remove parallel edges and self-loops
    spider_simp(g, quiet=True)

    # ensure all I/O are connected to a Z-spider
    bs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    for v in bs:
        # have to connect the (boundary) vertex to a Z-spider
        ns = list(g.neighbors(v))
        if len(ns) == 1 and g.type(ns[0]) == VertexType.BOUNDARY:
            _add_vertices_before_boundary(g, v, ns[0])

    # each Z-spider can only be connected to at most 1 I/O
    unfuse_multi_boundary_connections(g)

    # make drawings nice
    g.ensure_enough_distance()

    assert is_graph_like(g)


def unfuse_multi_boundary_connections(g: BaseGraph[VT, ET]) -> None:
    zs = [v for v in g.vertices() if g.type(v) == VertexType.Z]
    for v in zs:
        boundary_ns = [n for n in g.neighbors(v) if
                       g.type(n) == VertexType.BOUNDARY]
        if len(boundary_ns) <= 1:
            continue

        # add dummy spiders for all but one
        for b in boundary_ns[:-1]:
            _add_vertices_before_boundary(g, b, v)


def _add_vertices_before_boundary(g: BaseGraph[VT, ET], v: VT, w: VT) -> None:
    e = g.edge_object(g.edge(w, v))
    g.remove_edge(g.edge(w, v))
    n = (g.type(v) == VertexType.BOUNDARY) + (g.type(w) == VertexType.BOUNDARY)
    assert n > 0
    new_z_1 = _add_z_neighbour_if_boundary(g, v, w, n)
    new_z_2 = _add_z_neighbour_if_boundary(g, w, v, n)
    # n1 can be 0, so use `is None`!
    z_1: VT = v if new_z_1 is None else new_z_1
    z_2: VT = w if new_z_2 is None else new_z_2

    if e.is_simple_edge():
        n = _add_vertex_between(g, VertexType.Z, z_1, z_2,
                                Edge(had=(-1) % g.dim), Edge(had=1))
    else:  # e.is_had_edge():
        g.add_edge(g.edge(z_1, z_2), e)


def _add_z_neighbour_if_boundary(
        g: BaseGraph[VT, ET], b: VT, w: VT, n: int
) -> Optional[VT]:
    if g.type(b) == VertexType.BOUNDARY:
        new = g.add_vertex(
            VertexType.Z,
            qubit=((1 + n) * g.qubit(b) + g.qubit(w)) / (2 + n) or g.qubit(b),
            row=((1 + n) * g.row(b) + g.row(w)) / (2 + n) or g.row(b)
        )
        g.add_edge(g.edge(b, new), Edge(simple=1))
        return new
    return None
