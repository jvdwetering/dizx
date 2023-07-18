import itertools
from typing import Optional

from . import Edge
from .graph.base import BaseGraph, VT, ET
from .rules import local_complementation_simplification,\
    pivoting_simplification, boundary_pivoting
from .utils import VertexType
from .basicrules import x_color_change, _add_empty_vertex_between, z_fuse,\
    remove_self_loop_on_z, remove_parallel_edge_between_zs


def to_gh(g: BaseGraph[VT, ET]) -> None:
    """Turns every red node into a green node by changing regular edges into
    hadamard edges"""
    ty = g.types()
    # Copy `vertices` because it can change during the next for-loop
    vertices_copy = [v for v in g.vertices()]
    for v in vertices_copy:
        if ty[v] == VertexType.X:
            x_color_change(g, v)


def has_self_loop(g: BaseGraph[VT, ET]) -> bool:
    return any(g.connected(v, v) for v in g.vertices())


def is_gh(g: BaseGraph[VT, ET]) -> bool:
    """Check if a graph has only Z spiders that are connected via H-edges"""
    for v in g.vertices():
        if g.type(v) not in [VertexType.Z, VertexType.BOUNDARY]:
            return False

    for v1, v2 in itertools.combinations(g.vertices(), 2):
        if not g.connected(v1, v2):
            continue

        if g.type(v1) != VertexType.BOUNDARY and g.type(
                v2) != VertexType.BOUNDARY\
                and g.edge_object(g.edge(v1, v2)).is_simple_edge():
            return False

    return True


def io_connections_are_graph_like(g: BaseGraph[VT, ET]) -> bool:
    bs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    for b in bs:
        [z] = list(g.neighbors(b))
        b_neighbors =\
            [n for n in g.neighbors(z) if g.type(n) == VertexType.BOUNDARY]
        if len(b_neighbors) > 1:
            return False
    return True


def z_spiders_connected_to_single_io(g: BaseGraph[VT, ET]) -> bool:
    zs = [v for v in g.vertices() if g.type(v) == VertexType.Z]
    for z in zs:
        b_neighbors =\
            [n for n in g.neighbors(z) if g.type(n) == VertexType.BOUNDARY]
        if len(b_neighbors) > 1:
            return False


def is_graph_like(g: BaseGraph[VT, ET]) -> bool:
    """Checks if a ZX-diagram is graph-like"""
    return is_gh(g)\
        and not (has_self_loop(g))\
        and io_connections_are_graph_like(g)


def to_graph_like(g: BaseGraph[VT, ET]) -> None:
    """Puts a ZX-diagram in graph-like form"""

    # turn all red spiders into green spiders
    to_gh(g)

    # simplify: fuse along non-HAD edges
    fuse_along_simple_edges(g)

    #  remove self-loops
    for v in g.vertices():
        # if there's no self-loop, the function does nothing
        remove_self_loop_on_z(g, v)

    #  remove parallel edges
    for e in g.edges():
        # if there are no parallel edges, the function does nothing
        remove_parallel_edge_between_zs(g, *e)

    # each Z-spider can only be connected to at most 1 I/O
    unfuse_multi_boundary_connections(g)

    # ensure all I/O are connected to a Z-spider
    bs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    for v in bs:
        # have to connect the (boundary) vertex to a Z-spider
        [n] = list(g.neighbors(v))
        if g.edge_object(g.edge(v, n)).is_had_edge()\
                or g.type(n) == VertexType.BOUNDARY:
            _add_vertices_before_boundary(g, v, n)

    # make drawings nice
    g.ensure_enough_distance()

    assert is_graph_like(g)


def fuse_along_simple_edges(g):
    zs = [v for v in g.vertices() if g.type(v) == VertexType.Z]
    fused_zs = []
    for z in zs:
        if z in fused_zs:
            continue
        ns = [v for v in g.neighbors(z) if g.type(v) == VertexType.Z]
        fused_zs += [
            n for n in ns if
            z_fuse(g, z, n)
        ]


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
        _add_empty_vertex_between(g, z_1, z_2, Edge(had=(-1) % g.dim),
                                  Edge(had=1))
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


def simplify_lc(g: BaseGraph[VT, ET]) -> bool:
    for v in g.vertices():
        if local_complementation_simplification(g, v):
            simplify_lc(g)
            return True
    return False


def simplify_pivot(g: BaseGraph[VT, ET]):
    vertices = list(g.vertices())
    for i, v in enumerate(vertices):
        for w in vertices[i + 1:]:
            if pivoting_simplification(g, v, w):
                simplify_pivot(g)
                return True
    return False


def simplify_boundary_pivot(g: BaseGraph[VT, ET]):
    vertices = list(g.vertices())
    for i, v in enumerate(vertices):
        for w in vertices[i + 1:]:
            if boundary_pivoting(g, v, w) or boundary_pivoting(g, v, w):
                simplify_boundary_pivot(g)
                return True
    return False


def _internal_spiders(g: BaseGraph[VT, ET]) -> list[VT]:
    return [
        v for v in g.vertices() if
        g.type(v) == VertexType.Z and not [n for n in g.neighbors(v) if
                                           g.type(n) == VertexType.BOUNDARY]
    ]


def _boundary_spiders(g: BaseGraph[VT, ET]) -> list[VT]:
    return [
        v for v in g.vertices() if
        g.type(v) == VertexType.Z and [n for n in g.neighbors(v) if
                                       g.type(n) == VertexType.BOUNDARY]
    ]


def is_in_ap_form(g: BaseGraph[VT, ET]) -> bool:
    internals = set(_internal_spiders(g))
    return is_graph_like(g)\
        and all(g.phase(v).is_pauli() for v in internals)\
        and not any(
            set(g.neighbors(v)).intersection(internals)
            for v in internals
        )


def to_ap_form(g: BaseGraph[VT, ET]) -> None:
    to_graph_like(g)
    changed = True
    while changed:
        changed = simplify_lc(g) or simplify_pivot(g)

    # make drawings nice
    g.ensure_enough_distance()


def clifford_simp(g: BaseGraph[VT, ET]) -> None:
    to_graph_like(g)

    c1 = True
    while c1:
        c2 = True
        while c2:
            c2 = simplify_lc(g) or simplify_pivot(g)
        c1 = simplify_boundary_pivot(g)

    # make drawings nice
    g.ensure_enough_distance()


def is_in_gslc_form(g: BaseGraph[VT, ET]) -> None:
    pass