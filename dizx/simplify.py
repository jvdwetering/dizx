import itertools

from . import Edge
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType
from .basicrules import x_color_change


def to_gh(g: BaseGraph[VT, ET]) -> None:
    """Turns every red node into a green node by changing regular edges into
    hadamard edges"""
    ty = g.types()
    # Copy `vertices` because it can change during the next for-loop
    vertices_copy = [v for v in g.vertices()]
    for v in vertices_copy:
        if ty[v] == VertexType.X:
            x_color_change(g, v)


def is_graph_like(g: BaseGraph):
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


# def to_graph_like(g):
#     """Checks if a ZX-diagram is graph-like"""
#
#     # turn all red spiders into green spiders
#     to_gh(g)
#
#     # simplify: remove excess HAD's, fuse along non-HAD edges, remove parallel edges and self-loops
#     spider_simp(g, quiet=True)
#
#     # ensure all I/O are connected to a Z-spider
#     bs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
#     for v in bs:
#
#         # if it's already connected to a Z-spider, continue on
#         if any([g.type(n) == VertexType.Z for n in g.neighbors(v)]):
#             continue
#
#         # have to connect the (boundary) vertex to a Z-spider
#         ns = list(g.neighbors(v))
#         for n in ns:
#             # every neighbor is another boundary or an H-Box
#             assert (g.type(n) in [VertexType.BOUNDARY, VertexType.H_BOX])
#             if g.type(n) == VertexType.BOUNDARY:
#                 z1 = g.add_vertex(ty=zx.VertexType.Z)
#                 z2 = g.add_vertex(ty=zx.VertexType.Z)
#                 z3 = g.add_vertex(ty=zx.VertexType.Z)
#                 g.remove_edge(g.edge(v, n))
#                 g.add_edge(g.edge(v, z1), edgetype=EdgeType.SIMPLE)
#                 g.add_edge(g.edge(z1, z2), edgetype=EdgeType.HADAMARD)
#                 g.add_edge(g.edge(z2, z3), edgetype=EdgeType.HADAMARD)
#                 g.add_edge(g.edge(z3, n), edgetype=EdgeType.SIMPLE)
#             else:  # g.type(n) == VertexType.H_BOX
#                 z = g.add_vertex(ty=zx.VertexType.Z)
#                 g.remove_edge(g.edge(v, n))
#                 g.add_edge(g.edge(v, z), edgetype=EdgeType.SIMPLE)
#                 g.add_edge(g.edge(z, n), edgetype=EdgeType.SIMPLE)
#
#     # each Z-spider can only be connected to at most 1 I/O
#     vs = list(g.vertices())
#     for v in vs:
#         if not g.type(v) == VertexType.Z:
#             continue
#         boundary_ns = [n for n in g.neighbors(v) if
#                        g.type(n) == VertexType.BOUNDARY]
#         if len(boundary_ns) <= 1:
#             continue
#
#         # add dummy spiders for all but one
#         for b in boundary_ns[:-1]:
#             z1 = g.add_vertex(ty=zx.VertexType.Z)
#             z2 = g.add_vertex(ty=zx.VertexType.Z)
#
#             g.remove_edge(g.edge(v, b))
#             g.add_edge(g.edge(z1, z2), edgetype=EdgeType.HADAMARD)
#             g.add_edge(g.edge(b, z1), edgetype=EdgeType.SIMPLE)
#             g.add_edge(g.edge(z2, v), edgetype=EdgeType.HADAMARD)
#
#     assert (is_graph_like(g))
