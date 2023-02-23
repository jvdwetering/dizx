from . import Edge
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType


def check_z_elim(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.vertex_degree(v) == 2\
        and g.phase(v).is_zero()\
        and g.type(v) == VertexType.Z


def check_z_elim_edges(g: BaseGraph[VT, ET], v: VT) -> bool:
    v1, v2 = tuple(g.neighbors(v))

    edge1 = g.edge_object(g.edge(v1, v))
    edge2 = g.edge_object(g.edge(v, v2))
    et1, et2 = edge1.type(), edge2.type()

    assert edge1.dim == edge2.dim

    return (et1, et2) == (Edge.HadEdge, Edge.SimpleEdge)\
        or (et1, et2) == (Edge.SimpleEdge, Edge.HadEdge)\
        or (
                (et1, et2) == (Edge.HadEdge, Edge.HadEdge)
                and edge1.had == -edge2.had
        ) or (
                (et1, et2) == (Edge.SimpleEdge, Edge.SimpleEdge)
                and edge1.simple == pow(edge2.simple, -1, g.dim)
                and g.type(v1) == VertexType.X and g.type(v2) == VertexType.X
        )


def z_elim(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_z_elim(g, v):
        return False

    v1, v2 = tuple(g.neighbors(v))
    edge1 = g.edge_object(g.edge(v1, v))
    edge2 = g.edge_object(g.edge(v, v2))
    et1, et2 = edge1.type(), edge2.type()

    if (et1, et2) == (Edge.HadEdge, Edge.SimpleEdge):
        g.add_edge(g.edge(v1, v2), Edge(edge1.dim, edge1.had * edge2.simple))
    elif (et1, et2) == (Edge.SimpleEdge, Edge.HadEdge):
        g.add_edge(g.edge(v1, v2), Edge(edge1.dim, edge1.simple * edge2.had))
    elif (et1, et2) == (Edge.HadEdge, Edge.HadEdge)\
            and edge1.had == -edge2.had:
        g.add_edge(g.edge(v1, v2), Edge(edge1.dim, 0, 1))
    elif (et1, et2) == (Edge.SimpleEdge, Edge.SimpleEdge)\
            and edge1.simple == pow(edge2.simple, -1, g.dim)\
            and g.type(v1) == VertexType.X and g.type(v2) == VertexType.X:
        g.add_edge(g.edge(v1, v2), Edge(edge1.dim, 0, 1))
    else:
        return False

    g.remove_vertex(v)
    return True
