from . import Edge
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType


def check_x_color_change(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) == VertexType.X


def _set_edge_x_color_change(g: BaseGraph[VT, ET], v: VT, neigh: VT) -> None:
    et = g.edge(v, neigh)
    e = g.edge_object(et)
    if e.simple != 0 and e.had != 0:
        raise ValueError(f"The edge need to be normalised between vertex {v} "
                         f"and {neigh}.")
    neigh_type = type(neigh)
    if e.is_had_edge() and neigh_type == VertexType.Z:
        pass
    elif e.is_had_edge() and neigh_type == VertexType.X:
        pass
    elif e.is_had_edge() and neigh_type == VertexType.BOUNDARY:
        pass
    elif e.is_simple_edge() and (
            neigh_type == VertexType.Z or neigh_type == VertexType.X):
        g.set_edge_object(et, Edge(had=e.simple, simple=0))
    elif e.is_simple_edge() and neigh_type == VertexType.BOUNDARY:
        pass


def x_color_change(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_x_color_change(g, v):
        return False

    g.set_type(v, VertexType.Z)
    for neigh in g.neighbors(v):
        _set_edge_x_color_change(g, v, neigh)

    return True


def check_z_fuse(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> bool:
    return g.connected(v1, v2)\
        and g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z\
        and g.edge_object(g.edge(v1, v2)).is_simple_edge()


def z_fuse(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> bool:
    if not check_z_fuse(g, v1, v2):
        return False
    g.add_to_phase(v1, g.phase(v2))
    for v3 in g.neighbors(v2):
        if v3 != v1:
            g.add_edge(g.edge(v1, v3), g.edge_object(g.edge(v2, v3)))
    g.remove_vertex(v2)
    return True


def check_z_elim(g: BaseGraph[VT, ET], v: VT) -> bool:
    v1, v2 = tuple(g.neighbors(v))

    edge1 = g.edge_object(g.edge(v1, v))
    edge2 = g.edge_object(g.edge(v, v2))
    et1, et2 = edge1.type(), edge2.type()

    return g.vertex_degree(v) == 2\
        and g.phase(v).is_zero()\
        and g.type(v) == VertexType.Z\
        and (et1, et2) == (Edge.HadEdge, Edge.SimpleEdge)\
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
            and edge1.had - edge2.had % edge1.dim == 0:
        g.add_edge(g.edge(v1, v2), Edge(edge1.dim, 0, 1))
    elif (et1, et2) == (Edge.SimpleEdge, Edge.SimpleEdge)\
            and edge1.simple - pow(edge2.simple, -1, g.dim) % edge1.dim == 0\
            and g.type(v1) == VertexType.X and g.type(v2) == VertexType.X:
        g.add_edge(g.edge(v1, v2), Edge(edge1.dim, 0, 1))
    else:
        return False

    g.remove_vertex(v)
    return True
