from . import Edge, CliffordPhase
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType


def check_remove_parallel_edge_between_zs(
        g: BaseGraph[VT, ET], z1: VT, z2: VT) -> bool:
    return not (g.edge_object(g.edge(z1, z2)).is_reduced())\
        and g.type(z1) == VertexType.Z and g.type(z2) == VertexType.Z


def remove_parallel_edge_between_zs(
        g: BaseGraph[VT, ET], z1: VT, z2: VT) -> bool:
    if not check_remove_parallel_edge_between_zs(g, z1, z2):
        return False
    e = g.edge(z1, z2)
    eo = g.edge_object(e)
    g.add_to_phase(z1, CliffordPhase(dim=g.dim, y=2 * eo.had))
    g.set_edge_object(e, Edge(simple=1))
    return z_fuse(g, z1, z2)


def check_remove_self_loop_on_z(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.edge_object(g.edge(v, v)).is_edge_present()\
        and g.type(v) == VertexType.Z


def remove_self_loop_on_z(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_remove_self_loop_on_z(g, v):
        return False
    e = g.edge_object(g.edge(v, v))
    if e.is_simple_edge():
        pass  # We don't need to do anything
    if e.is_had_edge():
        g.add_to_phase(v, CliffordPhase(dim=g.dim, y=2 * e.had))
    g.remove_edge(g.edge(v, v))
    return True


def check_x_color_change(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) == VertexType.X


def _add_empty_vertex_between(
        g: BaseGraph[VT, ET], v: VT, w: VT,
        edge_to_v: Edge, edge_to_w: Edge) -> VT:
    new = g.add_vertex(
        VertexType.Z,
        qubit=(g.qubit(w) + g.qubit(v)) / 2 or g.qubit(w),
        row=(g.row(w) + g.row(v)) / 2 or g.row(w)
    )
    g.add_edge(g.edge(v, new), edge_to_v)
    g.add_edge(g.edge(new, w), edge_to_w)
    return new


def _set_empty_vertex_between(
        g: BaseGraph[VT, ET], v: VT, w: VT,
        edge_to_v: Edge, edge_to_w: Edge) -> VT:
    g.remove_edge(g.edge(v, w))
    return _add_empty_vertex_between(g, v, w, edge_to_v, edge_to_w)


def _set_edge_x_color_change(g: BaseGraph[VT, ET], v: VT, neigh: VT) -> None:
    et = g.edge(v, neigh)
    e = g.edge_object(et)
    if e.simple != 0 and e.had != 0:
        raise ValueError(f"The edge need to be normalised between vertex {v} "
                         f"and {neigh}.")
    neigh_type = g.type(neigh)
    if e.is_had_edge() and (
            neigh_type == VertexType.Z or neigh_type == VertexType.BOUNDARY):
        g.remove_edge(et)
        _add_empty_vertex_between(g, v, neigh, Edge(1), Edge(1))
    elif e.is_had_edge() and neigh_type == VertexType.X:
        g.set_edge_object(et, Edge.make(g.dim, had=0, simple=-e.had))
    elif e.is_simple_edge():
        g.set_edge_object(et, Edge(had=e.simple, simple=0))


def x_color_change(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_x_color_change(g, v):
        return False

    g.set_type(v, VertexType.Z)
    # Copy `neighbours` because it can change during the next for-loop
    current_neighbors = [n for n in g.neighbors(v)]
    for neigh in current_neighbors:
        _set_edge_x_color_change(g, v, neigh)

    return True


def check_z_fuse(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> bool:
    return v1 != v2 and g.connected(v1, v2)\
        and g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z\
        and g.edge_object(g.edge(v1, v2)).is_simple_edge()


def z_fuse(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> bool:
    if not check_z_fuse(g, v1, v2):
        return False
    g.add_to_phase(v1, g.phase(v2))
    for v3 in list(g.neighbors(v2)):
        if v2 == v3:
            g.add_edge(g.edge(v1, v1), g.edge_object(g.edge(v2, v2)))
            g.remove_edge(g.edge(v2, v2))
        elif v3 != v1:
            g.add_edge(g.edge(v1, v3), g.edge_object(g.edge(v2, v3)))
    g.remove_vertex(v2)
    return True


def check_z_elim(g: BaseGraph[VT, ET], v: VT) -> bool:
    if len(g.neighbors(v)) != 2:
        return False
    v1, v2 = tuple(g.neighbors(v))

    edge1 = g.edge_object(g.edge(v1, v))
    edge2 = g.edge_object(g.edge(v, v2))
    et1, et2 = edge1.type(), edge2.type()
    xb = (VertexType.X, VertexType.BOUNDARY)

    return g.vertex_degree(v) == 2\
        and g.phase(v).is_zero()\
        and g.type(v) == VertexType.Z\
        and (et1, et2) == (Edge.HadEdge, Edge.SimpleEdge)\
        or (et1, et2) == (Edge.SimpleEdge, Edge.HadEdge)\
        or (
                (et1, et2) == (Edge.HadEdge, Edge.HadEdge)
                and (edge1.had + edge2.had) % g.dim == 0
        ) or (
                (et1, et2) == (Edge.SimpleEdge, Edge.SimpleEdge)
                and (edge1.simple - pow(edge2.simple, -1, g.dim)) % g.dim == 0
                and g.type(v1) in xb and g.type(v2) in xb
        )


def z_elim(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_z_elim(g, v):
        return False

    v1, v2 = tuple(g.neighbors(v))
    edge1 = g.edge_object(g.edge(v1, v))
    edge2 = g.edge_object(g.edge(v, v2))
    et1, et2 = edge1.type(), edge2.type()
    xb = (VertexType.X, VertexType.BOUNDARY)

    if (et1, et2) == (Edge.HadEdge, Edge.SimpleEdge):
        g.add_edge(g.edge(v1, v2), Edge.make(
            g.dim, had=edge1.had * edge2.simple))
    elif (et1, et2) == (Edge.SimpleEdge, Edge.HadEdge):
        g.add_edge(g.edge(v1, v2), Edge.make(
            g.dim, had=edge1.simple * edge2.had))
    elif (et1, et2) == (Edge.HadEdge, Edge.HadEdge)\
            and (edge1.had + edge2.had) % g.dim == 0:
        g.add_edge(g.edge(v1, v2), Edge.make(g.dim, simple=1))
    elif (et1, et2) == (Edge.SimpleEdge, Edge.SimpleEdge)\
            and (edge1.simple - pow(edge2.simple, -1, g.dim)) % g.dim == 0\
            and g.type(v1) in xb and g.type(v2) in xb:
        g.add_edge(g.edge(v1, v2), Edge.make(g.dim, simple=1))
    else:
        return False

    g.remove_vertex(v)
    return True
