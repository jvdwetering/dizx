from .graph.base import BaseGraph, VT, ET
from .utils import VertexType


def check_z_elim(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.vertex_degree(v) == 2 and g.phase(v).is_zero() and g.type(v) == VertexType.Z


def z_elim(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_z_elim(g, v):
        return False

    v1, v2 = tuple(g.neighbors(v))
    edge1 = g.edge_object(g.edge(v1, v))
    edge2 = g.edge_object(g.edge(v, v2))

    g.add_edge(
        g.edge(v1, v2), edgetype=EdgeType.SIMPLE
        if g.edge_object(g.edge(v, v1)) == g.edge_object(g.edge(v, v2))
        else EdgeType.HADAMARD
    )
    g.remove_vertex(v)

    return True
