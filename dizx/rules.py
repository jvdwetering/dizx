from . import Edge, CliffordPhase
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType


def check_local_complementation(g: BaseGraph[VT, ET], v: VT) -> bool:
    """
    Checks if the local complementation simplification can be applied.

    Note: this function assumes that the graph is graph-like.
    """

    return g.type(v) == VertexType.Z and g.phase(v).is_strictly_clifford() \
        and all([g.type(n) == VertexType.Z for n in g.neighbors(v)])


def local_complementation_about_vertex(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_local_complementation(g, v):
        return False

    a, z = g.phase(v).as_vector()
    z_inv = pow(z, -1, g.dim)
    ns = list(g.neighbors(v))

    for n in ns:
        e = g.edge_object(g.edge(v, n)).had
        g.add_to_phase(n, CliffordPhase(
            dim=g.dim,
            x=-z_inv * a * e,
            y=-z_inv * (e ** 2)
        ))

    for i, n in enumerate(ns):
        for m in ns[i + 1:]:
            e_n = g.edge_object(g.edge(v, n)).had
            e_m = g.edge_object(g.edge(v, m)).had
            g.add_edge(g.edge(n, m),
                       Edge.make(dim=g.dim, had=-z_inv * e_n * e_m))

    g.remove_vertex(v)
