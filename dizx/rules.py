from . import Edge, CliffordPhase
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType


def check_pivoting_simplification(g: BaseGraph[VT, ET], v: VT, w: VT) -> bool:
    """
    Checks if the pivoting simplification can be applied.

    Note: this function assumes that the graph is graph-like.
    """
    return g.type(v) == VertexType.Z and g.type(w) == VertexType.Z\
        and g.phase(v).is_pauli() and g.phase(w).is_pauli()\
        and g.edge_object(g.edge(v, w)).is_had_edge()\
        and all([g.type(n) == VertexType.Z for n in g.neighbors(v)])\
        and all([g.type(n) == VertexType.Z for n in g.neighbors(w)])


def pivoting_simplification(g: BaseGraph[VT, ET], v: VT, w: VT) -> bool:
    if not check_pivoting_simplification(g, v, w):
        return False
    vp, wp = g.phase(v), g.phase(w)
    if not isinstance(vp, CliffordPhase) or not isinstance(wp, CliffordPhase):
        raise ValueError("The implementation of pivoting is only supported "
                         "with CliffordPhase phases.")

    epsilon = g.edge_object(g.edge(v, w)).had
    epsilon_inv = pow(epsilon, -1, g.dim)
    _ns = set(g.neighbors(v)).union(g.neighbors(w))
    _ns.remove(v)
    _ns.remove(w)
    ns = list(_ns)
    for n in ns:
        e = g.edge_object(g.edge(v, n)).had
        f = g.edge_object(g.edge(w, n)).had
        g.add_to_phase(n, CliffordPhase(
            dim=g.dim,
            x=-epsilon_inv * (vp.x * f + vp.y * e),
            y=-2 * epsilon_inv * f * e
        ))

    for i, n in enumerate(ns):
        for m in ns[i + 1:]:
            e_1 = g.edge_object(g.edge(v, n)).had
            e_2 = g.edge_object(g.edge(v, m)).had
            f_1 = g.edge_object(g.edge(w, n)).had
            f_2 = g.edge_object(g.edge(w, m)).had
            g.add_edge(g.edge(n, m), Edge.make(
                dim=g.dim,
                had=-epsilon_inv*(e_1 * f_2 + e_2 * f_1)
            ))

    g.remove_vertex(v)
    g.remove_vertex(w)

    def check_local_complementation_simplification(
            g: BaseGraph[VT, ET], v: VT) -> bool:
        """
        Checks if the local complementation simplification can be applied.

        Note: this function assumes that the graph is graph-like.
        """

        return g.type(v) == VertexType.Z and g.phase(v).is_strictly_clifford()\
            and all([g.type(n) == VertexType.Z for n in g.neighbors(v)])

    def local_complementation_simplification(
            g: BaseGraph[VT, ET], v: VT) -> bool:
        if not check_local_complementation_simplification(g, v):
            return False
        vp = g.phase(v)
        if not isinstance(vp, CliffordPhase):
            raise ValueError(
                "The implementation of local complementation is only "
                "supported with CliffordPhase phases.")

        z_inv = pow(vp.y, -1, g.dim)
        ns = list(g.neighbors(v))

        for n in ns:
            e = g.edge_object(g.edge(v, n)).had
            g.add_to_phase(n, CliffordPhase(
                dim=g.dim,
                x=-z_inv * vp.x * e,
                y=-z_inv * (e ** 2)
            ))

        for i, n in enumerate(ns):
            for m in ns[i + 1:]:
                e_n = g.edge_object(g.edge(v, n)).had
                e_m = g.edge_object(g.edge(v, m)).had
                g.add_edge(g.edge(n, m),
                           Edge.make(dim=g.dim, had=-z_inv * e_n * e_m))

        g.remove_vertex(v)
