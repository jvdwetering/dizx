from . import Edge, CliffordPhase, Phase
from .basicrules import _add_empty_vertex_between, _set_empty_vertex_between
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType


def _check_pivoting_base(g: BaseGraph[VT, ET], v: VT, w: VT) -> bool:
    return v != w\
        and g.type(v) == VertexType.Z and g.type(w) == VertexType.Z\
        and g.phase(v).is_pauli() and g.phase(w).is_pauli()\
        and g.edge_object(g.edge(v, w)).is_had_edge()


def check_pivoting_simplification(g: BaseGraph[VT, ET], v: VT, w: VT) -> bool:
    """
    Checks if the pivoting simplification can be applied.

    Note: this function assumes that the graph is graph-like.
    """
    return _check_pivoting_base(g, v, w)\
        and all([g.type(n) == VertexType.Z for n in g.neighbors(v)])\
        and all([g.type(n) == VertexType.Z for n in g.neighbors(w)])


def check_boundary_pivot_simplification(
        g: BaseGraph[VT, ET], v: VT, b: VT) -> bool:
    """
    Checks if the boundary pivot simplification can be applied.

    Note: this function assumes that the graph is graph-like.

    Args:
        v: the vertex that only connects to Z spiders
        b: the vertex connected to a boundary
    """
    return _check_pivoting_base(g, v, b)\
        and all(g.type(n) == VertexType.Z for n in g.neighbors(v))\
        and all(g.type(n) == VertexType.Z for n in g.neighbors(b)
                if g.type(n) != VertexType.BOUNDARY)\
        and any(g.type(n) == VertexType.BOUNDARY for n in g.neighbors(b))


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
                had=-epsilon_inv * (e_1 * f_2 + e_2 * f_1)
            ))

    g.remove_vertex(v)
    g.remove_vertex(w)

    return True


def unfuse_phase(g: BaseGraph[VT, ET], v: VT) -> VT:
    new = g.add_vertex(
        VertexType.Z,
        qubit=g.qubit(v),
        row=g.row(v)
    )
    g.add_edge(g.edge(v, new), Edge(simple=1))
    g.set_phase(new, g.phase(v))
    g.set_phase(v, CliffordPhase(g.dim))
    return new


def graph_like_unfuse_phase(g: BaseGraph[VT, ET], v: VT) -> VT:
    new = unfuse_phase(g, v)
    _set_empty_vertex_between(g, v, new, Edge(had=1),
                              Edge.make(g.dim, had=-1))
    return new


def boundary_pivoting(g: BaseGraph[VT, ET], v: VT, w: VT) -> bool:
    """
    Applies the boundary pivot simplification.

    Note: this function assumes that the graph is graph-like.

    Args:
        v: the vertex that only connects to Z spiders
        w: the vertex that is connected to a boundary
    Returns:
        Weather the simplification was applied
    """
    if not check_boundary_pivot_simplification(g, v, w):
        return False

    graph_like_unfuse_phase(g, w)
    [b] = [n for n in g.neighbors(w) if g.type(n) == VertexType.BOUNDARY]

    bn = _set_empty_vertex_between(g, w, b, Edge(simple=1), Edge(simple=1))
    _set_empty_vertex_between(g, w, bn, Edge(had=1), Edge.make(g.dim, had=-1))

    return pivoting_simplification(g, v, w)


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

    return True
