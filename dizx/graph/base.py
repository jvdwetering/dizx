# DiZX - Python library for quantum circuit rewriting
#        and optimisation using the qudit ZX-calculus
# Copyright (C) 2023 - Boldizsar Poor, Lia Yeh and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from ..utils import VertexType, FloatInt
from .edge import Edge
from .phase import Phase

from .scalar import Scalar
from .phase import Phase, CliffordPhase
from .edge import Edge

from typing import TYPE_CHECKING, Union, Optional, Generic, TypeVar, Any, Sequence
from typing import List, Dict, Set, Tuple, Mapping, Iterable, Callable, ClassVar
from typing_extensions import Literal, GenericMeta # type: ignore # https://github.com/python/mypy/issues/5753


class DocstringMeta(GenericMeta):
    """Metaclass that allows docstring 'inheritance'."""

    def __new__(mcls, classname, bases, cls_dict, **kwargs):
        cls = GenericMeta.__new__(mcls, classname, bases, cls_dict, **kwargs)
        mro = cls.__mro__[1:]
        for name, member in cls_dict.items():
            if not getattr(member, '__doc__'):
                for base in mro:
                    try:
                        member.__doc__ = getattr(base, name).__doc__
                        break
                    except AttributeError:
                        pass
        return cls

def pack_indices(lst: List[FloatInt]) -> Mapping[FloatInt,int]:
    d: Dict[FloatInt,int] = dict()
    if len(lst) == 0: return d
    list.sort(lst)
    i: int = 0
    x: Optional[FloatInt] = None
    for j in range(len(lst)):
        y = lst[j]
        if y != x:
            x = y
            d[y] = i
            i += 1
    return d


VT = TypeVar('VT', bound=int) # The type that is used for representing vertices (e.g. an integer)
ET = TypeVar('ET') # The type used for representing edges (e.g. a pair of integers)


class BaseGraph(Generic[VT, ET], metaclass=DocstringMeta):
    """Base class for letting graph backends interact with PyZX.
    For a backend to work with PyZX, there should be a class that implements
    all the methods of this class. For implementations of this class see 
    :class:`~pyzx.graph.graph_s.GraphS` or :class:`~pyzx.graph.graph_ig.GraphIG`."""

    backend = 'None'

    def __init__(self, dim) -> None:

        self.dim = dim  # The qudit dimension this Graph is working on

        self.scalar: Scalar = Scalar(dim)
        

    def __str__(self) -> str:
        return "Graph({} vertices, {} edges, dimension {})".format(
                str(self.num_vertices()),str(self.num_edges()),str(self.dim))

    def __repr__(self) -> str:
        return str(self)

    def copy(self, adjoint:bool=False, backend:Optional[str]=None) -> 'BaseGraph':
        """Create a copy of the graph. If ``adjoint`` is set, 
        the adjoint of the graph will be returned (inputs and outputs flipped, phases reversed).
        When ``backend`` is set, a copy of the graph with the given backend is produced. 
        By default the copy will have the same backend.

        Args:
            adjoint: set to True to make the copy be the adjoint of the graph
            backend: the backend of the output graph

        Returns:
            A copy of the graph

        Note:
            The copy will have consecutive vertex indices, even if the original
            graph did not.
        """
        from .graph import Graph # imported here to prevent circularity
        if (backend is None):
            backend = type(self).backend
        g = Graph(self.dim,backend = backend)
        g.scalar = self.scalar.copy()

        ty = self.types()
        ph = self.phases()
        qs = self.qubits()
        rs = self.rows()
        maxr = self.depth()
        vtab = dict()

        if adjoint:
            for i in ph: ph[i] = ph[i].adjoint()
        
        for v in self.vertices():
            i = g.add_vertex(ty[v],phase=ph[v])
            if v in qs: g.set_qubit(i,qs[v])
            if v in rs: 
                if adjoint: g.set_row(i, maxr-rs[v])
                else: g.set_row(i, rs[v])
            vtab[v] = i
        

        new_inputs = tuple(vtab[i] for i in self.inputs())
        new_outputs = tuple(vtab[i] for i in self.outputs())
        if not adjoint:
            g.set_inputs(new_inputs)
            g.set_outputs(new_outputs)
        else:
            g.set_inputs(new_outputs)
            g.set_outputs(new_inputs)
        
        etab = {e:g.edge(vtab[self.edge_s(e)],vtab[self.edge_t(e)]) for e in self.edges()}
        g.add_edges(etab.values())
        for e,f in etab.items():
            g.set_edge_object(f, self.edge_object(e))
        return g

    def adjoint(self) -> 'BaseGraph':
        """Returns a new graph equal to the adjoint of this graph."""
        return self.copy(adjoint=True)

    def clone(self) -> 'BaseGraph':
        """
        This method should return an identical copy of the graph, without any relabeling

        Used in lookahead extraction.
        """
        return self.copy()


    # def to_tensor(self, preserve_scalar:bool=True) -> np.ndarray:
    #     """Returns a representation of the graph as a tensor using :func:`~pyzx.tensor.tensorfy`"""
    #     return tensorfy(self, preserve_scalar)
    # def to_matrix(self,preserve_scalar:bool=True) -> np.ndarray:
    #     """Returns a representation of the graph as a matrix using :func:`~pyzx.tensor.tensorfy`"""
    #     return tensor_to_matrix(tensorfy(self, preserve_scalar), self.num_inputs(), self.num_outputs())


    # def to_tikz(self,draw_scalar:bool=False) -> str:
    #     """Returns a Tikz representation of the graph."""
    #     from ..tikz import to_tikz
    #     return to_tikz(self,draw_scalar)


    def vindex(self) -> VT:
        """The index given to the next vertex added to the graph. It should always
        be equal to ``max(g.vertices()) + 1``."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def depth(self) -> FloatInt:
        """Returns the value of the highest row number given to a vertex.
        This is -1 when no rows have been set."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def pack_circuit_rows(self) -> None:
        """Compresses the rows of the graph so that every index is used."""
        rows = [self.row(v) for v in self.vertices()]
        new_rows = pack_indices(rows)
        for v in self.vertices():
            self.set_row(v, new_rows[self.row(v)])

    def qubit_count(self) -> int:
        """Returns the number of inputs of the graph"""
        return self.num_inputs()

    def normalize(self) -> None:
        """Puts every node connecting to an input/output at the correct qubit index and row."""
        if self.num_inputs() == 0:
            self.auto_detect_io()
        max_r = self.depth() - 1
        if max_r <= 2: 
            for o in self.outputs():
                self.set_row(o,4)
            max_r = self.depth() -1
        claimed = []
        for q,i in enumerate(sorted(self.inputs(), key=self.qubit)):
            self.set_row(i,0)
            self.set_qubit(i,q)
            #q = self.qubit(i)
            n = list(self.neighbors(i))[0]
            if self.type(n) in (VertexType.Z, VertexType.X):
                claimed.append(n)
                self.set_row(n,1)
                self.set_qubit(n, q)
            # else: #directly connected to output
            #     e = self.edge(i, n)
            #     t = self.edge_type(e)
            #     self.remove_edge(e)
            #     v = self.add_vertex(VertexType.Z,q,1)
            #     self.add_edge(self.edge(i,v),toggle_edge(t))
            #     self.add_edge(self.edge(v,n),EdgeType.HADAMARD)
            #     claimed.append(v)
        for q, o in enumerate(sorted(self.outputs(),key=self.qubit)):
            #q = self.qubit(o)
            self.set_row(o,max_r+1)
            self.set_qubit(o,q)
            n = list(self.neighbors(o))[0]
            if n not in claimed:
                self.set_row(n,max_r)
                self.set_qubit(n, q)
            # else:
            #     e = self.edge(o, n)
            #     t = self.edge_type(e)
            #     self.remove_edge(e)
            #     v = self.add_vertex(VertexType.Z,q,max_r)
            #     self.add_edge(self.edge(o,v),toggle_edge(t))
            #     self.add_edge(self.edge(v,n),EdgeType.HADAMARD)

        self.pack_circuit_rows()

    def inputs(self) -> Tuple[VT, ...]:
        """Gets the inputs of the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def set_inputs(self, inputs: Tuple[VT, ...]):
        """Sets the inputs of the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def num_inputs(self) -> int:
        """Gets the number of inputs of the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def outputs(self) -> Tuple[VT, ...]:
        """Gets the outputs of the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def set_outputs(self, outputs: Tuple[VT, ...]):
        """Sets the outputs of the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def num_outputs(self) -> int:
        """Gets the number of outputs of the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def add_vertices(self, amount: int) -> List[VT]:
        """Add the given amount of vertices, and return the indices of the
        new vertices added to the graph, namely: range(g.vindex() - amount, g.vindex())"""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def add_vertex(self, 
                   ty:VertexType.Type=VertexType.BOUNDARY, 
                   qubit:FloatInt=-1, 
                   row:FloatInt=-1, 
                   phase:Optional[Phase]=None,
                   ) -> VT:
        """Add a single vertex to the graph and return its index.
        The optional parameters allow you to respectively set
        the type, qubit index, row index and phase of the vertex."""
        v = self.add_vertices(1)[0]
        self.set_type(v, ty)
        if phase is None:
            phase = CliffordPhase(self.dim)
        self.set_qubit(v, qubit)
        self.set_row(v, row)
        if phase: 
            self.set_phase(v, phase)
        return v

    def add_edges(self, edges: Iterable[ET], edge_object: Edge) -> None:
        """Adds a list of edges to the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def add_edge(self, edge: ET, edge_object: Edge) -> None:
        """Adds a single edge of the given type"""
        self.add_edges([edge], edge_object)


    def remove_vertices(self, vertices: Iterable[VT]) -> None:
        """Removes the list of vertices from the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def remove_vertex(self, vertex: VT) -> None:
        """Removes the given vertex from the graph."""
        self.remove_vertices([vertex])

    def remove_isolated_vertices(self) -> None:
        """Deletes all vertices and vertex pairs that are not connected to any other vertex."""
        rem: List[VT] = []
        for v in self.vertices():
            d = self.vertex_degree(v)
            if d == 0:
                rem.append(v)
                ty = self.type(v)
                if ty == VertexType.BOUNDARY:
                    raise TypeError("Diagram is not a well-typed ZX-diagram: contains isolated boundary vertex.")
                else: self.scalar.add_node(self.phase(v))
            if d == 1: # It has a unique neighbor
                if v in rem: continue # Already taken care of
                if self.type(v) == VertexType.BOUNDARY: continue # Ignore in/outputs
                w = list(self.neighbors(v))[0]
                if len(list(self.neighbors(w))) > 1: continue # But this neighbor has other neighbors
                if self.type(w) == VertexType.BOUNDARY: continue # It's a state/effect
                # At this point w and v are only connected to each other
                rem.append(v)
                rem.append(w)
                et = self.edge_object(self.edge(v, w))
                t1 = self.type(v)
                t2 = self.type(w)
                ## TODO: Actually implement correct scalars
                # if t1==t2:
                #     if et == EdgeType.SIMPLE:
                #         self.scalar.add_node(self.phase(v)+self.phase(w))
                #     else:
                #         self.scalar.add_spider_pair(self.phase(v), self.phase(w))
                # else:
                #     if et == EdgeType.SIMPLE:
                #         self.scalar.add_spider_pair(self.phase(v), self.phase(w))
                #     else:
                #         self.scalar.add_node(self.phase(v)+self.phase(w))
        self.remove_vertices(rem)

    def remove_edges(self, edges: List[ET]) -> None:
        """Removes the list of edges from the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def remove_edge(self, edge: ET) -> None:
        """Removes the given edge from the graph."""
        self.remove_edges([edge])

    def num_vertices(self) -> int:
        """Returns the amount of vertices in the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def num_edges(self) -> int:
        """Returns the amount of edges in the graph"""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def vertices(self) -> Sequence[VT]:
        """Iterator over all the vertices."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def edges(self) -> Sequence[ET]:
        """Iterator that returns all the edges. Output type depends on implementation in backend."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def vertex_set(self) -> Set[VT]:
        """Returns the vertices of the graph as a Python set. 
        Should be overloaded if the backend supplies a cheaper version than this."""
        return set(self.vertices())

    def edge_set(self) -> Set[ET]:
        """Returns the edges of the graph as a Python set. 
        Should be overloaded if the backend supplies a cheaper version than this."""
        return set(self.edges())

    def edge(self, s:VT, t:VT) -> ET:
        """Returns the edge object with the given source/target."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def edge_st(self, edge: ET) -> Tuple[VT, VT]:
        """Returns a tuple of source/target of the given edge."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def edge_s(self, edge: ET) -> VT:
        """Returns the source of the given edge."""
        return self.edge_st(edge)[0]
    def edge_t(self, edge: ET) -> VT:
        """Returns the target of the given edge."""
        return self.edge_st(edge)[1]

    def neighbors(self, vertex: VT) -> Sequence[VT]:
        """Returns all neighboring vertices of the given vertex."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def vertex_degree(self, vertex: VT) -> int:
        """Returns the degree of the given vertex."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def incident_edges(self, vertex: VT) -> Sequence[ET]:
        """Returns all neighboring edges of the given vertex."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def connected(self,v1: VT,v2: VT) -> bool:
        """Returns whether vertices v1 and v2 share an edge."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def edge_object(self, e: ET) -> Edge:
        """Returns the type of the given edge:
        ``EdgeType.SIMPLE`` if it is regular, ``EdgeType.HADAMARD`` if it is a Hadamard edge,
        0 if the edge is not in the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def set_edge_object(self, e: ET, t: Edge) -> None:
        """Sets the type of the given edge."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def type(self, vertex: VT) -> VertexType.Type:
        """Returns the type of the given vertex:
        VertexType.BOUNDARY if it is a boundary, VertexType.Z if it is a Z node,
        VertexType.X if it is a X node, VertexType.H_BOX if it is an H-box."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def types(self) -> Mapping[VT, VertexType.Type]:
        """Returns a mapping of vertices to their types."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def set_type(self, vertex: VT, t: VertexType.Type) -> None:
        """Sets the type of the given vertex to t."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)

    def phase(self, vertex: VT) -> Phase:
        """Returns the phase value of the given vertex."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def phases(self) -> Mapping[VT, Phase]:
        """Returns a mapping of vertices to their phase values."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def set_phase(self, vertex: VT, phase: Phase) -> None:
        """Sets the phase of the vertex to the given value."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def add_to_phase(self, vertex: VT, phase: Phase) -> None:
        """Add the given phase to the phase value of the given vertex."""
        self.set_phase(vertex,self.phase(vertex)+phase)

    def qubit(self, vertex: VT) -> FloatInt:
        """Returns the qubit index associated to the vertex. 
        If no index has been set, returns -1."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def qubits(self) -> Mapping[VT,FloatInt]:
        """Returns a mapping of vertices to their qubit index."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def set_qubit(self, vertex: VT, q: FloatInt) -> None:
        """Sets the qubit index associated to the vertex."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)

    def row(self, vertex: VT) -> FloatInt:
        """Returns the row that the vertex is positioned at. 
        If no row has been set, returns -1."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def rows(self) -> Mapping[VT, FloatInt]:
        """Returns a mapping of vertices to their row index."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def set_row(self, vertex: VT, r: FloatInt) -> None:
        """Sets the row the vertex should be positioned at."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)

    def set_position(self, vertex: VT, q: FloatInt, r: FloatInt):
        """Set both the qubit index and row index of the vertex."""
        self.set_qubit(vertex, q)
        self.set_row(vertex, r)
