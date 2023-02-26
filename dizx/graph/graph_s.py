# PyZX - Python library for quantum circuit rewriting 
#       and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import BaseGraph

from ..utils import VertexType, FloatInt

from .edge import Edge
from .phase import CliffordPhase

from typing import Tuple, Dict, List

from typing import Tuple, Dict

class GraphS(BaseGraph[int,Tuple[int,int]]):
    """Purely Pythonic implementation of :class:`~graph.base.BaseGraph`."""
    backend = 'simple'

    #The documentation of what these methods do 
    #can be found in base.BaseGraph
    def __init__(self, dim) -> None:
        BaseGraph.__init__(self,dim)
        self.graph: Dict[int,Dict[int,Edge]]            = dict()
        self._vindex: int                               = 0
        self.nedges: int                                = 0
        self.ty: Dict[int,VertexType.Type]              = dict()
        self._phase: Dict[int, Phase]                   = dict()
        self._qindex: Dict[int, FloatInt]               = dict()
        self._maxq: FloatInt                            = -1
        self._rindex: Dict[int, FloatInt]               = dict()
        self._maxr: FloatInt                            = -1

        self._inputs: Tuple[int, ...]                   = tuple()
        self._outputs: Tuple[int, ...]                  = tuple()
        
    def clone(self) -> 'GraphS':
        cpy = GraphS()
        for v, d in self.graph.items():
            cpy.graph[v] = d.copy()
        cpy._vindex = self._vindex
        cpy.nedges = self.nedges
        cpy.ty = self.ty.copy()
        cpy._phase = self._phase.copy()
        cpy._qindex = self._qindex.copy()
        cpy._maxq = self._maxq
        cpy._rindex = self._rindex.copy()
        cpy._maxr = self._maxr
        cpy._vdata = self._vdata.copy()
        cpy.scalar = self.scalar.copy()
        cpy._inputs = tuple(list(self._inputs))
        cpy._outputs = tuple(list(self._outputs))
        cpy.track_phases = self.track_phases
        cpy.phase_index = self.phase_index.copy()
        cpy.phase_master = self.phase_master
        cpy.phase_mult = self.phase_mult.copy()
        cpy.max_phase_index = self.max_phase_index
        return cpy

    def vindex(self): return self._vindex
    def depth(self): 
        if self._rindex: self._maxr = max(self._rindex.values())
        else: self._maxr = -1
        return self._maxr
    def qubit_count(self): 
        if self._qindex: self._maxq = max(self._qindex.values())
        else: self._maxq = -1
        return self._maxq + 1

    def inputs(self):
        return self._inputs

    def num_inputs(self):
        return len(self._inputs)

    def set_inputs(self, inputs):
        self._inputs = inputs

    def outputs(self):
        return self._outputs

    def num_outputs(self):
        return len(self._outputs)

    def set_outputs(self, outputs):
        self._outputs = outputs

    def add_vertices(self, amount):
        for i in range(self._vindex, self._vindex + amount):
            self.graph[i] = dict()
            self.ty[i] = VertexType.BOUNDARY
            self._phase[i] = CliffordPhase(self.dim)
        self._vindex += amount
        return range(self._vindex - amount, self._vindex)


    def add_edges(self, edges:List[Tuple[int,int]],eo: Edge):
        for e in edges:
            # self.nedges += 1
            # self.graph[s][t] = edgetype
            # self.graph[t][s] = edgetype
            self.add_edge(e,eo)

    def add_edge(self, e:Tuple[int,int], eo: Edge):
        v1,v2 = e
        t1,t2 = self.type(v1), self.type(v2)
        old = self.edge_object(e)
        if t1 == VertexType.BOUNDARY or t2 == VertexType.BOUNDARY:
            if old:  # There was already an edge present
                raise ValueError("Trying to add an edge to a boundary while there is already an edge present")
            if not eo.is_single():
                raise ValueError("Can't add compound edge to boundary vertex")
            self.graph[v1][v2] = eo
            self.graph[v2][v1] = eo
            self.nedges += 1
            return
        if t1 == t2 and t1 == VertexType.Z:  # Both spiders are Z-spiders
            if eo.simple != 0 or old.simple != 0: # We have some amount of simple edges, so the spiders 'fuse' and we can get rid of any H-edges
                h = (old.had + eo.had) % self.dim
                self.add_to_phase(v1,CliffordPhase(self.dim,x=0,y=2*h)) # magic
                # else: # It is an X spider
                #     self.add_to_phase(v1,CliffordPhase(self.dim,0,pow(-2*eo.had,-1,self.dim))) # more magic
                new = Edge(had=0, simple = 1)
                self.graph[v1][v2] = new
                self.graph[v2][v1] = new
                if not old: self.nedges += 1  # We have added a new edge
                return

            # no simple edges, so only H-edges
            h = (eo.had + old.had) % self.dim
            if h == 0: 
                if old:  # There was an old edge, but no longer
                    self.remove_edge(e)
                return  # No edge to add
            new = Edge(had=h,simple=0)
            self.graph[v1][v2] = new
            self.graph[v2][v1] = new
            if not old: self.nedges += 1  # We have added a new edge
            return

        if t1 == VertexType.X and t2 == VertexType.X:  # Both X-spiders
            if not eo.is_reduced():
                raise ValueError("Complicated edge types are currently not supported for X-spiders")
            if eo.is_had_edge():
                if old and old.is_simple_edge():
                    raise ValueError("Adding H-edge to regular edge between X-spider: complicated edge types are currently not supported for X-spiders")
                h = (eo.had + old.had) % self.dim
                if h == 0: 
                    if old:  # There was an old edge, but no longer
                        self.remove_edge(e)
                    return  # No edge to add
                new = Edge(had=h,simple=0)
                self.graph[v1][v2] = new
                self.graph[v2][v1] = new
                if not old: self.nedges += 1  # We have added a new edge
            else:  # eo is a simple edge
                if old and old.is_had_edge():
                    raise ValueError("Adding H-edge to regular edge between X-spider: complicated edge types are currently not supported for X-spiders")
                new = Edge(had=0,simple=1)  # Simple edges collapse to a single edge for X-X connections
                self.graph[v1][v2] = new
                self.graph[v2][v1] = new
                if not old: self.nedges += 1  # We have added a new edge
                return

        # We now know that one of them must be a Z spider and the other an X spider
        # This means that regular edges go modulo d, while Hadamard edges are collapsed to 1
        if not eo.is_reduced():
            raise ValueError("Complicated edge types are currently not supported for connections between Z- and X-spiders")
        if eo.is_simple_edge():
            if old and old.is_hadamard_edge():
                raise ValueError("Adding simple edge to regular edge between Z- and X-spider: complicated edge types are currently not supported")
            s = (eo.simple + old.simple) % self.dim
            if s == 0: 
                if old:  # There was an old edge, but no longer
                    self.remove_edge(e)
                return  # No edge to add
            new = Edge(had=0,simple=s)
            self.graph[v1][v2] = new
            self.graph[v2][v1] = new
            if not old: self.nedges += 1  # We have added a new edge
        else:  # eo is an H-edge
            if old and old.is_simple_edge():
                raise ValueError("Adding H-edge to regular edge between Z- and X-spider: complicated edge types are currently not supported")
            new = Edge(had=1,simple=0)  # H-edges collapse to a single edge for Z-X connections
            self.graph[v1][v2] = new
            self.graph[v2][v1] = new
            if not old: self.nedges += 1  # We have added a new edge
            return

    def remove_vertices(self, vertices):
        for v in vertices:
            vs = list(self.graph[v])
            # remove all edges
            for v1 in vs:
                self.nedges -= 1
                del self.graph[v][v1]
                if v != v1:
                    del self.graph[v1][v]
            # remove the vertex
            del self.graph[v]
            del self.ty[v]
            del self._phase[v]
            if v in self._inputs:
                self._inputs = tuple(u for u in self._inputs if u != v)
            if v in self._outputs:
                self._outputs = tuple(u for u in self._outputs if u != v)
            try: del self._qindex[v]
            except: pass
            try: del self._rindex[v]
            except: pass
            try: del self.phase_index[v]
            except: pass
        self._vindex = max(self.vertices(),default=0) + 1

    def remove_vertex(self, vertex):
        self.remove_vertices([vertex])

    def remove_edges(self, edges):
        for s,t in edges:
            self.nedges -= 1
            del self.graph[s][t]
            if s != t:
                del self.graph[t][s]

    def remove_edge(self, edge):
        self.remove_edges([edge])

    def num_vertices(self):
        return len(self.graph)

    def num_edges(self):
        #return self.nedges
        return len(self.edge_set())

    def vertices(self):
        return self.graph.keys()


    def edges(self):
        for v0,adj in self.graph.items():
            for v1 in adj:
                if v1 > v0: yield (v0,v1)

    def edge(self, s, t):
        return (s,t) if s < t else (t,s)

    def edge_set(self):
        return set(self.edges())

    def edge_st(self, edge):
        return edge

    def neighbors(self, vertex):
        return self.graph[vertex].keys()

    def vertex_degree(self, vertex):
        return len(self.graph[vertex])

    def incident_edges(self, vertex):
        return [(vertex, v1) if v1 > vertex else (v1, vertex) for v1 in self.graph[vertex]]

    def connected(self,v1,v2):
        return v2 in self.graph[v1]

    def edge_object(self, e):
        v1,v2 = e
        try:
            return self.graph[v1][v2]
        except KeyError:
            return Edge(0,0)

    def set_edge_object(self, e, t):
        v1,v2 = e
        self.graph[v1][v2] = t
        self.graph[v2][v1] = t

    def type(self, vertex):
        return self.ty[vertex]
    def types(self):
        return self.ty
    def set_type(self, vertex, t):
        self.ty[vertex] = t

    def phase(self, vertex):
        return self._phase.get(vertex,CliffordPhase(self.dim))
    def phases(self):
        return self._phase
    def set_phase(self, vertex, phase):
        self._phase[vertex] = phase
    def add_to_phase(self, vertex, phase):
        old_phase = self._phase.get(vertex, CliffordPhase(self.dim))
        self._phase[vertex] = old_phase + phase
    
    def qubit(self, vertex):
        return self._qindex.get(vertex,-1)
    def qubits(self):
        return self._qindex
    def set_qubit(self, vertex, q):
        if q > self._maxq: self._maxq = q
        self._qindex[vertex] = q

    def row(self, vertex):
        return self._rindex.get(vertex, -1)
    def rows(self):
        return self._rindex
    def set_row(self, vertex, r):
        if r > self._maxr: self._maxr = r
        self._rindex[vertex] = r

