# DiZX - Python library for quantum circuit rewriting
#        and optimisation using the qudit ZX-calculus
# Copyright (C) 2023 - Boldiszar Poor, Lia Yeh and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

from . import Circuit
from .gates import TargetMapper
from ..utils import VertexType, FloatInt, FractionLike, settings
from ..graph import Graph
from ..graph.base import BaseGraph, VT, ET
from ..graph.phase import Phase, CliffordPhase
from ..graph.edge import Edge

def circuit_to_graph(c: Circuit, compress_rows:bool=True) -> BaseGraph[VT, ET]:
    """Turns the circuit into a ZX-Graph.
    If ``compress_rows`` is set, it tries to put single qudit gates on different qudits,
    on the same row."""
    g = Graph(settings.dim)
    q_mapper: TargetMapper[VT] = TargetMapper()
    c_mapper: TargetMapper[VT] = TargetMapper()
    inputs = []
    outputs = []

    for i in range(c.qudits):
        v = g.add_vertex(VertexType.BOUNDARY,i,0)
        inputs.append(v)
        q_mapper.set_prev_vertex(i, v)
        q_mapper.set_next_row(i, 1)
        q_mapper.set_qudit(i, i)
    for i in range(c.dits):
        qudit = i+c.qudits
        v = g.add_vertex(VertexType.BOUNDARY, qudit, 0)
        inputs.append(v)
        c_mapper.set_prev_vertex(i, v)
        c_mapper.set_next_row(i, 1)
        c_mapper.set_qudit(i, qudit)

    for gate in c.gates:
        if gate.name == 'InitAncilla':
            l = gate.label # type: ignore
            try:
                q_mapper.add_label(l)
            except ValueError:
                raise ValueError("Ancilla label {} already in use".format(str(l)))
            v = g.add_vertex(VertexType.Z, q_mapper.to_qudit(l), q_mapper.next_row(l))
            q_mapper.set_prev_vertex(l, v)
        elif gate.name == 'PostSelect':
            l = gate.label # type: ignore
            try:
                q = q_mapper.to_qudit(l)
                r = q_mapper.next_row(l)
                u = q_mapper.prev_vertex(l)
                q_mapper.remove_label(l)
            except ValueError:
                raise ValueError("PostSelect label {} is not in use".format(str(l)))
            v = g.add_vertex(VertexType.Z, q, r)
            g.add_edge(g.edge(u,v),Edge(simple=1))
        else:
            if not compress_rows: #or not isinstance(gate, (ZPhase, XPhase, HAD)):
                r = max(q_mapper.max_row(), c_mapper.max_row())
                q_mapper.set_all_rows(r)
                c_mapper.set_all_rows(r)
            gate.to_graph(g, q_mapper, c_mapper)
            if not compress_rows: # or not isinstance(gate, (ZPhase, XPhase, HAD)):
                r = max(q_mapper.max_row(), c_mapper.max_row())
                q_mapper.set_all_rows(r)
                c_mapper.set_all_rows(r)

    r = max(q_mapper.max_row(), c_mapper.max_row())
    for mapper in (q_mapper, c_mapper):
        for l in mapper.labels():
            o = mapper.to_qudit(l)
            v = g.add_vertex(VertexType.BOUNDARY, o, r)
            outputs.append(v)
            u = mapper.prev_vertex(l)
            g.add_edge(g.edge(u,v),Edge(simple=1))

    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))

    return g