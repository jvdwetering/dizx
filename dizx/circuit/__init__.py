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

import os
from typing import List, Union, Optional, Iterator, Dict

import numpy as np

from qiskit import QuantumCircuit

from .gates import Gate, gate_types, XPhase, ZPhase, NEG, X, Z, S, CZ, CX, SWAP, HAD, Measurement

from ..graph.base import BaseGraph
from .. import symplectic
from ..utils import settings

CircuitLike = Union['Circuit', Gate]

# Note that many of the method of Circuit contain inline imports. These are
# there to prevent circular imports.

__all__ = ['Circuit', 'id']

class Circuit(object):
    """Class for representing quantum circuits.

    This class is mostly just a wrapper for a list of gates with methods for converting
    between different representations of a quantum circuit.

    The methods in this class that convert a specification of a circuit into an instance of this class,
    generally do not check whether the specification is well-defined. If a bad input is given,
    the behaviour is undefined."""
    def __init__(self, qudit_amount: int, dim: int = settings.dim, name: str = '', dit_amount: Optional[int] = None) -> None:
        self.qudits: int        = qudit_amount
        self._dim = dim
        self.dits: int = 0 if dit_amount is None else dit_amount
        self.gates:  List[Gate] = []
        self.name:   str        = name

    ### BASIC FUNCTIONALITY
    @property
    def dim(self) -> int:
        return self._dim

    def __str__(self) -> str:
        return f"Circuit[{self.dim}]({self.qudits} qudits, {self.dits} dits," \
               f" {self.gates} gates)"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> 'Circuit':
        c = Circuit(self.qudits, self.dim, self.name, self.dits)
        c.gates = [g.copy() for g in self.gates]
        return c

    def adjoint(self) -> 'Circuit':
        c = Circuit(self.qudits, self.dim, self.name + 'Adjoint', self.dits)
        for g in reversed(self.gates):
            c.gates.append(g.to_adjoint())
        return c

    # def verify_equality(self, other: 'Circuit', up_to_swaps: bool = False) -> bool:
    #     """Composes the other circuit with the adjoint of this circuit, and tries to reduce
    #     it to the identity using :func:`simplify.full_reduce``. If successful returns True,
    #     if not returns None.

    #     Note:
    #         A successful reduction to the identity is strong evidence that the two
    #         circuits are equal, if this function is not able to reduce the graph to the identity
    #         this does not prove anything.

    #     Args:
    #         other: the circuit to compare equality to.
    #         up_to_swaps: if set to True, only checks equality up to a permutation of the qudits.

    #     """
    #     if self.dits or other.dits:
    #         # TODO once full_gnd_reduce is merged
    #         raise NotImplementedError("The equality verification does not support hybrid circuits.")

    #     from ..simplify import full_reduce
    #     c = self.adjoint()
    #     c.add_circuit(other)
    #     g = c.to_graph()
    #     full_reduce(g)
    #     if (g.num_vertices() == self.qudits*2 and
    #             all(g.edge_type(e) == EdgeType.SIMPLE for e in g.edges())):
    #         if up_to_swaps:
    #             return True
    #         else:
    #             return all(g.connected(v,w) for v,w in zip(g.inputs(),g.outputs()))
    #     else:
    #         return False

    def add_gate(self, gate: Union[Gate,str], *args, **kwargs) -> None:
        """Adds a gate to the circuit. ``gate`` can either be
        an instance of a :class:`Gate`, or it can be the name of a gate,
        in which case additional arguments should be given.

        Example::

            circuit.add_gate("CNOT", 1, 4) # adds a CNOT gate with control 1 and target 4
            circuit.add_gate("ZPhase", 2, phase=Fraction(3,4)) # Adds a ZPhase gate on qudit 2 with phase 3/4
        """
        if isinstance(gate, str):
            gate_class = gate_types[gate]
            gate = gate_class(*args, **kwargs) # type: ignore
        self.gates.append(gate)

    def prepend_gate(self, gate, *args, **kwargs):
        """The same as add_gate, but adds the gate to the start of the circuit, not the end.
        """
        if isinstance(gate, str):
            gate_class = gates.gate_types[gate]
            gate = gate_class(*args, **kwargs)
        self.gates.insert(0, gate)

    def add_gates(self, gates: str, qudit: int) -> None:
        """Adds a series of single qudit gates on the same qudit.
        ``gates`` should be a space-separated string of gatenames.

        Example::

            circuit.add_gates("S T H T H", 1)
        """
        for g in gates.split(" "):
            self.add_gate(g, qudit)

    def to_basic_gates(self) -> 'Circuit':
        """Returns a new circuit with every gate expanded in terms of X/Z phases, Z, S, Hadamard,
        and the 2-qudit gate CZ."""
        c = Circuit(self.qudits, name=self.name, dit_amount=self.dits)
        for g in self.gates:
            c.gates.extend(g.to_basic_gates())
        return c

    def split_phase_gates(self) -> 'Circuit':
        c = Circuit(self.qudits, name=self.name)
        for g in self.gates:
            if isinstance(g, (ZPhase, XPhase)):
                c.gates.extend(g.split_phases())
            else:
                c.add_gate(g)
        return c

    def add_circuit(self, other: 'Circuit') -> None:
        """Adds the gates of another circuit to this circuit.
        If the other circuit has more qudits than this circuit, the number of qudits
        in this circuit is updated to match."""
        self.gates.extend(other.gates)
        if other.qudits > self.qudits:
            self.qudits = other.qudits

    ### OPERATORS

    def __iadd__(self, other: CircuitLike) -> 'Circuit':
        if isinstance(other, Circuit):
            self.add_circuit(other)
            if other.qudits > self.qudits:
                self.qudits = other.qudits
        elif isinstance(other, Gate):
            self.add_gate(other)
            if other._max_target() + 1 > self.qudits:
                self.qudits = other._max_target() + 1
        else:
            raise Exception("Cannot add object of type", type(other), "to Circuit")
        return self

    def __add__(self, other: CircuitLike) -> 'Circuit':
        c = self.copy()
        c += other
        return c

    def __len__(self) -> int:
        return len(self.gates)

    def __iter__(self) -> Iterator[Gate]:
        return iter(self.gates)

    def __matmul__(self, other: CircuitLike) -> 'Circuit':
        return self.tensor(other)


    ### MATRIX EMULATION (FOR E.G. Mat2.gauss)

    def row_add(self, q0: int, q1: int):
        self.add_gate("CNOT", q0, q1)

    def col_add(self, q0: int, q1: int):
        self.prepend_gate("CNOT", q1, q0)


    ### CONVERSION METHODS


    # @staticmethod
    # def from_graph(g:BaseGraph, split_phases:bool=True) -> 'Circuit':
    #     """Produces a :class:`Circuit` containing the gates of the given ZX-graph.
    #     If the ZX-graph is not circuit-like then the behaviour of this function
    #     is undefined.
    #     ``split_phases`` governs whether nodes with phases should be split into
    #     Z,S, and T gates or if generic ZPhase/XPhase gates should be used."""
    #     from .graphparser import graph_to_circuit
    #     return graph_to_circuit(g, split_phases=split_phases)

    def to_graph(self, zh:bool=False, compress_rows:bool=True) -> BaseGraph:
        """Turns the circuit into a ZX-Graph.
        If ``compress_rows`` is set, it tries to put single qudit gates on different qudits,
        on the same row."""
        from .graphparser import circuit_to_graph

        return circuit_to_graph(self if zh else self.to_basic_gates(),
            compress_rows)

    # def to_tensor(self, preserve_scalar:bool=True) -> np.ndarray:
    #     """Returns a numpy tensor describing the circuit."""
    #     return self.to_graph().to_tensor(preserve_scalar)
    # def to_matrix(self, preserve_scalar=True) -> np.ndarray:
    #     """Returns a numpy matrix describing the circuit."""
    #     return self.to_graph().to_matrix(preserve_scalar)

    def to_qasm(self) -> str:
        """Produces a QASM description of the circuit."""
        s = """OPENQASM D.0;\ninclude "qelib1.inc";\n"""
        s += "qreg q[{!s}];\n".format(self.qudits)
        for g in self.gates:
            s += g.to_qasm() + "\n"
        return s
    
    def to_symplectic_matrix(self) -> symplectic.Matrix:
        """Calculates the symplectic representation of a Circuit c. Only supports Clifford gates."""
        qudits = self.qudits
        mat = symplectic.ID(qudits)

        for g in self.gates:
            if isinstance(g, (Z, X)):
                continue # Pauli gates are identities in the symplectic representation, so we can ignore them
            elif isinstance(g, S):
                m = symplectic.S(g.target, qudits, reps=g.repetitions)
            elif isinstance(g, HAD):
                m = symplectic.H(g.target, qudits, reps=g.repetitions)
            elif isinstance(g, CX):
                m = symplectic.CX(g.control,g.target,qudits,reps=g.repetitions)
            elif isinstance(g, CZ):
                m = symplectic.CZ(g.control,g.target,qudits,reps=g.repetitions)
            elif isinstance(g, SWAP):
                if g.repetitions % 2 == 0: continue
                m = symplectic.SWAP(g.control,g.target,qudits)
            else:
                raise ValueError("Unsupported gate", str(g))
            mat = m*mat # We multiply this way since circuit order goes the opposite direction of matrix multiplication order
        
        return mat
    
    def to_qiskit_rep(self) -> QuantumCircuit:
        from qiskit.circuit.library import HGate, CXGate, CZGate, SwapGate
        qc = QuantumCircuit(self.qudits)
        for g in self.gates:
            if g.repetitions == 1:
                label = ""
            else:
                label = "^" + str(g.repetitions)
            if isinstance(g, (Z, X, HAD, S)):
                label = g.name + label
                qc.append(HGate(label=label),[g.target])
            elif isinstance(g, CX):
                qc.append(CXGate(label=label),[g.control,g.target])
            elif isinstance(g, CZ):
                qc.append(CZGate(label=label),[g.control,g.target])
            elif isinstance(g, SWAP):
                qc.append(SwapGate(label=label),[g.control,g.target])
            else:
                raise ValueError("Unsupported gate", str(g))
        return qc
            
            

def id(n: int) -> Circuit:
    return Circuit(n)
