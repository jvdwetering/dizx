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

"""
This file contains the definition of commonly used
quantum gates for use in the Circuit class.
"""

import copy
import math
from typing import Dict, List, Optional, Type, ClassVar, TypeVar, Generic, Set, TYPE_CHECKING
from ..graph.phase import Phase, CliffordPhase
from ..graph.edge import Edge
from ..utils import settings

from ..utils import VertexType
from ..graph.base import BaseGraph, VT, ET

# We need this type variable so that the subclasses of Gate return the correct type for functions like copy()
Tvar = TypeVar('Tvar', bound='Gate')

if TYPE_CHECKING:
    from . import Circuit

class TargetMapper(Generic[VT]):
    """
    This class is used to map the target parameters of a gate to rows, qudits, and vertices
    when converting them into a graph. Used by :func:`~pyzx.circuit.gates.Gate.to_graph`.
    """
    _qudits: Dict[int, int]
    _rows: Dict[int, int]
    _prev_vs: Dict[int, VT]

    def __init__(self):
        self._qudits = {}
        self._rows = {}
        self._prev_vs = {}

    def labels(self) -> Set[int]:
        """
        Returns the mapped labels.
        """
        return set(self._qudits.keys())

    def to_qudit(self, l: int) -> int:
        """
        Maps a label to the qudit id in the graph.
        """
        return self._qudits[l]

    def set_qudit(self, l: int, q: int) -> None:
        """
        Sets the qudit id for a label.
        """
        self._qudits[l] = q

    def next_row(self, l: int) -> int:
        """
        Returns the next free row in the label's qudit.
        """
        return self._rows[l]

    def set_next_row(self, l: int, row: int) -> None:
        """
        Sets the next free row in the label's qudit.
        """
        self._rows[l] = row

    def advance_next_row(self, l: int) -> None:
        """
        Advances the next free row in the label's qudit by one.
        """
        self._rows[l] += 1

    def shift_all_rows(self, n: int) -> None:
        """
        Shifts all 'next rows' by n.
        """
        for l in self._rows.keys():
            self._rows[l] += n

    def set_all_rows(self, n: int) -> None:
        """
        Set the value of all 'next rows'.
        """
        for l in self._rows.keys():
            self._rows[l] = n

    def max_row(self) -> int:
        """
        Returns the highest 'next row' number.
        """
        return max(self._rows.values(), default=0)

    def prev_vertex(self, l: int) -> VT:
        """
        Returns the previous vertex in the label's qudit.
        """
        return self._prev_vs[l]

    def set_prev_vertex(self, l: int, v: VT) -> None:
        """
        Sets the previous vertex in the label's qudit.
        """
        self._prev_vs[l] = v

    def add_label(self, l: int) -> None:
        """
        Adds a tracked label.

        :raises: ValueError if the label is already tracked.
        """
        if l in self._qudits:
            raise ValueError("Label {} already in use".format(str(l)))
        q = len(self._qudits)
        self.set_qudit(l, q)
        r = self.max_row()
        self.set_all_rows(r)
        self.set_next_row(l, r + 1)

    def remove_label(self, l: int) -> None:
        """
        Removes a tracked label.

        :raises: ValueError if the label is not tracked.
        """
        if l not in self._qudits:
            raise ValueError("Label {} not in use".format(str(l)))
        self.set_all_rows(self.max_row() + 1)
        del self._qudits[l]
        del self._rows[l]
        del self._prev_vs[l]


class Gate(object):
    """Base class for representing quantum gates."""
    name: ClassVar[str] = "BaseGate"
    qasm_name: ClassVar[str] = 'undefined'
    qasm_name_adjoint: ClassVar[str] = 'undefined'
    index = 0
    repetitions: int = 1 # Allows multiple gates in a row to be represented by a single gate.

    def __init__(self, dim: int = settings.dim) -> None:
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def __str__(self) -> str:
        name = self.name
        if self.repetitions != 1:
            name += f"^{self.repetitions}"
        attribs = []
        if hasattr(self, "control"):
            attribs.append(str(self.control))  # type: ignore # See issue #1424
        if hasattr(self, "target"):
            attribs.append(
                str(self.target))  # type: ignore #https://github.com/python/mypy/issues/1424
        if hasattr(self, "phase") and self.printphase:  # type: ignore
            attribs.append("phase={!s}".format(self.phase))  # type: ignore
        return "{}{}({})".format(
            name,
            ("*" if (hasattr(self, "adjoint") and self.adjoint and self.repetitions == 1) else ""),
            # type: ignore
            ",".join(attribs))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):
            return False
        for a in ["target", "control", "phase", "adjoint", "repetitions"]:
            if hasattr(self, a):
                if not hasattr(other, a):
                    return False
                if getattr(self, a) != getattr(other, a):
                    return False
            elif hasattr(other, a):
                return False
        assert isinstance(other, Gate)
        if self.index != other.index:
            return False
        return True

    def _max_target(self) -> int:
        qudits = self.target  # type: ignore
        if hasattr(self, "control"):
            qudits = max(
                [qudits, self.control])  # type: ignore # See issue #1424
        return qudits

    def __add__(self, other) -> "Circuit":
        from . import Circuit
        c = Circuit(self._max_target() + 1)
        c.add_gate(self)
        c += other
        return c

    def __matmul__(self, other) -> "Circuit":
        from . import Circuit
        c = Circuit(self._max_target() + 1)
        c.add_gate(self)
        c2 = Circuit(other._max_target() + 1)
        c2.add_gate(other)
        return c @ c2
    
    def __pow__(self: Tvar, num: int) -> Tvar:
        g = self.copy()
        g.repetitions = self.repetitions * num
        return g
    
    def __xor__(self: Tvar, num: int) -> Tvar:
        """Alias for g**power."""
        return self**num

    def copy(self: Tvar) -> Tvar:
        return copy.deepcopy(self)
    
    def merge(self, other) -> None:
        """Merges this gate with another gate, typically used for combining gates in a circuit."""
        if self.name != other.name:
            raise ValueError("Cannot merge gates of different types: {} and {}".format(self.name, other.name))
        if hasattr(self, "target") and hasattr(other, "target"):
            if self.target != other.target:
                raise ValueError("Cannot merge gates with different targets: {} and {}".format(self.target, other.target))
        if hasattr(self, "control") and hasattr(other, "control"):
            if self.control != other.control:
                raise ValueError("Cannot merge gates with different controls: {} and {}".format(self.control, other.control))
        if hasattr(self, "phase") and hasattr(other, "phase"):
            self.phase += other.phase
        
        self.repetitions += other.repetitions

    def to_adjoint(self: Tvar) -> Tvar:
        g = self.copy()
        g.repetitions = -g.repetitions
        if hasattr(g, "phase"):
            g.phase = -g.phase  # type: ignore
        if hasattr(g, "adjoint"):
            g.adjoint = not g.adjoint  # type: ignore
        return g

    def reposition(
            self: Tvar, mask: List[int],
            bit_mask: Optional[List[int]] = None) -> Tvar:
        g = self.copy()
        if hasattr(g, "target"):
            g.target = mask[g.target]  # type: ignore
        if hasattr(g, "control"):
            g.control = mask[g.control]  # type: ignore
        return g

    def to_basic_gates(self) -> List['Gate']:
        return [self] * self.repetitions

    def to_qasm(self) -> str:
        n = self.qasm_name
        if n == 'undefined':
            bg = self.to_basic_gates()
            if len(bg) == 1:
                raise TypeError(
                    "Gate {} doesn't have a QASM description".format(
                        str(self)))
            return "\n".join(g.to_qasm() for g in bg)
        if hasattr(self, "adjoint") and self.adjoint:  # type: ignore
            n = self.qasm_name_adjoint

        args = []
        for a in ["ctrl1", "ctrl2", "control", "target"]:
            if hasattr(self, a):
                args.append("q[{:d}]".format(getattr(self, a)))
        param = ""
        if hasattr(self, "printphase") and self.printphase:  # type: ignore
            param = "({}*pi)".format(float(self.phase))  # type: ignore
        return "{}{} {};".format(n, param, ", ".join(args))

    def to_graph(
            self, g: BaseGraph[VT, ET], q_mapper: TargetMapper[VT],
            c_mapper: TargetMapper[VT]) -> None:
        """
        Add the converted gate to the graph.

        :param g: The graph to add the gate to.
        :param q_mapper: A mapper for qudit labels.
        :param c_mapper: A mapper for dit labels.
        """
        raise NotImplementedError(
            "to_graph() must be implemented by each Gate subclass.")

    def graph_add_node(
            self,
            g: BaseGraph[VT, ET],
            mapper: TargetMapper[VT],
            t: VertexType.Type,
            l: int, r: int,
            phase: Optional[Phase] = None,
            e: Edge = Edge(simple=1)) -> VT:
        v = g.add_vertex(t, mapper.to_qudit(l), r, phase)
        g.add_edge(g.edge(mapper.prev_vertex(l), v), e)
        mapper.set_prev_vertex(l, v)
        return v


class ZPhase(Gate):
    name = 'ZPhase'
    printphase: ClassVar[bool] = True
    qasm_name = 'rz'

    def __init__(
            self, target: int,
            phase=CliffordPhase(settings.dim, 0, 0)) -> None:
        self.target = target
        self.phase = phase

    def to_graph(self, g, q_mapper, _c_mapper):
        self.graph_add_node(g, q_mapper, VertexType.Z, self.target,
                            q_mapper.next_row(self.target), self.phase)
        q_mapper.advance_next_row(self.target)

    def split_phases(self) -> List['ZPhase']:
        if self.phase.is_zero():
            return []
        if self.phase.is_pauli():
            return [Z(self.target)] * self.phase.x
        if self.phase.is_pure_clifford():
            return [S(self.target)] * self.phase.y
        if self.phase.is_clifford():
            return [Z(self.target)] * self.phase.x + [
                S(self.target)] * self.phase.y
        else:
            return [self]


class Z(ZPhase):
    name = 'Z'
    qasm_name = 'z'
    qasm_name_adjoint = 'zdg'
    printphase = False

    def __init__(self, target: int, adjoint: bool = False, repetitions: int = 1) -> None:
        super().__init__(target, phase=CliffordPhase(settings.dim, 1, 0))
        self.adjoint = adjoint
        self.repetitions = repetitions * (-1 if adjoint else 1)

    def to_basic_gates(self):
        return [Z(self.target)] * (settings.dim - 1) if self.adjoint else [
            Z(self.target)]


class S(ZPhase):
    name = 'S'
    qasm_name = 's'
    qasm_name_adjoint = 'sdg'
    printphase = False

    def __init__(self, target: int, adjoint: bool = False, repetitions: int = 1) -> None:
        super().__init__(target, CliffordPhase(settings.dim, 0, 1))
        self.adjoint = adjoint
        self.repetitions = repetitions * (-1 if adjoint else 1)

    def to_basic_gates(self):
        return [S(self.target)] * (settings.dim - 1) if self.adjoint else [
            S(self.target)]


class XPhase(Gate):
    name = 'XPhase'
    printphase: ClassVar[bool] = True
    qasm_name = 'rx'

    def __init__(
            self, target: int,
            phase=CliffordPhase(settings.dim, 0, 0)) -> None:
        self.target = target
        self.phase = phase

    def to_graph(self, g, q_mapper, _c_mapper):
        self.graph_add_node(g, q_mapper, VertexType.X, self.target,
                            q_mapper.next_row(self.target), self.phase)
        q_mapper.advance_next_row(self.target)

    def split_phases(self) -> List['XPhase']:
        if self.phase.is_zero():
            return [NEG(self.target)]
        gates: List[Gate] = [HAD(self.target)]
        if self.phase.is_pauli():
            gates += [Z(self.target)] * self.phase.x
        elif self.phase.is_pure_clifford():
            gates += [S(self.target)] * self.phase.y
        elif self.phase.is_clifford():
            gates += [Z(self.target)] * self.phase.x + [
                S(self.target)] * self.phase.y
        else:
            return [self]
        gates += [HAD(self.target)]
        return gates


class NEG(XPhase):
    name = 'NEG'
    qasm_name = 'neg'
    printphase = False

    def __init__(self, target: int) -> None:
        super().__init__(target, phase=CliffordPhase(settings.dim, 0, 0))

    def to_basic_gates(self):
        return [HAD(self.target)] * 2


class X(Gate):
    name = 'X'
    qasm_name = 'x'
    qasm_name_adjoint = 'xdg'
    printphase = False

    def __init__(self, target: int, adjoint: bool = False, repetitions: int=1) -> None:
        self.target = target
        self.adjoint = adjoint
        self.repetitions = repetitions * (-1 if adjoint else 1)

    def to_basic_gates(self):
        return [HAD(self.target)] * 3 + [Z(self.target),
                                         HAD(self.target)] if self.adjoint else [
                                                                                    HAD(self.target),
                                                                                    Z(self.target)] + [
                                                                                    HAD(self.target)] * 3


class HAD(Gate):
    name = 'HAD'
    qasm_name = 'h'
    qasm_name_adjoint = 'hdg'

    def __init__(self, target: int, adjoint: bool = False) -> None:
        self.target = target
        self.adjoint = adjoint
        if adjoint: self.repetitions = -1

    def to_basic_gates(self):
        return [HAD(self.target)] * 3 if self.adjoint else [HAD(self.target)]

    def to_graph(self, g, q_mapper, _c_mapper):
        v = g.add_vertex(VertexType.Z, q_mapper.to_qudit(self.target),
                         q_mapper.next_row(self.target))
        g.add_edge((q_mapper.prev_vertex(self.target), v), Edge(had=1))
        q_mapper.set_prev_vertex(self.target, v)
        q_mapper.advance_next_row(self.target)


class GateWithControl(Gate):
    """Base class for gates that have a control qudit."""
    control: int = -1

    def __init__(self, control: int, target: int, adjoint: bool = False) -> None:
        super().__init__()
        self.control = control
        self.target = target
        self.adjoint = adjoint
        if adjoint: self.repetitions = -self.repetitions

class CX(GateWithControl):
    name = 'CX'
    qasm_name = 'cx'
    qasm_name_adjoint = 'cxdg'

    def to_basic_gates(self):
        return [HAD(self.target)] * 3 + [CZ(self.control, self.target),
                                         HAD(self.target)] if self.adjoint else [
                                                                                    HAD(self.target),
                                                                                    CZ(self.control,
                                                                                       self.target)] + [
                                                                                    HAD(self.target)] * 3

    def to_graph(self, g, q_mapper, c_mapper):
        r = max(q_mapper.next_row(self.target),
                q_mapper.next_row(self.control))
        c = self.graph_add_node(g, q_mapper, VertexType.Z, self.control, r)
        t = self.graph_add_node(g, q_mapper, VertexType.X, self.target, r)
        g.add_edge((t, c), Edge(simple=1))
        q_mapper.set_next_row(self.target, r + 1)
        q_mapper.set_next_row(self.control, r + 1)
        g.scalar.add_power(1)


class CZ(GateWithControl):
    name = 'CZ'
    qasm_name = 'cz'
    qasm_name_adjoint = 'czdg'

    def to_basic_gates(self):
        return [HAD(self.target)] * 2 + [CZ(self.control, self.target)] + [
            HAD(self.target)] * 2 if self.adjoint else [
            CZ(self.control, self.target)]

    def to_graph(self, g, q_mapper, _c_mapper):
        r = max(q_mapper.next_row(self.target),
                q_mapper.next_row(self.control))
        t = self.graph_add_node(g, q_mapper, VertexType.Z, self.target, r)
        c = self.graph_add_node(g, q_mapper, VertexType.Z, self.control, r)
        g.add_edge((t, c), Edge(had=1))
        q_mapper.set_next_row(self.target, r + 1)
        q_mapper.set_next_row(self.control, r + 1)
        g.scalar.add_power(1)


class SWAP(GateWithControl):
    name = 'SWAP'
    qasm_name = 'swap'

    def to_basic_gates(self):
        return [
            CX(self.control, self.target),
            CX(self.target, self.control),
            CX(self.control, self.target)
        ]

    def to_graph(self, g, q_mapper, c_mapper):
        for gate in self.to_basic_gates():
            gate.to_graph(g, q_mapper, c_mapper)


class InitAncilla(Gate):
    name = 'InitAncilla'

    def __init__(self, label):
        self.label = label


class PostSelect(Gate):
    name = 'PostSelect'

    def __init__(self, label):
        self.label = label

    # def to_graph(self, g, labels, qs, _cs, rs, _crs):
    #     v = g.add_vertex(VertexType.Z, self.label, 0)


class DiscardDit(Gate):
    name = 'DiscardDit'

    def __init__(self, target):
        self.target = target

    def reposition(self, _mask, bit_mask=None):
        g = self.copy()
        g.target = bit_mask[g.target]
        return g

    def to_graph(self, g, _q_mapper, c_mapper):
        r = c_mapper.next_row(self.target)
        self.graph_add_node(g,
                            c_mapper,
                            VertexType.Z,
                            self.target,
                            r,
                            ground=True)
        u = g.add_vertex(VertexType.X, c_mapper.to_qudit(self.target), r + 1)
        c_mapper.set_prev_vertex(self.target, u)
        c_mapper.set_next_row(self.target, r + 2)


class Measurement(Gate):
    target: int
    result_dit: Optional[int]

    # This gate has special syntax in qasm: https://qiskit.github.io/openqasm/language/insts.html

    def __init__(self, target: int, result_dit: Optional[int]) -> None:
        self.target = target
        self.result_dit = result_dit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Measurement):
            return False
        if self.target != other.target:
            return False
        if self.result_dit != other.result_dit:
            return False
        return False

    def reposition(self, mask, bit_mask=None):
        g = self.copy()
        g.target = mask[self.target]
        if self.result_dit is not None and bit_mask is not None:
            g.result_dit = bit_mask[self.result_dit]
        return g

    def to_graph(self, g, q_mapper, c_mapper):
        # Discard previous dit value
        if self.result_dit is not None:
            DiscardDit(self.result_dit).to_graph(g, q_mapper, c_mapper)
        # qudit measurement
        r = q_mapper.next_row(self.target)
        if self.result_dit is None:
            r = max(r, c_mapper.next_row(self.result_dit))
        v = self.graph_add_node(g,
                                q_mapper,
                                VertexType.Z,
                                self.target,
                                r,
                                ground=True)
        q_mapper.set_next_row(self.target, r + 1)
        # Classical result
        if self.result_dit is not None:
            u = self.graph_add_node(g,
                                    c_mapper,
                                    VertexType.X,
                                    self.result_dit,
                                    r)
            g.add_edge(g.edge(v, u), Edge(simple=1))
            c_mapper.set_next_row(self.result_dit, r + 1)


gate_types: Dict[str, Type[Gate]] = {
    "XPhase": XPhase,
    "ZPhase": ZPhase,
    "NEG": NEG,
    "X": X,
    "Z": Z,
    "S": S,
    "CZ": CZ,
    "CX": CX,
    "CNOT": CX,
    "SWAP": SWAP,
    "H": HAD,
    "HAD": HAD,
    "InitAncilla": InitAncilla,
    "PostSelect": PostSelect,
    "DiscardDit": DiscardDit,
    "Measurement": Measurement,
}

qasm_gate_table: Dict[str, Type[Gate]] = {
    "x": X,
    "xdg": X,
    "z": Z,
    "zdg": Z,
    "s": S,
    "sdg": S,
    "h": HAD,
    "hdg": HAD,
    "neg": NEG,
    "cx": CX,
    "cxdg": CX,
    "cz": CZ,
    "czdg": CZ,
    "swap": SWAP,
    "measure": Measurement,
}
