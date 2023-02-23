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


"""This file contains the Scalar class used to represent a global scalar in a Graph."""

import math
import cmath
from fractions import Fraction

from .phase import CliffordPhase, Phase

from ..utils import FloatInt, FractionLike

__all__ = ['Scalar']

def cexp(val) -> complex:
    return cmath.exp(1j*math.pi*val)

class Scalar(object):
    """Represents a global scalar for a Graph instance."""
    def __init__(self, dim) -> None:
        self.dim = dim
        self.power_dim: int = 0 # Stores power of square root of the dimension
        self.phase: Fraction = Fraction(0) # Stores complex phase of the number
        self.floatfactor: complex = 1.0
        self.is_unknown: bool = False # Whether this represents an unknown scalar value
        self.is_zero: bool = False

    def __repr__(self) -> str:
        return "Scalar({})".format(str(self))

    def __str__(self) -> str:
        if self.is_unknown:
            return "UNKNOWN"
        s = "{0.real:.2f}{0.imag:+.2f}i = ".format(self.to_number())
        if self.floatfactor != 1.0:
            s += "{0.real:.2f}{0.imag:+.2f}i".format(self.floatfactor)
        if self.phase:
            s += "exp({}ipi)".format(str(self.phase))
        s += "sqrt({:d})^{:d}".format(self.dim, self.power_dim)
        return s

    def __complex__(self) -> complex:
        return self.to_number()

    def copy(self) -> 'Scalar':
        s = Scalar(self.dim)
        s.power_dim = self.power_dim
        s.phase = self.phase
        s.floatfactor = self.floatfactor
        s.is_unknown = self.is_unknown
        s.is_zero = self.is_zero
        return s

    def to_number(self) -> complex:
        if self.is_zero: return 0
        val = cexp(self.phase)
        val *= math.sqrt(self.dim)**self.power_dim
        return val*self.floatfactor

    def set_unknown(self) -> None:
        self.is_unknown = True

    def add_power(self, n) -> None:
        """Adds a factor of sqrt(d)^n to the scalar."""
        self.power_dim += n
    def add_phase(self, phase: FractionLike) -> None:
        """Multiplies the scalar by a complex phase."""
        self.phase = (self.phase + phase) % 2
    def add_node(self, node: Phase) -> None:
        """A solitary spider with a phase ``node`` is converted into the
        scalar."""
        self.add_float(node.get_phase())
    def add_float(self,f: complex) -> None:
        self.floatfactor *= f

    def mult_with_scalar(self, other: 'Scalar') -> None:
        """Multiplies two instances of Scalar together."""
        self.power_dim += other.power_dim
        self.phase = (self.phase +other.phase)%2
        self.floatfactor *= other.floatfactor
        if other.is_zero: self.is_zero = True
        if other.is_unknown: self.is_unknown = True

    def add_clifford_spider_pair(self, p1: CliffordPhase, p2: CliffordPhase) -> None:
        """Add the scalar corresponding to a connected pair of spiders (p1)-H-(p2)."""
        assert p1.y == 0
        self.add_power(1)
        omega_pow = pow(2, -2, self.dim) * p1.x * p2.x + \
                    pow(2, -3, self.dim) * pow(p1.x, 2, self.dim) * p2.y
        self.add_phase(Fraction(2 * omega_pow, self.dim))

    def add_spider_pair(self, p1: Phase, p2: Phase) -> None:
        if isinstance(p1, CliffordPhase) and isinstance(p2, CliffordPhase):
            return self.add_clifford_spider_pair(p1, p2)
        raise NotImplementedError()
