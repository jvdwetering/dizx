from __future__ import annotations
import abc
from cmath import exp, pi


class Phase(abc.ABC):

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @abc.abstractmethod
    def get_phase(self) -> complex:
        pass

    @abc.abstractmethod
    def adjoint(self) -> Phase:
        pass

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def __sub__(self, other):
        pass

    @abc.abstractmethod
    def is_pauli(self) -> bool:
        pass

    @abc.abstractmethod
    def is_clifford(self) -> bool:
        pass

    @abc.abstractmethod
    def is_pure_clifford(self) -> bool:
        pass


class CliffordPhase(Phase):

    def __init__(self, dim: int, x: int = 0, y: int = 0) -> None:
        super().__init__(dim)
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._x

    def get_phase(self) -> complex:
        omega = exp(1j * 2 * pi / self.dim)
        ret = 0 + 0j
        for k in range(self.dim):
            ret += omega ** ((self._x * k + self._y * k * k) / 2)
        return ret

    def adjoint(self) -> CliffordPhase:
        return CliffordPhase(self.dim, -self.x, -self.y)

    def __add__(self, other: CliffordPhase) -> CliffordPhase:
        if self.dim != other.dim:
            raise ValueError("Only phases with equal dimensions can be added")
        return CliffordPhase(self.dim, self.x + other.x, self.y + other.y)

    def __sub__(self, other: CliffordPhase) -> CliffordPhase:
        if self.dim != other.dim:
            raise ValueError("Only phases with equal dimensions can be "
                             "subtracted")
        return CliffordPhase(self.dim, self.x - other.x, self.y - other.y)

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __repr__(self) -> str:
        return f"CliffordPhase(dim={self.dim}, x={self.x}, y={self.y})"

    def is_pauli(self) -> bool:
        return self.y == 0

    def is_clifford(self) -> bool:
        return True

    def is_pure_clifford(self) -> bool:
        return self.x == 0