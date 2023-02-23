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

"""This file contains the Edge class used to represent the edges between two
nodes in a Graph."""


class Edge(object):
    SimpleEdge = 1
    HadEdge = 2

    def __init__(self, dim, had=0, simple=0):
        self.dim = dim
        self._had = had % self.dim
        self._simple = simple % self.dim

    @property
    def had(self) -> int:
        return self._had

    @property
    def simple(self) -> int:
        return self._simple

    def is_edge_present(self) -> bool:
        return self.had != 0 or self.simple != 0

    def is_had_edge(self) -> bool:
        return self.is_edge_present() and self.simple == 0

    def is_simple_edge(self) -> bool:
        return self.is_edge_present() and self.had == 0

    def is_reduced(self) -> bool:
        return self.had == 0 or self.simple == 0

    def type(self) -> int:
        if not self.is_reduced():
            raise ValueError(
                "This edge is not in reduced form, so it doesn't have a definitive type")
        return Edge.SimpleEdge if self.had == 0 else Edge.HadEdge

    def to_tuple(self):
        return self.had, self.simple

    def __add__(self, edge2):
        assert (self.dim == edge2.dim)
        return Edge(self.dim, self.had + edge2.had,
                    self.simple + edge2.simple)

    def __bool__(self):
        return self.is_edge_present()

    def __int__(self):
        return self.had + self.simple

    def __str__(self):
        return f"Edge(h={self.had},s={self.simple})"
