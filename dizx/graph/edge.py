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

"""This file contains the Edge class used to represent the edges between two nodes in a Graph."""

class Edge(object):
    def __init__(self,dim,had=0,simple=0):
        self.dim = dim
        self.had = had % self.dim
        self.simple = simple % self.dim

    def isEdgePresent(self) -> bool:
        return False if (self._numH == 0 and self._numNotH == 0) else True

    def is_had_edge(self) -> bool:
        return True if (self.isEdgePresent() and self._numNotH == 0) else False
        
    def is_simple_edge(self) -> bool:
        return True if (self.isEdgePresent() and self._numH == 0) else False

    def __add__(edge1, edge2):
        assert(edge1.dim == edge2.dim)
        return Edge(edge1.dim, edge1.had + edge2.had, edge1.simple + edge2.simple)

    def __bool__(self):
        return self.isEdgePresent()

    def __str__(self):
        return "Edge(h={},s={})" % (self.had, self.simple)