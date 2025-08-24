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

__version__ = "0.0.1"

from .graph import Graph, Edge, Phase, CliffordPhase
from .utils import VertexType, toggle_vertex, settings
from .drawing import draw
from .circuit import Circuit, gates, id
from . import simplify
from . import symplectic

if __name__ == '__main__':
    print("Please execute this as a module by running 'python -m dizx'")
