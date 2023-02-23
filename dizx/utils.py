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
from argparse import ArgumentTypeError
from fractions import Fraction
from typing import Union, Optional, List, Dict, Any
from typing_extensions import Literal, Final


FloatInt = Union[float,int]
FractionLike = Union[Fraction,int]


class VertexType:
    """Type of a vertex in the graph."""
    Type = Literal[0,1,2]
    BOUNDARY: Final = 0
    Z: Final = 1
    X: Final = 2

def toggle_vertex(ty: VertexType.Type) -> VertexType.Type:
    """Swap the X and Z vertex types."""
    if not vertex_is_zx(ty):
        return ty
    return VertexType.Z if ty == VertexType.X else VertexType.X


tikz_classes = {
    'boundary': 'none',
    'Z': 'Z dot',
    'X': 'X dot',
    'Z phase': 'Z phase dot',
    'X phase': 'X phase dot',
    'H': 'hadamard',
    'edge': '',
    'H-edge': 'hadamard edge'
}

class Settings(object): # namespace class
    mode: Literal["notebook", "browser", "shell"] = "shell"
    drawing_backend: Literal["d3","matplotlib"] = "d3"
    drawing_auto_hbox: bool = False
    javascript_location: str = "" # Path to javascript files of pyzx
    d3_load_string: str = ""
    tikzit_location: str = "" # Path to tikzit executable
    show_labels: bool = False
    tikz_classes: Dict[str,str] = tikz_classes
    dim: int = 3

settings = Settings()

settings.javascript_location = os.path.join(os.path.dirname(__file__), 'js')