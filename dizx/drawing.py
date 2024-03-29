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

__all__ = ['draw']

import os
import math
import json
import string
import random
from typing import Dict, List, Tuple, Optional, Iterable, Any

from matplotlib import lines, path, patches
from typing_extensions import Literal

from IPython.display import display, HTML

from .utils import settings, VertexType, FloatInt
from .graph.base import BaseGraph, VT, ET
from .graph import Edge

import matplotlib.pyplot as plt


def draw(g: BaseGraph[VT,ET], labels: bool=False, **kwargs) -> Any:
    """Draws the given Graph. 
    Depending on the value of ``pyzx.settings.drawing_backend``
    either uses matplotlib or d3 to draw."""

    # allow global setting to labels=False
    # TODO: probably better to make labels Optional[bool]
    labels = labels or settings.show_labels

    if settings.drawing_backend == "d3":
        return draw_d3(g, labels, **kwargs)
    elif settings.drawing_backend == "matplotlib":
        return draw_matplotlib(g, labels, **kwargs)
    else:
        raise TypeError("Unsupported drawing backend '{}'".format(settings.drawing_backend))


def draw_matplotlib(
        g:      BaseGraph[VT,ET], 
        labels: bool                             =False, 
        figsize:Tuple[FloatInt,FloatInt]         =(8,2), 
        h_edge_draw: Literal['blue', 'box']      ='blue', 
        show_scalar: bool                        =False,
        rows: Optional[Tuple[FloatInt,FloatInt]] =None
        ) -> Any: # TODO: Returns a matplotlib figure
    if plt is None:
        raise ImportError("This function requires matplotlib to be installed. "
            "If you are running in a Jupyter notebook, you can instead use `zx.draw_d3`.")
    fig1 = plt.figure(figsize=figsize)
    ax = fig1.add_axes([0, 0, 1, 1], frameon=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    vs_on_row: Dict[FloatInt, int] = {} # count the vertices on each row
    for v in g.vertices():
        vs_on_row[g.row(v)] = vs_on_row.get(g.row(v), 0) + 1
    
    #Dict[VT,Tuple[FloatInt,FloatInt]]
    layout = {v:(g.row(v),-g.qubit(v)) for v in g.vertices()}

    if rows is not None:
        minrow,maxrow = rows
        vertices: Iterable[VT] = [v for v in g.vertices() if (minrow<=g.row(v) and g.row(v) <=maxrow)]
        edges: Iterable[ET] = [e for e in g.edges() if g.edge_s(e) in vertices and g.edge_t(e) in vertices]
    else:
        vertices = g.vertices()
        edges = g.edges()
    
    for e in edges:
        sp = layout[g.edge_s(e)]
        tp = layout[g.edge_t(e)]
        et = g.edge_t(e)
        n_row = vs_on_row.get(g.row(g.edge_s(e)), 0)

        
        dx = tp[0] - sp[0]
        dy = tp[1] - sp[1]
        bend_wire = (dx == 0) and h_edge_draw == 'blue' and n_row > 2
        ecol = '#0099ff' if h_edge_draw == 'blue' and et == 2 else 'black'

        if bend_wire:
            bend = 0.25
            mid = (sp[0] + 0.5 * dx + bend * dy, sp[1] + 0.5 * dy - bend * dx)

            pth = path.Path([sp,mid,tp], [path.Path.MOVETO, path.Path.CURVE3, path.Path.LINETO])
            patch = patches.PathPatch(pth, edgecolor=ecol, linewidth=0.8, fill=False)
            ax.add_patch(patch)
        else:
            pos = 0.5 if dx == 0 or dy == 0 else 0.4
            mid = (sp[0] + pos*dx, sp[1] + pos*dy)
            ax.add_line(lines.Line2D([sp[0],tp[0]],[sp[1],tp[1]], color=ecol, linewidth=0.8, zorder=0))

        if h_edge_draw == 'box' and et == 2: #hadamard edge
            w = 0.2
            h = 0.15
            diag = math.sqrt(w*w+h*h)
            angle = math.atan2(dy,dx)
            angle2 = math.atan2(h,w)
            centre = (mid[0] - diag/2*math.cos(angle+angle2),
                      mid[1] - diag/2*math.sin(angle+angle2))
            ax.add_patch(patches.Rectangle(centre,w,h,angle=angle/math.pi*180,facecolor='yellow',edgecolor='black'))

        #plt.plot([sp[0],tp[0]],[sp[1],tp[1]], 'k', zorder=0, linewidth=0.8)
    
    for v in vertices:
        p = layout[v]
        t = g.type(v)
        a = g.phase(v)
        a_offset = 0.5

        if t == VertexType.Z:
            ax.add_patch(patches.Circle(p, 0.2, facecolor='green', edgecolor='black', zorder=1))
        elif t == VertexType.X:
            ax.add_patch(patches.Circle(p, 0.2, facecolor='red', edgecolor='black', zorder=1))
        else:
            ax.add_patch(patches.Circle(p, 0.1, facecolor='black', edgecolor='black', zorder=1))

        if labels: plt.text(p[0]+0.25, p[1]+0.25, str(v), ha='center', color='gray', fontsize=5)
        # if a: plt.text(p[0], p[1]-a_offset, phase_to_s(a, t), ha='center', color='blue', fontsize=8)
    
    if show_scalar:
        x = min((g.row(v) for v in g.vertices()), default = 0)
        y = -sum((g.qubit(v) for v in g.vertices()))/(g.num_vertices()+1)
        ax.text(x-5,y,g.scalar.to_latex())

    ax.axis('equal')
    plt.close()
    return fig1
    #plt.show()

# Provides functions for displaying pyzx graphs in jupyter notebooks using d3

# make sure we get a fresh random seed
random_graphid = random.Random()

# def init_drawing() -> None:
#     if settings.mode not in ("notebook", "browser"): return
#
#     library_code = '<script type="text/javascript">\n'
#     for lib in ['d3.v5.min.inline.js']:
#         with open(os.path.join(settings.javascript_location, lib), 'r') as f:
#             library_code += f.read() + '\n'
#     library_code += '</script>'
#     display(HTML(library_code))

def draw_d3(
    g: BaseGraph[VT,ET],
    labels:bool=False, 
    scale:Optional[FloatInt]=None
    ) -> Any:
    
    # tracking global sequence can cause clashes if you restart the kernel without clearing ouput, so
    # use an 8-digit random alphanum instead.
    graph_id = ''.join(random_graphid.choice(string.ascii_letters + string.digits) for _ in range(8))

    minrow = min([g.row(v) for v in g.vertices()], default=0)
    maxrow = max([g.row(v) for v in g.vertices()], default=0)
    minqub = min([g.qubit(v) for v in g.vertices()], default=0)
    maxqub = max([g.qubit(v) for v in g.vertices()], default=0)

    if scale is None:
        scale = 800 / (maxrow-minrow + 2)
        if scale > 50: scale = 50
        if scale < 20: scale = 20

    node_size = 0.2 * scale
    if node_size < 2: node_size = 2

    w = (maxrow-minrow + 2) * scale
    h = (maxqub-minqub + 3) * scale

    nodes = [{'name': str(v),
              'x': (g.row(v)-minrow + 1) * scale,
              'y': (g.qubit(v)-minqub + 2) * scale,
              't': g.type(v),
              'phase': str(g.phase(v)).replace('(','').replace(')','').replace('0,0',''),
              }
             for v in g.vertices()]

    links = []
    for e in g.edges():
        s,t = g.edge_st(e)
        name = "{}, {}".format(str(s),str(t))
        eo = g.edge_object(e)
        phase = str(int(eo))
        ty = 3 if eo.type() == Edge.HadEdge else 4
        if ty == 4 and phase == '1':  # It is just a regular edge, so we are not gonna do anything fancy
            links.append({'source': str(s), 'target': str(t), 't':1})
        else:  # We are going to add a dummy H-box-like thing so that we don't have to draw multiple parallel wires
            x = (0.5*(g.row(s) + g.row(t))-minrow + 1) * scale
            y = (0.5*(g.qubit(s) + g.qubit(t))-minqub + 2) * scale
            if phase == '1': phase = ''
            nodes.append({'name': name, 'x': x, 'y': y, 't': ty, 'phase': phase})
            links.append({'source':str(s), 'target': name, 't':1})
            links.append({'source':name, 'target': str(t), 't':1})
    # links = [{'source': str(g.edge_s(e)),
    #           'target': str(g.edge_t(e)),
    #           't': str(g.edge_object(e).type()) } for e in g.edges()]
    graphj = json.dumps({'nodes': nodes, 'links': links})

    with open(os.path.join(settings.javascript_location, 'zx_viewer.inline.js'), 'r') as f:
        library_code = f.read() + '\n'

    text = """<div style="overflow:auto" id="graph-output-{id}"></div>
<script type="module">
var d3;
if (d3 == null) {{ d3 = await import("https://cdn.skypack.dev/d3@5"); }}
{library_code}
showGraph('#graph-output-{id}',
  JSON.parse('{graph}'), {width}, {height}, {scale},
  {node_size}, {labels});
</script>""".format(library_code=library_code,
                    id = graph_id,
                    graph = graphj, 
                    width=w, height=h, scale=scale, node_size=node_size,
                    labels='true' if labels else 'false')

    display(HTML(text))
    
