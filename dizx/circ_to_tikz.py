class Gate(object):
	tikz_class = 'gate'
	label = ''

class GateOneQubit(Gate):
	def __init__(self, target):
		self.target = target

class GateTwoQubit(Gate):
	tikz_class = 'medium box'
	def __init__(self, target1, target2):
		self.target1 = target1
		self.target2 = target2

class S(GateOneQubit): 
	label = '$S$'

class H(GateOneQubit): 
	label = '$H$'

class CZ(GateTwoQubit): pass

class BoxOneQubit(GateOneQubit): 
	tikz_class = 'gate'
	name = ''
	def __init__(self, param1, param2, target):
		super().__init__(target)
		self.param1 = param1
		self.param2 = param2
		self.label = f'${self.name}_{{{self.param1}, {self.param2}}}$'

class BoxTwoQubit(GateTwoQubit):
	tikz_class = 'medium box'
	name = ''
	def __init__(self, param1, param2, target1, target2):
		super().__init__(target1, target2)
		self.param1 = param1
		self.param2 = param2
		self.label = f'${self.name}_{{{self.param1}, {self.param2}}}$'

class BoxA(BoxOneQubit):
	name = 'A'
class BoxB(BoxTwoQubit):
	name = 'B'
class BoxC(BoxOneQubit):
	name = 'C'
class BoxD(BoxTwoQubit):
	name = 'D'
class BoxE(BoxOneQubit):
	name = 'E'
class BoxF(BoxOneQubit):
	name = 'F'

gate_table = {
	's': S,
	'h': H,
	'a': BoxA,
	'c': BoxC,
	'e': BoxE,
	'f': BoxF,
	'cz': CZ,
	'b': BoxB,
	'd': BoxD

}

tikz_format = r"""\begin{tikzpicture}
	\begin{pgfonlayer}{nodelayer}
	NODES_HERE
	\end{pgfonlayer}
	\begin{pgfonlayer}{edgelayer}
	EDGES_HERE
	\end{pgfonlayer}
\end{tikzpicture}

"""

class Circuit(object):
	spacing_horizontal = 2
	spacing_vertical = 1.5
	def __init__(self):
		self.gates: list[Gate] = []

	def add_gate(self, gate):
		self.gates.append(gate)

	def to_tikz(self):
		qubit_pos = []
		node_defs = []
		edge_defs = []
		spacing_vertical = self.spacing_vertical
		spacing_horizontal = self.spacing_horizontal
		for g in self.gates:
			if isinstance(g, GateOneQubit):
				if len(qubit_pos) < g.target:
					qubit_pos.extend([0]*(g.target-len(qubit_pos)))
				qubit_pos[g.target-1] = qubit_pos[g.target-1] + 1
				x = qubit_pos[g.target-1]*spacing_horizontal
				y = (g.target-1)*spacing_vertical
				node_defs.append(f"\\node [style={g.tikz_class}] ({len(node_defs)}) at ({x}, {y}) {{{g.label}}};")
			elif isinstance(g, CZ): # We need to treat this case separately, due to its unique look
				t1 = min([g.target1, g.target2])
				t2 = max([g.target1, g.target2])
				if t1 == t2: raise ValueError(f"CZ gate has two equal targets: {g.name} at {t1} {t2}")
				if len(qubit_pos) < t2:
					qubit_pos.extend([0]*(t2-len(qubit_pos)))
				x = max([qubit_pos[t1-1],qubit_pos[t2-1]]) + 1
				qubit_pos[t1-1] = x
				qubit_pos[t2-1] = x
				x = x*spacing_horizontal
				t1 = (t1-1)*spacing_vertical
				t2 = (t2-1)*spacing_vertical
				v = len(node_defs)
				node_defs.append(f"\\node [style=cnot ctrl] ({v}) at ({x}, {t1}) {{}};")
				node_defs.append(f"\\node [style=cnot ctrl] ({v+1}) at ({x}, {t2}) {{}};")
				edge_defs.append(f"\\draw ({v}) to ({v+1});")
			elif isinstance(g, GateTwoQubit): # All the other two qubit gates
				t1 = min([g.target1, g.target2])
				t2 = max([g.target1, g.target2])
				if t1 == t2: raise ValueError(f"Multi-gate gate has two equal targets: {g.name} at {t1} {t2}")
				avg = (t1+t2)/2
				if len(qubit_pos) < t2:
					qubit_pos.extend([0]*(t2-len(qubit_pos)))
				x = max([qubit_pos[t1-1],qubit_pos[t2-1]]) + 1
				qubit_pos[t1-1] = x
				qubit_pos[t2-1] = x
				x = x*spacing_horizontal
				avg = (avg-1)*spacing_vertical
				node_defs.append(f"\\node [style={g.tikz_class}] ({len(node_defs)}) at ({x}, {avg}) {{{g.label}}};")
			else:
				raise ValueError(f"Unsupported gate {str(g)}")
		width = (max(qubit_pos)+1)*spacing_horizontal
		for i in range(len(qubit_pos)): # Add boundary nodes at start and end of qubit lines
			v = len(node_defs)
			node_defs.append(f"\\node [style=none] ({v}) at (0, {i*spacing_vertical}) {{}};")
			node_defs.append(f"\\node [style=none] ({v+1}) at ({width}, {i*spacing_vertical}) {{}};")
			edge_defs.append(f"\\draw ({v}) to ({v+1});")
		output = tikz_format.replace("NODES_HERE", "\n\t".join(node_defs))
		output = output.replace("EDGES_HERE", "\n\t".join(edge_defs))
		return output


def file_to_circ(fname):
	with open(fname,'r') as f:
		data = f.read()
	i = data.find('BEGIN')
	if i == -1: raise ValueError("Could not find starting point BEGIN in file")
	j = data.find('END',i)
	if j == -1: raise ValueError("Could not find end point of circuit END")
	data = data[i+5:j].strip()

	c = Circuit()

	name = ""
	for line in data.splitlines():
		try:
			line = line.strip()
			if len(line) == 0: continue # Blank space
			if line.startswith('#'): continue  # Line is commented out
			if line.find(' ') == -1: raise ValueError(f"Malformed line, expected a space: {line}")
			arguments = line.split(' ')
			command = arguments[0]
			arguments = arguments[1:]
			command = command.lower()
			if command == 'name': 
				name = arguments
				continue

			if command in ('h','s'):
				if len(arguments) != 1: raise ValueError(f"Wrong number of parameters in line {line}")
				target = int(arguments[0])
				c.add_gate(gate_table[command](target))
			if command == 'cz':
				if len(arguments) != 2: raise ValueError(f"Wrong number of parameters in line {line}")
				target1, target2 = int(arguments[0]), int(arguments[1])
				c.add_gate(CZ(target1, target2))

			if command in ('a', 'c', 'e', 'f'):
				if len(arguments) != 3: raise ValueError(f"Wrong number of parameters in line {line}")
				param1, param2 = arguments[0], arguments[1]
				target = int(arguments[2])
				c.add_gate(gate_table[command](param1, param2, target))

			if command in ('b', 'd'):
				if len(arguments) != 4: raise ValueError(f"Wrong number of parameters in line {line}")
				param1, param2 = arguments[0], arguments[1]
				target1, target2 = int(arguments[2]), int(arguments[3])
				c.add_gate(gate_table[command](param1, param2, target1, target2))

		except Exception as e:
			raise ValueError(f"Error in processing line {line}:\n" + str(e))

	return c

if __name__ == '__main__':
	fname = 'test.qc'
	c = file_to_circ(fname)
	tikz = c.to_tikz()
	with open('test.tikz', 'w') as f:
		f.write(tikz)