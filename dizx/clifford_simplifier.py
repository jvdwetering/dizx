from typing import Optional, List
from .circuit import Circuit
from .circuit.gates import CZ, CX, SWAP, HAD, Z, X, S, NEG, Gate, GateWithControl

class DAG:
    """A class representing a directed acyclic graph (DAG) for a qudit Clifford circuit."""
    def __init__(self, node: Optional[Gate] = None) -> None:
        self.node: Optional[Gate] = node
        self.parents: List['DAG'] = []
        self.children: List['DAG'] = []

    def add_child(self, child: 'DAG') -> None:
        """Add a child node to the DAG."""
        self.children.append(child)
        child.parents.append(self)
    
    def insert_between_child(self, new_node: 'DAG', child: Optional['DAG']=None) -> None:
        """Insert a new node between this node and a child node. If the child is None, it will just be added as a regular new child."""
        if child is None:
            self.add_child(new_node)
        elif child in self.children:
            self.children.remove(child)
            child.parents.remove(self)
            self.add_child(new_node)
            new_node.add_child(child)
        else:
            raise ValueError("Child not found in children list.")
    
    def merge(self, other: 'DAG') -> None:
        """Merge another DAG instance into this one. This assumes that the other DAG is a child of this one."""
        if other in self.children:
            self.children.remove(other)
            for child in other.children:
                child.parents.remove(other)
                self.add_child(child)
            other.children.clear()
            other.parents.clear()
        else:
            raise ValueError("Cannot merge DAGs that are not connected.")
    
    def __repr__(self) -> str:
        return f"DAG(node={self.node}, children=[{', '.join(repr(child) for child in self.children)}])"


class CliffordSimplifier:
    """Takes in a qudit Clifford circuit and simplifies it by trying to normalise it
    by moving all the gates to the end of the circuit. This method should be usable
    by proof assistants to verify the correctness of the simplification. 
    Hence, it remembers all the steps it did and makes sure each step is a small atomic step."""
    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit.copy()
        self.create_dag()
        self.circuit_list: list[Circuit] = [self.circuit] # Stores the intermediate steps of the simplification

    def create_dag(self) -> None:
        """Create a directed acyclic graph (DAG) from the circuit. This is a dependency graph
        where each node represents a gate and edges represent dependencies between gates, 
        in particular, a child will be a gate that happens after its parent on the same qubit."""
        self.dag = DAG()
        latest_gate: dict[int,Optional[DAG]] = {i: None for i in range(self.circuit.qudits)}

        for gate in self.circuit.gates:
            assert hasattr(gate, 'target'), "Gate must have a target qudit."
            node = DAG(gate)
            if isinstance(gate, GateWithControl):
                if latest_gate[gate.control] is not None:
                    latest_gate[gate.control].add_child(node)
                    latest_gate[gate.control] = node
                if latest_gate[gate.target] is not None:
                    if node not in latest_gate[gate.target].children:
                        latest_gate[gate.target].add_child(node)
                        latest_gate[gate.target] = node
                if latest_gate[gate.control] is None and latest_gate[gate.target] is None:
                    # Only if both control and target are not set, we can just add the node as a child of the DAG
                    self.dag.add_child(node)
                    latest_gate[gate.target] = node
                    latest_gate[gate.control] = node
            else:
                if latest_gate[gate.target] is None: # First gate on this qudit
                    latest_gate[gate.target] = node
                    self.dag.add_child(node)
                else:
                    latest_gate[gate.target].add_child(node) # type: ignore
                    latest_gate[gate.target] = node


    def topological_sort(self) -> Circuit:
        """Converts the DAG back into a list of gates in topological order."""
        visited: list[Gate] = []
        sorted_gates: list[Gate] = []
        
        def sort_step(dag: DAG) -> bool:
            """Recursively sorts the DAG, ensuring that all parents are visited before its added to the list."""
            results = False
            if all(p.node in visited for p in dag.parents if p.node is not None):
                if dag.node.index == 0:
                    dag.node.index = len(sorted_gates) # This is needed to make all the gates 'unique' so that the equality check works correctly
                if dag.node is not None and dag.node not in sorted_gates:
                    sorted_gates.append(dag.node)
                    visited.append(dag.node)
                    results = True
                results = results or any(sort_step(child) for child in dag.children)
            return results
        
        while True:
            made_progress = False
            for child in self.dag.children:
                if sort_step(child):
                    made_progress = True
            if not made_progress:
                break

        c = self.circuit.copy()
        c.gates = sorted_gates
        return c
    
    def simple_optimize(self) -> Circuit:
        """Runs a simple optimization on the circuit, combining gates and removing identity gates."""
        success = False
        while True:
            success = success or self.combine_gates()
            success = success or self.remove_identity_gate()
            if success: 
                success = False
                continue  # We made progress, so we try again
            success = self.push_pauli()
            if success:
                success = False
                continue
            break  # No more progress can be made
        return self.circuit

    def combine_gates(self) -> bool:
        """Combines gates in the circuit to reduce the number of gates."""
        def try_merge(dag: DAG) -> bool:
            """Tries to merge the current DAG node with its children if they are the same gate."""
            if dag.node is not None and len(dag.children) == 1:
                child = dag.children[0]
                if child.node is not None and child.node.name == dag.node.name:
                    dag.node.merge(child.node)  # Combines the two gates into one
                    dag.merge(child)
                    return True
            for child in dag.children:
                if try_merge(child):
                    return True
            return False
        
        success = False
        while try_merge(self.dag):
            success = True
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success
    
    def remove_identity_gate(self) -> bool:
        """Removes identity gates from the circuit."""
        def is_identity(gate: Gate) -> bool:
            if isinstance(gate, (Z,X,S, CZ, CX)):
                return gate.repetitions % self.circuit.dim == 0
            if isinstance(gate, HAD):
                return gate.repetitions % 4 == 0
            if isinstance(gate, (NEG,SWAP)):
                return gate.repetitions % 2 == 0
            return False
        
        def try_remove_identity(dag: DAG) -> bool:
            """Tries to remove identity gates from the current DAG node."""
            for child in dag.children:
                if child.node is not None and is_identity(child.node):
                    dag.merge(child)  # Remove the child node
                    return True
                elif try_remove_identity(child):
                    return True
            return False
        
        success = try_remove_identity(self.dag) # We removed something, so time to stop
        
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success
    
    def push_pauli(self) -> bool:
        """Tries to push a Pauli gate one step to the right in the circuit."""
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, (Z, X)):
                # It is a Pauli so it should have at most one child
                if len(dag.children) == 0: # It is already at the end of the circuit
                    return False
                assert len(dag.children) == 1
                child = dag.children[0]
                assert child.node is not None, "Child node must not be None"
                if isinstance(child.node, (Z, X)): # Child is also a Pauli. 
                    # We assume gates are maximally merged, so it must be the other Pauli
                    # We normalize so that the Z is always before the X
                    if isinstance(dag.node, X) and isinstance(child.node, Z):
                        # We interchange the two gates
                        dag.node, child.node = child.node, dag.node
                        return True
                elif isinstance(child.node, HAD):
                    # Hadamard gate, we can push the Pauli through it
                    child.node.repetitions = child.node.repetitions % 4 # We assume repetitions is not 0 now.
                    is_h_adjoint = child.node.repetitions == 3
                    if child.node.repetitions == 2: 
                        dag.node.repetitions = -dag.node.repetitions
                    elif isinstance(dag.node, Z):
                        new_x = X(dag.node.target)
                        new_x.repetitions = dag.node.repetitions if not is_h_adjoint else -dag.node.repetitions
                        dag.node = new_x
                    elif isinstance(dag.node, X):
                        new_z = Z(dag.node.target)
                        new_z.repetitions = -dag.node.repetitions if not is_h_adjoint else dag.node.repetitions
                        dag.node = new_z
                    dag.node, child.node = child.node, dag.node
                    return True
                elif isinstance(child.node, S): # Commuting Pauli with S gate
                    if isinstance(dag.node, Z): # Just commute them past each other
                        dag.node, child.node = child.node, dag.node
                    elif isinstance(dag.node, X): # We get a new Z gate in between
                        new_z = Z(dag.node.target)
                        new_z.repetitions = dag.node.repetitions * child.node.repetitions
                        node = DAG(new_z)
                        dag.node, child.node = child.node, dag.node
                        dag.insert_between_child(node, child) # Insert the new Z gate between the two
                    return True
                elif isinstance(child.node, (CZ, CX, SWAP)):
                    q = dag.node.target
                    target = child.node.target
                    control = child.node.control
                    control_child = None
                    target_child = None
                    for child2 in child.children:
                        if (child2.node is not None and child2.node.target == control 
                            or isinstance(child2.node, GateWithControl) and child2.node.control == control):
                            control_child = child2
                        if (child2.node is not None and child2.node.target == target or 
                            isinstance(child2.node, GateWithControl) and child2.node.control == target):
                            target_child = child2
                    double_child = control_child is not None and control_child is target_child
                    two_qubit_gate = child.node.copy()
                    pauli = dag.node.copy()
                    dag.merge(child) # Remove the child node
                    dag.node = two_qubit_gate # Put the two-qubit gate in place of the Pauli
                    if isinstance(child.node, SWAP): # We assume the SWAP only occurs as a single repetition
                        new_target = target if control == q else control
                        pauli.target = new_target
                        dag.insert_between_child(DAG(pauli), target_child if new_target == target else control_child)
                        if double_child: # The child is on both the target and control, so it should still be a child of the SWAP
                            assert control_child is not None
                            dag.add_child(control_child)
                        return True
                    elif isinstance(child.node, CZ):
                        if isinstance(pauli, Z): # We can just push the Z gate through the CZ
                            dag.insert_between_child(DAG(pauli), control_child if control == q else target_child)
                            if double_child: # The child is on both the target and control, so it should still be a child of the CZ
                                assert control_child is not None
                                dag.add_child(control_child)
                        elif isinstance(pauli, X): # We get a new Z gate in between
                            new_target = target if control == q else control
                            new_z = Z(new_target)
                            new_z.repetitions = pauli.repetitions * two_qubit_gate.repetitions
                            node_old = DAG(pauli)
                            node_new = DAG(new_z)
                            if not double_child:
                                dag.insert_between_child(node_new, target_child if new_target == target else control_child)
                                dag.insert_between_child(node_old, control_child if new_target == target else target_child)
                            else:
                                assert control_child is not None
                                dag.children.remove(control_child)
                                control_child.parents.remove(dag)
                                dag.add_child(node_new)
                                dag.add_child(node_old)
                                node_old.add_child(control_child)
                                node_new.add_child(control_child)
                        return True
                    elif isinstance(child.node, CX):
                        is_on_control = child.node.control == q
                        if ((is_on_control and isinstance(pauli, Z)) or 
                            (not is_on_control and isinstance(pauli, X))): # In these cases we can just commute the Pauli through the CX
                            dag.insert_between_child(DAG(pauli), control_child if control == q else target_child)
                            if double_child: # The child is on both the target and control, so it should still be a child of the CX
                                assert control_child is not None
                                dag.add_child(control_child)
                        else:
                            # We either have a Z on the target or an X on the control
                            new_target = target if is_on_control else control
                            new_pauli = X(new_target) if isinstance(pauli, X) else Z(new_target)
                            new_pauli.repetitions = pauli.repetitions * two_qubit_gate.repetitions
                            if isinstance(new_pauli, Z):
                                new_pauli.repetitions = -new_pauli.repetitions
                            node_old = DAG(pauli)
                            node_new = DAG(new_pauli)
                            if not double_child:
                                dag.insert_between_child(node_new, target_child if new_target == target else control_child)
                                dag.insert_between_child(node_old, control_child if new_target == target else target_child)
                            else:
                                assert control_child is not None
                                dag.children.remove(control_child)
                                control_child.parents.remove(dag)
                                dag.add_child(node_new)
                                dag.add_child(node_old)
                                node_old.add_child(control_child)
                                node_new.add_child(control_child)
                        return True
            # If we reach here, we didn't push the Pauli gate, so we try see if one of the children can push a Pauli

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success
    
    def push_double_hadamard(self) -> bool:
        """Tries to push a Hadamard^2 = NEG gate one step to the right in the circuit.
        Note that it does not push past Paulis in order for the full strategy to terminate."""
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, HAD) and dag.node.repetitions %2 == 0:
                # It is a Hadamard so it should have at most one child
                if len(dag.children) == 0:
                    return False
                assert len(dag.children) == 1
                child = dag.children[0]
                assert child.node is not None, "Child node must not be None"
                if isinstance(child.node, HAD):
                    # We merge the H^2 with its adjacent H gates
                    dag.node.repetitions = (dag.node.repetitions + child.node.repetitions) % 4
                    dag.merge(child)
                    return True
                elif isinstance(child.node, (Z, X)):
                    pass # We just stop here, we don't push past Paulis
                elif isinstance(child.node, (CZ, CX)): # Pushing H^2 through CX/CX just makes them their adjoint
                    child.node.repetitions = -child.node.repetitions
                    dag.node, child.node = child.node, dag.node
                    return True
                elif isinstance(child.node, SWAP):
                    # Pushing H^2 through SWAP just changes the target of the H^2
                    q = dag.node.target
                    target = child.node.target
                    control = child.node.control
                    control_child = None
                    target_child = None
                    for child2 in child.children:
                        if (child2.node is not None and child2.node.target == control 
                            or isinstance(child2.node, GateWithControl) and child2.node.control == control):
                            control_child = child2
                        if (child2.node is not None and child2.node.target == target or 
                            isinstance(child2.node, GateWithControl) and child2.node.control == target):
                            target_child = child2
                    double_child: bool = control_child is not None and control_child is target_child
                    swap = child.node.copy()
                    H2 = dag.node.copy()
                    dag.merge(child) # Remove the SWAP node
                    dag.node = swap # Put the two-qubit gate in place of the H^2
                    new_target = target if control == q else control
                    H2.target = new_target
                    dag.insert_between_child(DAG(H2), target_child if new_target == target else control_child)
                    if double_child: # The child is on both the target and control, so it should still be a child of the SWAP
                        assert control_child is not None
                        dag.add_child(control_child)
                    return True
                elif isinstance(child.node, S): # Commuting H^2 with S gate
                    # Pushing H^2 through S creates a Z^{-1} gate
                    # Or equivalently H^2 ; S = S ; H^2 ; Z 
                    # Which is useful for us, since we want the Paulis after the H^2 anyway
                    new_z = Z(dag.node.target)
                    new_z.repetitions = child.node.repetitions
                    dag.node, child.node = child.node, dag.node
                    if len(child.children) == 0:
                        child2 = None
                    else:
                        assert len(child.children) == 1, "S gate should have at most one child"
                        child2 = child.children[0]
                    child.insert_between_child(DAG(new_z), child2)
                    return True
            # If we reach here, we didn't push the H^2 gate, so we try see if one of the children can push one

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success