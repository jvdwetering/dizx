from typing import Optional, List
from .circuit import Circuit
from .circuit.gates import CZ, CX, SWAP, HAD, Z, X, S, NEG, Gate, GateWithControl

class DAG:
    """A class representing a directed acyclic graph (DAG) for a qudit Clifford circuit."""
    def __init__(self, node: Optional[Gate] = None) -> None:
        self.node: Optional[Gate] = node
        self.parents: List['DAG'] = []
        self.children: List['DAG'] = []
        self.ancestors: set['DAG'] = set()

    def add_child(self, child: 'DAG') -> None:
        """Add a child node to the DAG."""
        self.children.append(child)
        child.parents.append(self)
        child.ancestors.add(self)
        child.ancestors.update(self.ancestors)
    
    def remove_child(self, child: 'DAG') -> None:
        """Remove a child node from the DAG."""
        if child in self.children:
            self.children.remove(child)
            child.parents.remove(self)
            child.refresh_ancestors()
        else:
            raise ValueError("Child not found in children list.")
    
    def refresh_ancestors(self) -> None:
        """Refresh the ancestors set for this node and its children."""
        self.ancestors = set()
        for parent in self.parents:
            self.ancestors.add(parent)
            self.ancestors.update(parent.ancestors)
        for child in self.children:
            child.refresh_ancestors()
    
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
    
    def insert_single_qudit_gate_after(self, gate: Gate) -> None:
        """Insert a new single-qudit gate after this node in the DAG."""
        if self.node is None:
            raise ValueError("Both DAGs must have a node to insert after.")
        is_two_qubit = isinstance(self.node, GateWithControl)
        new_node = DAG(gate)
        if is_two_qubit:
            if self.node.target != gate.target and self.node.control != gate.target:
                raise ValueError("Cannot insert a single-qudit gate with a different target qudit than the current gate,", self.node, gate)
        elif self.node.target != gate.target:
            raise ValueError("Cannot insert a gate with a different target qudit,", self.node, gate)
        for child in self.children:
            if child.node.target == gate.target or (isinstance(child.node, GateWithControl) and child.node.control == gate.target):
                self.insert_between_child(new_node, child)
                break
        else:
            self.add_child(new_node)
        if is_two_qubit:
            for child in self.children:
                if isinstance(child.node, GateWithControl) and (child.node.target == gate.target or child.node.control == gate.target):
                    self.remove_child(child)
                    new_node.add_child(child)
    
    def insert_single_qudit_gate_before(self, gate: Gate) -> None:
        """Insert a new single-qudit gate before this node in the DAG."""
        if self.node is None:
            raise ValueError("Both DAGs must have a node to insert before.")
        if isinstance(gate, GateWithControl):
            raise ValueError("Cannot insert a two-qudit gate before a single-qudit gate.")
        is_two_qubit = isinstance(self.node, GateWithControl)
        new_node = DAG(gate)
        if is_two_qubit:
            if self.node.target != gate.target and self.node.control != gate.target:
                raise ValueError("Cannot insert a single-qudit gate with a different target qudit than the current gate,", self.node, gate)
        elif self.node.target != gate.target:
            raise ValueError("Cannot insert a gate with a different target qudit,", self.node, gate)
        for parent in self.parents:
            if parent.node.target == gate.target or (isinstance(parent.node, GateWithControl) and parent.node.control == gate.target):
                parent.insert_between_child(new_node, self)
                break
        else:
            # Can only occur if the ultimate parent is the root node
            parent = self.parents[0]
            while parent.node is not None:
                parent = parent.parents[0]
            parent.add_child(new_node)
            new_node.add_child(self)
        if is_two_qubit:
            # If we had a two-qubit parent, it might now no longer be a parent of this node
            for parent in self.parents:
                if isinstance(parent.node, GateWithControl) and (parent.node.target == gate.target or parent.node.control == gate.target):
                    parent.remove_child(self)
                    parent.add_child(new_node)
        
    def insert_two_qudit_gate_after(self, gate: Gate) -> None:
        """Insert a new two-qudit gate after this node in the DAG.
        Requires that both this DAG and the new_node have `not self.node is None`, so that it can check the qudit targets.
        We currently restrict to both gates having to act on the same two-qudits, so no three-qudit rewrites at the moment."""
        if self.node is None:
            raise ValueError("DAG must have a node to insert after.")
        if not isinstance(self.node, GateWithControl) or not isinstance(gate, GateWithControl):
            raise ValueError("Both gates must be two-qudit gates", self.node, gate)
        if {self.node.target, self.node.control} != {gate.target, gate.control}:
            raise ValueError("Cannot insert a two-qudit gate with different target and control qudits than the current gate,", self.node, gate)
        new_node = DAG(gate)
        for child in self.children:
            self.remove_child(child)  # Remove the child from the current DAG
            new_node.add_child(child)  # Add the child to the new node
    
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
    
    def pretty_print(self, level=0) -> str:
        indent = "  " * level
        s = f"{indent}{self.node}\n"
        for child in self.children:
            s += child.pretty_print(level + 4)
        return s
    
    def __str__(self) -> str:
        return self.pretty_print()


class CliffordSimplifier:
    """Takes in a qudit Clifford circuit and simplifies it by trying to normalise it
    by moving all the gates to the end of the circuit. This method should be usable
    by proof assistants to verify the correctness of the simplification. 
    Hence, it remembers all the steps it did and makes sure each step is a small atomic step."""
    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit.copy()
        self.create_dag()
        self.circuit_list: list[Circuit] = [] # Stores the intermediate steps of the simplification
        self.topological_sort() # Populates self.circuit_list with the initial circuit

    def create_dag(self) -> None:
        """Create a directed acyclic graph (DAG) from the circuit. This is a dependency graph
        where each node represents a gate and edges represent dependencies between gates, 
        in particular, a child will be a gate that happens after its parent on the same qubit."""
        self.dag = DAG()
        latest_gate: dict[int,Optional[DAG]] = {i: None for i in range(self.circuit.qudits)}
        index = 1
        for gate in self.circuit.gates:
            assert hasattr(gate, 'target'), "Gate must have a target qudit."
            gate.index = index
            index += 1
            node = DAG(gate)
            if isinstance(gate, GateWithControl):
                parent1 = latest_gate[gate.target]
                parent2 = latest_gate[gate.control]
                if parent1 is not None and parent2 is not None:
                    if parent1 is parent2:
                        parent1.add_child(node)
                    else:
                        if parent2 in parent1.ancestors: # We can only be a child if there are no dependencies in between
                            parent1.add_child(node)
                        elif parent1 in parent2.ancestors:
                            parent2.add_child(node)
                        else:
                            parent1.add_child(node)
                            parent2.add_child(node)
                elif parent1 is not None:
                    parent1.add_child(node)
                elif parent2 is not None:
                    parent2.add_child(node)
                else: # Only if both control and target are not set, we can just add the node as a child of the DAG
                    self.dag.add_child(node)
                latest_gate[gate.target] = node
                latest_gate[gate.control] = node
            else:
                if latest_gate[gate.target] is None: # First gate on this qudit
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
                assert dag.node is not None, "DAG node must not be None"
                if dag.node.index == 0:
                    dag.node.index = len(sorted_gates)+1 # This is needed to make all the gates 'unique' so that the equality check works correctly
                if dag.node is not None and dag.node not in sorted_gates:
                    gate = dag.node
                    sorted_gates.append(gate)
                    visited.append(gate)
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
        sorted_gates = [g.copy() for g in sorted_gates]
        c.gates = sorted_gates
        return c
    
    def simple_optimize(self) -> bool:
        """Runs a simple optimization on the circuit, combining gates, removing identity gates,
        pushing Paulis and double Hadamards."""
        success = False
        loops = 0
        while True:
            loops += 1
            success = success or self.combine_gates()
            success = success or self.remove_identity_gate()
            if success: 
                success = False
                continue  # We made progress, so we try again
            success = self.push_pauli()
            if success:
                success = False
                continue
            success = self.push_double_hadamard()
            if success:
                success = False
                continue
            break  # No more progress can be made
        return loops != 1 # Whether it applied at least one optimization step
    
    def single_qudit_optimize(self) -> bool:
        """Runs a single-qudit optimization on the circuit, running `simple_optimize` and then trying to apply
        the Euler decomposition rewrites."""
        success = False
        loops = 0
        while True:
            self.simple_optimize()
            success = self.euler_decomp()
            if success:
                success = False
                continue
            success = self.euler_decomp2()
            if success:
                success = False
                continue
            break # No more progress can be made
        return loops != 1 # Whether it applied at least one optimization step

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
        # TODO: Also do the case where there is a H^3 = H H^2, so that we only have to check H cases for all the other rewrites
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
    
    def euler_decomp(self) -> bool:
        """Tries to apply the substitution H;S;H -> S^-1;H;S^-1;X in some place in the circuit."""

        def try_decomp(dag: DAG) -> bool:
            """Tries to apply the substitution H;S;H -> S^-1;H;S^-1;X in the current DAG node."""
            if dag.node is not None and isinstance(dag.node, HAD) and dag.node.repetitions % 4 == 1:
                # It is a Hadamard so it should have at most one child
                if len(dag.children) == 0:
                    return False
                assert len(dag.children) == 1
                child = dag.children[0]
                assert child.node is not None, "Child node must not be None"
                if isinstance(child.node, S) and child.node.repetitions % self.circuit.dim == 1:
                    if len(child.children) == 0:
                        return False
                    assert len(child.children) == 1, "S gate should have at most one child"
                    grandchild = child.children[0]
                    if isinstance(grandchild.node, HAD) and grandchild.node.repetitions % 4 == 1:
                        # We have a H;S;H gate, so we can apply the substitution
                        new_s = S(child.node.target)
                        new_s.repetitions = -1
                        new_s2 = S(child.node.target)
                        new_s2.repetitions = -1
                        new_h = HAD(child.node.target)
                        new_x = X(child.node.target)
                        dag.node = new_s
                        child.node = new_h
                        grandchild.node = new_x
                        child.insert_between_child(DAG(new_s2), grandchild)
                        return True
            # If we reach here, we didn't apply the decomposition, so we try see if one of the children can push one

            for child in dag.children:
                if try_decomp(child):
                    return True
            return False

        success = try_decomp(self.dag)
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success
    
    def euler_decomp2(self) -> bool:
        """Tries to apply the substitution S^-1;H;S^-1 -> H;S;H;X^{-1}, but only if there is a H at the start, in order to avoid a loop with euler_decomp()."""

        def try_decomp(dag: DAG) -> bool:
            """Tries to apply the substitution H;S;H -> S^-1;H;S^-1;X in the current DAG node."""
            if dag.node is not None and isinstance(dag.node, S) and (dag.node.repetitions+1) % self.circuit.dim == 0: # It is S^{-1}
                # It is an S so it should have at most one child and one parent
                if len(dag.children) == 0:
                    return False
                assert len(dag.children) == 1
                child = dag.children[0]
                print("We have a potential match with", dag.node, "followed by", child.node)
                assert child.node is not None, "Child node must not be None"
                if len(dag.parents) ==0 or not isinstance(dag.parents[0].node, HAD):
                    pass
                elif isinstance(child.node, HAD) and child.node.repetitions % 4 == 1:
                    if len(child.children) == 0:
                        return False
                    assert len(child.children) == 1, "H gate should have at most one child"
                    grandchild = child.children[0]
                    if isinstance(grandchild.node, S) and (grandchild.node.repetitions+1) % self.circuit.dim == 0: # It is S^{-1}
                        # We have a S^{-1};H;S^{-1} gate, so we can apply the substitution
                        new_s = S(child.node.target)
                        new_h = HAD(child.node.target)
                        new_h2 = HAD(child.node.target)
                        new_x = X(child.node.target)
                        new_x.repetitions = -1
                        dag.node = new_h
                        child.node = new_s
                        grandchild.node = new_x
                        child.insert_between_child(DAG(new_h2), grandchild)
                        return True
            # If we reach here, we didn't apply the decomposition, so we try see if one of the children can push one

            for child in dag.children:
                if try_decomp(child):
                    return True
            return False

        success = try_decomp(self.dag)
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success
    
    def push_S_gate(self) -> bool:
        """Tries to push an S gate one step to the right in the circuit."""
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, S):
                # It is an S so it should have at most one child
                if len(dag.children) == 0:
                    return False
                assert len(dag.children) == 1
                child = dag.children[0]
                assert child.node is not None, "Child node must not be None"
                if isinstance(child.node, CZ) or isinstance(child.node, CX) and child.node.control == dag.node.target:
                    # We can push the S gate through a CZ/CX gate
                    new_s = dag.node.copy()
                    dag.node = child.node.copy()  # We replace the S with the CZ/CX gate
                    dag.merge(child)  # Remove the CZ/CX node
                    # We insert the S gate after the CZ/CX gate
                    dag.insert_single_qudit_gate_after(new_s)
                    return True
                
            # If we reach here, we didn't push the S gate, so we try see if one of the children can push one

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success

    def push_CZ_past_CX(self) -> bool:
        """Tries to push a CZ past a CX gate in the circuit.
        For instance: CZ^a(c,t); CX^b(c,t) = CX^b(c,t); CZ^a(c,t); S^{-2ab}(c) ; Z^{-ab}(c)"""
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, CZ):
                if len(dag.children) == 0:
                    return False
                cz = dag.node
                if len(dag.children) == 1 and dag.children[0].node is not None and isinstance(dag.children[0].node, CX):
                    # CZ has a single CX child, so they must have the same target and control qudits
                    child = dag.children[0]
                    cx = child.node
                    assert cx.target in (cz.target, cz.control) and cx.control in (cz.target, cz.control), "CZ and CX must have the same target and control qudits"
                    dag.node = cx.copy()  # We interchange the CZ with the CX gate
                    child.node = cz.copy()
                    new_s = S(cx.control)
                    new_s.repetitions = -2 * cz.repetitions * cx.repetitions
                    new_z = Z(cx.control)
                    new_z.repetitions = -cz.repetitions * cx.repetitions
                    child.insert_single_qudit_gate_after(new_z)
                    child.insert_single_qudit_gate_after(new_s)
                    return True
                elif len(dag.children) == 2:
                    # TODO: implement the case where the CZ and CX overlap on only one qudit
                    pass
                
            # If we reach here, we didn't push the CZ gate, so we try see if one of the children can push one

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)

        return success
    
    def push_SWAP(self) -> bool:
        """Tries to push a SWAP gate one step to the right in the circuit.
        We try to push it through CZ, CX, H, and S gates.
        Note that we do not push it through Paulis, since that would create a loop with `push_pauli()`."""
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, SWAP):
                swap = dag.node
                # It is a SWAP so it should have at most two children
                if len(dag.children) == 0:
                    return False
                if len(dag.children) == 1:
                    child = dag.children[0]
                    assert child.node is not None, "Child node must not be None"
                    if isinstance(child.node, CZ) or isinstance(child.node, CX):
                        # We can push the SWAP gate through a CZ/CX gate
                        # We just interchange the two gates
                        new_swap = swap.copy()
                        dag.node = child.node.copy()
                        child.node = new_swap
                        return True
                    if isinstance(child.node, (HAD, S)):
                        # We can push the SWAP gate through a H/S gate by changing its target
                        new_gate = child.node.copy()
                        new_gate.target = swap.target if new_gate.target != swap.target else swap.control
                        dag.insert_single_qudit_gate_before(new_gate)
                        dag.merge(child)  # Remove the child node
                        return True
                elif len(dag.children) == 2: # TODO: implement this
                    # We are only going to push one of the children, the other one will remain a child of the SWAP
                    child1 = dag.children[0]
                    child2 = dag.children[1]
                    assert child1.node is not None, "Child node must not be None"
                    if not isinstance(child1.node, (CZ,CX, HAD, S)):
                        child1 = dag.children[1]
                        child2 = dag.children[0]
                    assert child1.node is not None, "Child node must not be None"
                    if isinstance(child1.node, (HAD, S)):
                        new_gate = child1.node.copy()
                        new_gate.target = swap.target if new_gate.target != swap.target else swap.control
                        dag.insert_single_qudit_gate_before(new_gate)
                        dag.merge(child1)  # Remove the child node
                        # This last step might have added new children, but we need to verify that is correct
                        for grandchild in child2.children:
                            if grandchild in dag.children:
                                grandchild.parents.remove(dag)
                                dag.children.remove(grandchild)
                        return True
                    elif isinstance(child1.node, CZ) or isinstance(child1.node, CX): #TODO: implement this
                        pass

            # If we reach here, we didn't push the SWAP gate, so we try see if one of the children can push one
            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.circuit = self.topological_sort()
            self.circuit_list.append(self.circuit)
        
        return success