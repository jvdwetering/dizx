from typing import Optional, List

import ipywidgets as widgets
from IPython.display import display, clear_output

from sympy import mod_inverse

from .circuit import Circuit
from .circuit.gates import CZ, CX, SWAP, HAD, MUL, Z, X, S, Gate, GateWithControl
from . import symplectic


class SemanticsException(Exception):
    """Exception type for telling the user that the rewrite that was applied 
    has not preserved the semantics of the circuit."""

class DAG:
    """A class representing a directed acyclic graph (DAG) for a qudit Clifford circuit."""
    def __init__(self, node: Optional[Gate] = None) -> None:
        self.node: Optional[Gate] = node
        self.parents: List['DAG'] = []
        self.children: List['DAG'] = []
        self.ancestors: set['DAG'] = set()
    
    def copy(self) -> 'DAG':
        d = DAG(self.node.copy() if self.node is not None else None)
        self.my_copy = d # Nodes might share children, so we remember if we already copied a node, and reuse that.
        # d.parents = [p for p in self.parents]
        # d.refresh_ancestors()
        for child in self.children:
            if hasattr(child, 'my_copy') and child.my_copy is not None:
                d.add_child(child.my_copy)
            else:
                d.add_child(child.copy())
        
        if self.node is None: # We are the top level node, and hence we are now done with the copy operation
            self._reset_copied_flag()
        return d
    
    def _reset_copied_flag(self) -> None:
        self.my_copy = None
        for child in self.children:
            child._reset_copied_flag()

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
    
    def find_first_descendant_on_qudit(self, index:int) -> Optional['DAG']:
        """Finds the first DAG among the children and grandchildren whose node has a target or control equal to index.
        Needed if we want to inject for instance a single-qudti gate after the current two-qudit gate.
        This is only really meaningful if self.node has a target or control equal to index."""
        candidates: list['DAG'] = []
        for child in self.children:
            if child.node is None: continue
            g = child.node
            if g.target == index or isinstance(g,GateWithControl) and g.control == index:
                return child
            candidate = child.find_first_descendant_on_qudit(index)
            if candidate is not None: candidates.append(candidate)
        
        if not candidates: return None
        candidate = candidates[0]
        if len(candidates) == 1: return candidate
        for can in candidates[1:]:
            if can in candidate.ancestors: # We want the earliest candidate in the circuit, which will be the ancestor of the other candidates
                candidate = can
        return candidate
    
    def find_first_ancestor_on_qudit(self, index:int) -> 'DAG':
        """Finds the first DAG among the children and grandchildren whose node has a target or control equal to index.
        Needed if we want to inject for instance a single-qudti gate after the current two-qudit gate.
        This is only really meaningful if self.node has a target or control equal to index."""
        candidates: list['DAG'] = []
        for parent in self.parents:
            if parent.node is None: 
                candidates.append(parent)
                continue
            g = parent.node
            if g.target == index or isinstance(g,GateWithControl) and g.control == index:
                return parent
            candidate = parent.find_first_ancestor_on_qudit(index)
            candidates.append(candidate)
        
        if not candidates: return self # We are the root node
        candidate = candidates[0]
        if len(candidates) == 1: return candidate
        for can in candidates[1:]:
            if candidate in can.ancestors: # We want the latest candidate, so if our current candidate is an ancestor, we should switch
                candidate = can
        return candidate
    
    def insert_between_child(self, new_node: 'DAG', child: Optional['DAG']=None) -> None:
        """Insert a new node between this node and a child node. If the child is None, it will just be added as a regular new child,
        and it will look for ancestors that that child should be the parent of (if any)."""
        if child is None:
            desc_target = self.find_first_descendant_on_qudit(new_node.node.target)
            if isinstance(new_node.node,GateWithControl):
                desc_control = self.find_first_descendant_on_qudit(new_node.node.control)
            else:
                desc_control = None
            self.add_child(new_node)
            if desc_target is not None:
                new_node.add_child(desc_target)
            if desc_control is not None: # TODO: Actually figure out what needs to happen if we add a two-qudit gate.
                new_node.add_child(desc_control)
        elif child in self.children:
            self.children.remove(child)
            child.parents.remove(self)
            self.add_child(new_node)
            new_node.add_child(child)
        else:
            raise ValueError("Child not found in children list.")
    
    def insert_single_qudit_gate_after(self, gate: Gate) -> 'DAG':
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
            self.insert_between_child(new_node)
        if is_two_qubit:
            for child in self.children:
                if isinstance(child.node, GateWithControl) and (child.node.target == gate.target or child.node.control == gate.target):
                    self.remove_child(child)
                    new_node.add_child(child)
        return new_node
    
    def insert_single_qudit_gate_before(self, gate: Gate) -> 'DAG':
        """Insert a new single-qudit gate before this node in the DAG."""
        if self.node is None:
            raise ValueError("Both DAGs must have a node to insert before.")
        if isinstance(gate, GateWithControl):
            raise ValueError("Cannot insert a two-qudit gate before a single-qudit gate.")
        is_two_qubit = isinstance(self.node, GateWithControl)
        ancestor = self.find_first_ancestor_on_qudit(gate.target)
        return ancestor.insert_single_qudit_gate_after(gate)
        # new_node = DAG(gate)
        # if is_two_qubit:
        #     if self.node.target != gate.target and self.node.control != gate.target:
        #         raise ValueError("Cannot insert a single-qudit gate with a different target qudit than the current gate,", self.node, gate)
        # elif self.node.target != gate.target:
        #     raise ValueError("Cannot insert a gate with a different target qudit,", self.node, gate)
        # for parent in self.parents:
        #     if parent.node.target == gate.target or (isinstance(parent.node, GateWithControl) and parent.node.control == gate.target):
        #         parent.insert_between_child(new_node, self)
        #         break
        # else:
        #     # Can only occur if the ultimate parent is the root node
        #     parent = self.parents[0]
        #     while parent.node is not None:
        #         parent = parent.parents[0]
        #     parent.add_child(new_node)
        #     new_node.add_child(self)
        # if is_two_qubit:
        #     # If we had a two-qubit parent, it might now no longer be a parent of this node
        #     for parent in self.parents:
        #         if isinstance(parent.node, GateWithControl) and (parent.node.target == gate.target or parent.node.control == gate.target):
        #             parent.remove_child(self)
        #             parent.add_child(new_node)
        # return new_node
        
    def insert_two_qudit_gate_after(self, gate: Gate) -> 'DAG':
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
        self.add_child(new_node)
        for child in self.children.copy():
            if child is new_node: continue
            self.remove_child(child)  # Remove the child from the current DAG
            new_node.add_child(child)  # Add the child to the new node
        return new_node
        
    
    def merge(self, other: 'DAG') -> None:
        """Merge another DAG instance into this one. This assumes that the other DAG is a child of this one."""
        if other in self.children:
            self.children.remove(other)
            for child in other.children:
                child.parents.remove(other)
                self.add_child(child)
            other.parents.remove(self)
            if len(other.parents) > 0: # Only possible if other is a two-qubit gate
                # The parents of other should definitely be our parents, since other was a child
                # However, our current parents might no longer be valid, so we have to check that

                # I think this is the right condition: it should not be a parent if any other of our new parents is a direct child of it
                # This happens precisely when the ancestor set is a superset of the ancestors + the parent itself
                parents_to_remove = [p for p in self.parents if any(p2.ancestors.issuperset(p.ancestors.union({p})) for p2 in other.parents)]
                for p in parents_to_remove:
                    p.remove_child(self)
                for p in other.parents:
                    p.children.remove(other)
                    if p not in self.parents:
                        p.add_child(self)
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
        self.circuit: Circuit = circuit.copy()
        self.dags: list[DAG] = []
        self.steps_done: list[int] = []
        self.circuit_list: list[Circuit] = [] # Stores the intermediate steps of the simplification
        self.max_index = 1
        self.create_dag()
        self.topological_sort() # Populates self.circuit_list with the initial circuit
        self.check_semantics_each_step = False
        self.verbose = False

    def create_dag(self) -> None:
        """Create a directed acyclic graph (DAG) from the circuit. This is a dependency graph
        where each node represents a gate and edges represent dependencies between gates, 
        in particular, a child will be a gate that happens after its parent on the same qubit."""
        self.dag = DAG()
        latest_gate: dict[int,Optional[DAG]] = {i: None for i in range(self.circuit.qudits)}
        for gate in self.circuit.gates:
            assert hasattr(gate, 'target'), "Gate must have a target qudit."
            gate.index = self.max_index
            self.max_index += 1
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
        self.circuit_list.append(self.circuit.copy())
        self.dags.append(self.dag.copy())


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
                    dag.node.index = self.max_index # This is needed to make all the gates 'unique' so that the equality check works correctly
                    self.max_index += 1
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
        c.gates = [g.copy() for g in sorted_gates]
        return c
    
    def update_circuit(self, description:str="") -> None:
        self.circuit = self.topological_sort()
        self.circuit_list.append(self.circuit)
        self.steps_done.append(description)
        if self.verbose:
            print(description)
            print(self.circuit)
        self.dags.append(self.dag.copy())
        if self.check_semantics_each_step:
            mat1 = self.circuit_list[-1].to_symplectic_matrix()
            mat2 = self.circuit_list[-2].to_symplectic_matrix()
            if not symplectic.compare_matrices(mat1, mat2, self.circuit.dim):
                raise SemanticsException("Semantics were not preserved by the last rewrite applied")
    
    def display_widget(self) -> None:
        w = StepperWidget(self)
        w.show()
    
    def simple_optimize(self) -> bool:
        """Runs a simple optimization on the circuit, combining gates, removing identity gates,
        and trying the following 'push gate to the right strategies':
        - pushing Pauli gates
        - pushing double Hadamard gates
        - pushing SWAP gates
        - commuting S past CX and CZ controls
        - commuting S past CX targets
        - commuting CZ past CX if they act on the same qudits."""
        success = False
        loops = 0
        while True:
            loops += 1
            success = success or self.combine_gates()
            success = success or self.remove_identity_gate()
            if success: 
                success = False
                continue  # We made progress, so we try again
            if self.push_pauli(): continue
            if self.push_double_hadamard(): continue
            if self.push_SWAP(): continue
            if self.push_S_gate(): continue
            if self.push_S_past_CX(): continue
            if self.push_CZ_past_CX(): continue
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
                    if not isinstance(dag.node, GateWithControl) or (
                        dag.node.target == child.node.target and dag.node.control == child.node.control) or (
                            isinstance(dag.node, CZ) and dag.node.target in (child.node.target,child.node.control) 
                            and dag.node.control in (child.node.target,child.node.control)):
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
            self.update_circuit("Combine gates")

        return success
    
    def remove_identity_gate(self) -> bool:
        """Removes identity gates from the circuit."""
        def is_identity(gate: Gate) -> bool:
            if isinstance(gate, (Z,X,S, CZ, CX)):
                return gate.repetitions % self.circuit.dim == 0
            if isinstance(gate, HAD):
                return gate.repetitions % 4 == 0
            if isinstance(gate, SWAP):
                return gate.repetitions % 2 == 0
            if isinstance(gate, MUL):
                return gate.mult_value % self.circuit.dim == 1
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
            self.update_circuit("Remove identity")

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
                    two_qubit_gate = child.node.copy()
                    pauli = dag.node.copy()
                    dag.node = two_qubit_gate # Put the two-qubit gate in place of the Pauli
                    dag.merge(child) # Remove the child node
                    q = pauli.target
                    target = two_qubit_gate.target
                    control = two_qubit_gate.control
                    control_child = None
                    target_child = None
                    # for grandchild in child.children:
                    #     if (grandchild.node is not None and grandchild.node.target == control 
                    #         or isinstance(grandchild.node, GateWithControl) and grandchild.node.control == control):
                    #         control_child = grandchild
                    #     if (grandchild.node is not None and grandchild.node.target == target or 
                    #         isinstance(grandchild.node, GateWithControl) and grandchild.node.control == target):
                    #         target_child = grandchild
                    # double_grandchild = control_child is not None and control_child is target_child and  control_child is not target_child
                    
                    
                    if isinstance(two_qubit_gate, SWAP): # We assume the SWAP only occurs as a single repetition
                        new_target = target if control == q else control
                        pauli.target = new_target
                        dag.insert_single_qudit_gate_after(pauli)
                        # dag.insert_between_child(DAG(pauli), target_child if new_target == target else control_child)
                        # if double_grandchild: # The child is on both the target and control, so it should still be a child of the SWAP
                        #     assert control_child is not None
                        #     dag.add_child(control_child)
                        return True
                    elif isinstance(two_qubit_gate, CZ):
                        if isinstance(pauli, Z): # We can just push the Z gate through the CZ
                            dag.insert_single_qudit_gate_after(pauli)
                            # dag.insert_between_child(DAG(pauli), control_child if control == q else target_child)
                            # if double_grandchild: # The child is on both the target and control, so it should still be a child of the CZ
                            #     assert control_child is not None
                            #     dag.add_child(control_child)
                        elif isinstance(pauli, X): # We get a new Z gate in between
                            new_target = target if control == q else control
                            new_z = Z(new_target)
                            new_z.repetitions = pauli.repetitions * two_qubit_gate.repetitions
                            # node_old = DAG(pauli)
                            # node_new = DAG(new_z)
                            dag.insert_single_qudit_gate_after(pauli)
                            dag.insert_single_qudit_gate_after(new_z)
                            # if not double_grandchild:
                            #     dag.insert_between_child(node_new, target_child if new_target == target else control_child)
                            #     dag.insert_between_child(node_old, control_child if new_target == target else target_child)
                            # else:
                            #     assert control_child is not None
                            #     dag.children.remove(control_child)
                            #     control_child.parents.remove(dag)
                            #     dag.add_child(node_new)
                            #     dag.add_child(node_old)
                            #     node_old.add_child(control_child)
                            #     node_new.add_child(control_child)
                        return True
                    elif isinstance(two_qubit_gate, CX):
                        is_on_control = child.node.control == q
                        if ((is_on_control and isinstance(pauli, Z)) or 
                            (not is_on_control and isinstance(pauli, X))): # In these cases we can just commute the Pauli through the CX
                            dag.insert_single_qudit_gate_after(pauli)

                            # dag.insert_between_child(DAG(pauli), control_child if control == q else target_child)
                            # if double_grandchild: # The child is on both the target and control, so it should still be a child of the CX
                            #     assert control_child is not None
                            #     dag.add_child(control_child)
                        else:
                            # We either have a Z on the target or an X on the control
                            new_target = target if is_on_control else control
                            new_pauli = X(new_target) if isinstance(pauli, X) else Z(new_target)
                            new_pauli.repetitions = pauli.repetitions * two_qubit_gate.repetitions
                            if isinstance(new_pauli, Z):
                                new_pauli.repetitions = -new_pauli.repetitions
                            dag.insert_single_qudit_gate_after(pauli)
                            dag.insert_single_qudit_gate_after(new_pauli)
                            # node_old = DAG(pauli)
                            # node_new = DAG(new_pauli)
                            # if not double_grandchild:
                            #     dag.insert_between_child(node_new, target_child if new_target == target else control_child)
                            #     dag.insert_between_child(node_old, control_child if new_target == target else target_child)
                            # else:
                            #     assert control_child is not None
                            #     dag.children.remove(control_child)
                            #     control_child.parents.remove(dag)
                            #     dag.add_child(node_new)
                            #     dag.add_child(node_old)
                            #     node_old.add_child(control_child)
                            #     node_new.add_child(control_child)
                        return True
            # If we reach here, we didn't push the Pauli gate, so we try see if one of the children can push a Pauli

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.update_circuit("Push Pauli")

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
                    H2 = dag.node.copy()
                    dag.node = child.node.copy()
                    dag.node.repetitions = -dag.node.repetitions
                    dag.merge(child)
                    dag.insert_single_qudit_gate_after(H2)
                    return True
                elif isinstance(child.node, SWAP):
                    # Pushing H^2 through SWAP just changes the target of the H^2
                    q = dag.node.target
                    target = child.node.target
                    control = child.node.control
                    H2 = dag.node.copy()
                    dag.node = child.node.copy() # Put the two-qubit gate in place of the H^2
                    dag.merge(child) # Remove the SWAP node
                    new_target = target if control == q else control
                    H2.target = new_target
                    dag.insert_single_qudit_gate_after(H2)
                    return True
                elif isinstance(child.node, S): # Commuting H^2 with S gate
                    # Pushing H^2 through S creates a Z^{-1} gate
                    # Or equivalently H^2;S = S;H^2;Z^{-1} 
                    # Which is useful for us, since we want the Paulis after the H^2 anyway
                    new_z = Z(dag.node.target)
                    new_z.repetitions = -child.node.repetitions
                    dag.node, child.node = child.node, dag.node
                    child.insert_single_qudit_gate_after(new_z)
                    return True
            # If we reach here, we didn't push the H^2 gate, so we try see if one of the children can push one

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.update_circuit("Push H^2")

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
            self.update_circuit("Apply Euler H;S;H -> S^-1;H;S^-1;X")

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
            self.update_circuit("Apply Euler S^-1;H;S^-1 -> H;S;H;X^{-1}")

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
                    # We can push the S gate through a CZ/CX gate control
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
            self.update_circuit("Commute S")

        return success
    
    def push_S_past_CX(self) -> bool:
        """Tries to push an S gate one step to the right in the circuit.
        We use the rewrite 
        S^a(1);CX^b(0,1) = CX^b(0,1);CZ^{-ab}(0,1);S^a(1);S^{ab^2}(0);Z^{ab(b+1)/2)}(0)
        """
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, S):
                # It is an S so it should have at most one child
                if len(dag.children) == 0:
                    return False
                assert len(dag.children) == 1
                child = dag.children[0]
                assert child.node is not None, "Child node must not be None"
                if isinstance(child.node, CX) and child.node.target == dag.node.target:
                    s = dag.node.copy()
                    cx = child.node.copy()
                    a = s.repetitions
                    b = cx.repetitions
                    new_cz = CZ(cx.control,cx.target)
                    new_cz.repetitions = -a*b
                    s2 = S(cx.control,repetitions=a*b**2)
                    if b != 1: # Only need to add a Z if cx has number of repetitions not equal to 1.
                        z = Z(cx.control,repetitions=a*b*(b+1)//2)
                    dag.node = cx  # We replace the S with the CX gate
                    dag.merge(child)  # Put the CX on the place of the original S
                    new_node = dag.insert_two_qudit_gate_after(new_cz)
                    new_node.insert_single_qudit_gate_after(s)
                    new_new_node = new_node.insert_single_qudit_gate_after(s2)
                    if b != 1:
                        new_new_node.insert_single_qudit_gate_after(z)
                    return True
            # If we reach here, we didn't push the S gate, so we try see if one of the children can push one

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.update_circuit("Push S past CX")

        return success
    
    def push_H_gate(self) -> bool:
        """Tries to push an H gate one step to the right in the circuit.
        Specifically, we try to push it through a CZ to produce a CX, or through a CX to produce a CZ.
        We use the following rewrites:
        H(t);CX^a(c,t) = CZ^-a(c,t);H(t)
        H^3(t);CX^a(c,t) = CZ^a(c,t);H^3(t)
        H(t);CZ^a(c,t) = CX^a(c,t);H(t)
        H^3(t);CZ^a(c,t) = CX^-a(c,t);H^3(t)
        """
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, HAD) and dag.node.repetitions % 2 == 1:
                # It is a HAD so it should have at most one child
                if len(dag.children) == 0:
                    return False
                assert len(dag.children) == 1
                child = dag.children[0]
                assert child.node is not None, "Child node must not be None"
                if isinstance(child.node, CX) and child.node.target == dag.node.target or isinstance(child.node, CZ):
                    h_copy = dag.node.copy()
                    target = dag.node.target
                    control = child.node.control if target==child.node.target else child.node.target
                    new_gate = (CX if isinstance(child.node, CZ) else CZ)(control,target)
                    if h_copy.repetitions % 4 == 1:
                        if isinstance(new_gate, CX):
                            new_gate.repetitions = child.node.repetitions
                        else:
                            new_gate.repetitions = -child.node.repetitions
                    else: # We have a H^3
                        if isinstance(new_gate, CX):
                            new_gate.repetitions = -child.node.repetitions
                        else:
                            new_gate.repetitions = child.node.repetitions
                    
                    dag.node = new_gate  # We replace the HAD with the CX/CZ gate
                    dag.merge(child)  # Put the CX/CZ on the place of the original HAD
                    dag.insert_single_qudit_gate_after(h_copy)
                    return True

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.update_circuit("Push S past CX")

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
                    # CZ has a single CX child.
                    # Check that they share control and targets
                    child = dag.children[0]
                    cx = child.node
                    # TODO: implement the case where CZ and CX only overlap on one qudit
                    if cx.target in (cz.target, cz.control) and cx.control in (cz.target, cz.control):
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
            self.update_circuit("Push CZ past CX")

        return success
    
    def transform_CX_to_SWAP(self) -> bool:
        """Finds a pair of CX(c,t);CX(t,c) and transforms them into a SWAP and a CNOT pointing the other way.
        The exact rewrite rule is as follows:
        CX^a(c,t);CX^{a^{-1}}(t,c)+ = CX^{-a^{-1}}(t,c);SWAP(c,t);Mult_a(t);Mult_{-a^{-1}}(c)
        """
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, CX):
                if len(dag.children) == 0:
                    return False
                cx1 = dag.node
                if len(dag.children) == 1 and dag.children[0].node is not None and isinstance(dag.children[0].node, CX):
                    # CZ has a single CX child.
                    # Check that they share control and targets
                    child = dag.children[0]
                    cx2 = child.node
                    assert cx2 is not None
                    if cx2.target == cx1.control and cx2.control == cx1.target and (cx1.repetitions*cx2.repetitions + 1) % self.circuit.dim == 0: # type: ignore
                        # new_cx = cx1.copy()
                        # new_cx.target = cx1.control
                        # new_cx.control = cx1.target
                        dag.node = cx2.copy()  # We set the two CX gates, to a CX pointing the other way
                        dag.node.repetitions *= -1
                        swap = SWAP(cx1.control,cx1.target)
                        child.node = swap # And a SWAP
                        mul1: Gate
                        mul2: Gate
                        if cx1.repetitions % self.circuit.dim == 1: # We do some special cases where there is just
                            mul2 = HAD(cx1.control)  # one resulting gate, which is just H^2.
                            mul2.repetitions = 2
                            child.insert_single_qudit_gate_after(mul2)
                        elif (cx1.repetitions + 1) % self.circuit.dim == 0:
                            mul1 = HAD(cx1.target)
                            mul1.repetitions = 2
                            child.insert_single_qudit_gate_after(mul1)
                        else:
                            mul1 = MUL(cx1.target,cx1.repetitions,self.circuit.dim)
                            mul2 = MUL(cx1.control,cx1.repetitions,self.circuit.dim).to_adjoint()
                            mul2.mult_value *= -1
                            child.insert_single_qudit_gate_after(mul1)
                            child.insert_single_qudit_gate_after(mul2)
                        return True

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.update_circuit("Push CZ past CX")

        return success
    
    
    def toggle_pair_of_CX_gates(self) -> bool:
        """Finds a pair of CX^a(c,t);CX^b(t,c) where a*b + 1 != 0, 
        and transforms them into CX gates pointing the other way, and some single qudit gates.
        In order to prevent infinite loops, we only apply this if there is another CX gate after the second one
        (this prevents a loop if we apply the combine_gates() strategy after this one).
        The exact rewrite rule is as follows: define c = a*b+1, then
        CX^a(c,t);CX^b(t,c) = CX^{b*c^{-1}}(t,c);CX(c,t)^{a*c};Mult_c(c);Mult_{c^{-1}}(t)
        Note that this rewrite is complementary to the one in `transform_CX_to_SWAP()` which applies when a*b + 1 = 0.
        """
        def try_push(dag: DAG) -> bool:
            if dag.node is not None and isinstance(dag.node, CX):
                if len(dag.children) == 0:
                    return False
                cx1 = dag.node
                if len(dag.children) == 1 and dag.children[0].node is not None and isinstance(dag.children[0].node, CX):
                    # CZ has a single CX child.
                    # Check that they share control and targets
                    child = dag.children[0]
                    cx2 = child.node
                    assert cx2 is not None
                    a = cx1.repetitions
                    b = cx2.repetitions
                    c = (a*b + 1) % self.circuit.dim
                    if cx2.target == cx1.control and cx2.control == cx1.target and c != 0: # type: ignore
                        if len(child.children) == 1 and child.children[0].node is not None and isinstance(child.children[0].node, CX):
                            dag.node = cx2.copy()  # We set the two CX gates, to a CX pointing the other way
                            dag.node.repetitions = (b * pow(c, -1, self.circuit.dim)) % self.circuit.dim
                            child.node = cx1.copy()
                            child.node.repetitions = (a * c) % self.circuit.dim
                            mul1: Gate
                            mul2: Gate
                            if (c+1) % self.circuit.dim == 0: # We do some special cases where the
                                mul1 = HAD(cx1.target) # resulting multipliers are jsut H^2
                                mul1.repetitions = 2
                                mul2 = HAD(cx1.control)  
                                mul2.repetitions = 2
                                child.insert_single_qudit_gate_after(mul1)
                                child.insert_single_qudit_gate_after(mul2)
                            else:
                                mul1 = MUL(cx1.target, c, self.circuit.dim)
                                mul2 = MUL(cx1.control,c, self.circuit.dim).to_adjoint()
                                child.insert_single_qudit_gate_after(mul1)
                                child.insert_single_qudit_gate_after(mul2)
                            return True

            for child in dag.children:
                if try_push(child):
                    return True
            return False
        
        success = try_push(self.dag)
        if success:
            self.update_circuit("Push CZ past CX")

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
                    gate = child.node
                    assert gate is not None, "Child node must not be None"
                    if isinstance(gate, CZ) or isinstance(gate, CX) and (
                        gate.target in (swap.target, swap.control) and gate.control in (swap.target, swap.control)):
                        # We can push the SWAP gate through a CZ/CX gate
                        # We just interchange the two gates
                        new_swap = swap.copy()
                        dag.node = gate.copy()
                        if isinstance(gate, CX):
                            dag.node.target, dag.node.control = gate.control, gate.target
                        child.node = new_swap
                        return True
                    if isinstance(child.node, S) or isinstance(child.node, HAD) and child.node.repetitions % 2 == 1:
                        # We can push the SWAP gate through a H/S gate by changing its target
                        new_gate = child.node.copy()
                        new_gate.target = swap.target if new_gate.target != swap.target else swap.control
                        dag.insert_single_qudit_gate_before(new_gate)
                        dag.merge(child)  # Remove the child node
                        return True
                elif len(dag.children) == 2:
                    # We are only going to push one of the children, the other one will remain a child of the SWAP
                    child1 = dag.children[0]
                    child2 = dag.children[1]
                    assert child1.node is not None, "Child node must not be None"
                    if not isinstance(child1.node, (CZ,CX, HAD, S)):
                        child1 = dag.children[1]
                        child2 = dag.children[0]
                    assert child1.node is not None, "Child node must not be None"
                    if isinstance(child1.node, S) or isinstance(child1.node, HAD) and child1.node.repetitions % 2 == 1:
                        new_gate = child1.node.copy()
                        new_gate.target = swap.target if new_gate.target != swap.target else swap.control
                        dag.insert_single_qudit_gate_before(new_gate)
                        dag.merge(child1)  # Remove the child node
                        # This last step might have added new children, but we need to verify that is correct
                        # for grandchild in child2.children:
                        #     if grandchild in dag.children:
                        #         grandchild.parents.remove(dag)
                        #         dag.children.remove(grandchild)
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
            self.update_circuit("Push SWAP")
        
        return success
    

class StepperWidget:
    """Helper class for displaying an IPyWidget for stepping through all the rewrite steps in the circuit.
    Call .show() in a Jupyter notebook cell to display."""
    def __init__(self, circuitsimp: CliffordSimplifier):
        self.obj = circuitsimp
        self.index = 0
        self.max_index = len(self.obj.circuit_list) - 1
        self.cached_circuits = {}
        
        # Slider to move through steps
        self.slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.max_index,
            step=1,
            description='Step:',
            continuous_update=False
        )
        
        # Buttons
        self.prev_button = widgets.Button(description="Previous")
        self.next_button = widgets.Button(description="Next")
        
        # Output area
        self.output = widgets.Output()
        
        # Wire up events
        self.slider.observe(self.on_slider_change, names='value')
        self.prev_button.on_click(self.on_prev)
        self.next_button.on_click(self.on_next)
        
        # Layout
        controls = widgets.HBox([self.prev_button, self.next_button, self.slider])
        self.widget = widgets.VBox([controls, self.output])
        
        self.update_display()
    
    def on_slider_change(self, change):
        self.index = change['new']
        self.update_display()
    
    def on_prev(self, b):
        if self.index > 0:
            self.index -= 1
            self.slider.value = self.index
    
    def on_next(self, b):
        if self.index < self.max_index:
            self.index += 1
            self.slider.value = self.index
    
    def update_display(self):
        with self.output:
            clear_output(wait=True)
            print("Step done:", self.obj.steps_done[self.index] if self.index != self.max_index else "Final")
            print("Circuit:")
            # if not self.index in self.cached_circuits:
            #     self.cached_circuits[self.index] = self.obj.circuit_list[self.index].to_qiskit_rep().draw("mpl")
            # display(self.cached_circuits[self.index])

            display(self.obj.circuit_list[self.index].to_qiskit_rep().draw("mpl"))
            # print(self.obj.circuit_list[self.index])
            print("DAG:")
            print(self.obj.dags[self.index])
    
    def show(self):
        display(self.widget)