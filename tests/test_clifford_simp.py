import unittest
import sys

sys.path.append('..')

from dizx import Circuit
from dizx.clifford_simplifier import CliffordSimplifier
from dizx.symplectic import compare_matrices

class TestSimplify(unittest.TestCase):
    def test_push_S_past_CX(self):
        c = Circuit.from_qasm_file('tests/circuits/push_S_past_CX1.qasm')
        cs = CliffordSimplifier(c)
        self.assertTrue(cs.push_S_past_CX()) # Assert that the rewrite was applied
        m1 = c.to_symplectic_matrix()
        m2 = cs.circuit.to_symplectic_matrix()
        self.assertTrue(compare_matrices(m1,m2,modulus=3))
    
    def test_push_S_past_CX2(self):
        c = Circuit.from_qasm_file('tests/circuits/push_S_past_CX2.qasm')
        cs = CliffordSimplifier(c)
        self.assertTrue(cs.push_S_past_CX()) # Assert that the rewrite was applied
        m1 = c.to_symplectic_matrix()
        m2 = cs.circuit.to_symplectic_matrix()
        self.assertTrue(compare_matrices(m1,m2,modulus=3))

    def test_push_pauli_past_CZ(self):
        c = Circuit.from_qasm_file('tests/circuits/pauli_push1.qasm')
        cs = CliffordSimplifier(c)
        self.assertTrue(cs.push_pauli())
        m1 = c.to_symplectic_matrix()
        m2 = cs.circuit.to_symplectic_matrix()
        self.assertTrue(compare_matrices(m1,m2,modulus=3))
    
    def test_remove_identity(self):
        c = Circuit.from_qasm_file('tests/circuits/remove_identity1.qasm')
        cs = CliffordSimplifier(c)
        self.assertTrue(cs.remove_identity_gate())
        m1 = c.to_symplectic_matrix()
        m2 = cs.circuit.to_symplectic_matrix()
        self.assertTrue(compare_matrices(m1,m2,modulus=3))

    def test_push_SWAP(self):
        c = Circuit.from_qasm_file('tests/circuits/push_swap1.qasm')
        cs = CliffordSimplifier(c)
        self.assertTrue(cs.push_SWAP()) # Assert that the rewrite was applied
        m1 = c.to_symplectic_matrix()
        m2 = cs.circuit.to_symplectic_matrix()
        self.assertTrue(compare_matrices(m1,m2,modulus=3))
    
    def test_transform_CX_to_SWAP(self):
        c = Circuit.from_qasm_file('tests/circuits/transform_CX_to_SWAP.qasm')
        cs = CliffordSimplifier(c)
        self.assertTrue(cs.transform_CX_to_SWAP()) # Assert that the rewrite was applied
        m1 = c.to_symplectic_matrix()
        m2 = cs.circuit.to_symplectic_matrix()
        self.assertTrue(compare_matrices(m1,m2,modulus=3))