from dizx.circuit import Circuit
from dizx.utils import settings

import random


def CNOT_HAD_PHASE_circuit(
        qudits: int,
        depth: int,
        dim: int = settings.dim,
        p_had: float = 0.2,
        p_t: float = 0.2,
        clifford:bool=False
        ) -> Circuit:
    """Construct a Circuit consisting of CNOT, HAD and phase gates.
    The default phase gate is the T gate, but if ``clifford=True``\ , then
    this is replaced by the S gate.

    Args:
        qudits: number of qubits of the circuit
        depth: number of gates in the circuit
        p_had: probability that each gate is a Hadamard gate
        p_t: probability that each gate is a T gate (or if ``clifford`` is set, S gate)
        clifford: when set to True, the phase gates are S gates instead of T gates.

    Returns:
        A random circuit consisting of Hadamards, CNOT gates and phase gates.

    """
    p_cnot = 1-p_had-p_t
    c = Circuit(qudits)
    for _ in range(depth):
        r = random.random()
        if r > 1-p_had:
            c.add_gate("HAD",random.randrange(qudits))
        elif r > 1-p_had-p_t:
            if not clifford: c.add_gate("T",random.randrange(qudits))
            else:
                for _ in range(random.randrange(c.dim)):
                    c.add_gate("S",random.randrange(qudits))
        else:
            tgt = random.randrange(qudits)
            while True:
                ctrl = random.randrange(qudits)
                if ctrl!=tgt: break
            c.add_gate("CNOT",tgt,ctrl)
    return c
