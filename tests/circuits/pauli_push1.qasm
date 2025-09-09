OPENQASM 2.0;
include "qelib1.inc";
quditdim 3 // demonstration of bug in pushing Pauli past CZ
qreg q[2];
cz q[0], q[1];
h q[1];
cx^-1 q[1], q[0];
cx^2 q[0], q[1];
cz^-1 q[0], q[1];
cx^-1 q[1], q[0];
cz^4 q[1], q[0];
s^4 q[0];
s^-4 q[1];
z^-8 q[1];
cz^-1 q[0], q[1];
h^2 q[0];
s^-2 q[1];
hdg q[1];