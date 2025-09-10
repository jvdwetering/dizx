OPENQASM 2.0;
include "qelib1.inc";
quditdim 3
qreg q[2];
cz q[0], q[1];
h q[0];
cx q[0], q[1];
cx^2 q[1], q[0];
h^3 q[1];
swap q[0], q[1];
h^6 q[1];
x^4 q[0];