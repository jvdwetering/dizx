OPENQASM 2.0;
include "qelib1.inc";
quditdim 3
qreg q[2];
cz q[0], q[1];
h q[0];
cx q[0], q[1];
cz q[0], q[1];
cx^2 q[1], q[0];
cz^8 q[1], q[0];
s^-6 q[0];
s^-16 q[1];
z^-12 q[1];
s^-1 q[1];
h^3 q[1];
swap q[0], q[1];
x^2 q[0];
h^6 q[1];
