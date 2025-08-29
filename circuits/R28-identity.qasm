OPENQASM 2.0;
include "qelib1.inc";
quditdim 3
qreg q[2];
cz^2 q[0], q[1];
h^2 q[0];
h q[1];
cz q[0], q[1];
cx q[1], q[0];
z^-1 q[1];
s^-2 q[1];
cx^-2 q[0], q[1];
s^-2 q[1];
h^-2 q[1];
s^-2 q[1];
cx^-1 q[1], q[0];
cz^-1 q[0], q[1];
h^-1 q[1];