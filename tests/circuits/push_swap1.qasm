OPENQASM 2.0;
include "qelib1.inc";
quditdim 3
qreg q[2];
cz q[0], q[1];
h q[0];
cz q[0], q[1];
cx q[0], q[1];
swap q[0], q[1];
s^-2 q[1];
h^-1 q[1];
cz^-2 q[0], q[1];
h^-1 q[1];
s^-2 q[1];
s^-1 q[0];
h^-1 q[0];