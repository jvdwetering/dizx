OPENQASM 2.0;
include "qelib1.inc";
quditdim 3
qreg q[2];
cz^-1 q[0], q[1];
s^-2 q[1];
cx^2 q[0], q[1];
cz^4 q[0], q[1];
cx^-1 q[1], q[0];
