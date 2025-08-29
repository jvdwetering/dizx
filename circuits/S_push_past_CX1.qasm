OPENQASM 2.0;
include "qelib1.inc";
quditdim 3
qreg q[2];
s^-4 q[0];
cx^-2 q[1], q[0];
s^-1 q[1];
h^3 q[1];
s^-2 q[0];
swap q[0], q[1];
x^2 q[0];
h^6 q[1];