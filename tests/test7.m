% Test 7: Valid scalar expansion and scalar-matrix operations, no warnings expected
% s is scalar, A is 3x4.
% A + s and s * A are both valid via scalar expansion

A = zeros(3, 4);
s = 2;
C = A + s;
D = s * A;