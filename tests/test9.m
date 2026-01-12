% Test 9: Valid indexing behavior, no warnings are expected
% A(1,2) is treated as a scalar.
% x = A(1,2) and y = x + 1 should both remain scalar

A = zeros(3, 4);
x = A(1, 2);
y = x + 1;
B = zeros(3, 4);
C = A + B;