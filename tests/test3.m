% Test 3: Valid matrix multiplication, no warnings are expected
% A is 3x4 and x is 4x1, so A * x is valid and produces a 3x1 column vector

A = zeros(3, 4);
x = zeros(4, 1);
y = A * x;