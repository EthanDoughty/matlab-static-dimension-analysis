% Test 4: Invalid matrix multiplication, dimension mismatch, warning is expected
% A is 3x4 and x is 5x1, so the inner dimensions of 4 and 5 do not match

A = zeros(3, 4);
x = zeros(5, 1);
y = A * x;