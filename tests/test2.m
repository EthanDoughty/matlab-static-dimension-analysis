% Test 2: Invalid matrix addition, dimensions mismatch, warning is expected
% A is 3x4 and B is 4x4, so A + B is invalid

A = zeros(3, 4);
B = zeros(4, 4);
C = A + B;