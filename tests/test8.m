% Test 8: Invalid elementwise operation mismatch, warning is expected
% A is 3x4 and B is 3x5, so A .* B is invalid due to differing column sizes

A = zeros(3, 4);
B = zeros(3, 5);
C = A .* B;