% Test 10: Invalid symbolic mismatch inside control-flow, warning is expected
% A is n x k and B is n x n.
% A + B has incompatible dimensions (k vs n), so the analysis should detect it

n = 4;
k = 5;

A = zeros(n, k);   % n x k
B = zeros(n, n);   % n x n

if n > 0
    C = A + B;
else
    C = A;
end