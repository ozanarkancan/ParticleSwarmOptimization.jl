alpine1(x) = sum(abs(x .* sin(x) + 0.1 .* x))
alpine2(x) = prod(sqrt(x) .* sin(x))
chung_reynolds(x) = sum(x .^ 2) ^ 2
qing(x) = sum((x .^ 2 - (1:length(x))) .^ 2)
