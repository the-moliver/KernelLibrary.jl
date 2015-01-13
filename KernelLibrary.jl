# Julia implementation of most of the kernel functions from
# crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html

linear_kernel(x::Array, y::Array; c::Real=0) = (x'*y .+ c)

polynomial_kernel(x::Array, y::Array, d::Real; α::Real=1., c::Real=1.) = (α.*(x'*y) .+ c).^ d

laplacian_kernel(x::Array, y::Array, σ::Real) = exp(-sum((x.-y).^2.,1).^.5 ./ σ)

gaussian_kernel(x::Array, y::Array, σ::Real) = exp(-sum((x.-y).^2.,1) ./ 2*σ.^ 2.)

anova_kernel(x::Array, y::Array, σ::Real, d::Real) = sum(exp(-σ*(x.-y).^2.).^d,1)

rbf_kernel(x::Array, y::Array, σ::Real) = exp(-sum(abs(x.-y),1) ./ 2*sigma.^ 2.)

fractrbf_kernel(x::Array, y::Array, σ::Real, fract::Real) = exp(-sum(abs(x .- y).^fract,1).^.(1/fract) ./ 2*σ.^ 2.)

function rationalquad_kernel(x::Array, y::Array, c::Real)
	n = sum((x.-y).^2.,1)
	vec(1 .- n./(n .+ c))
end

multiquad_kernel(x::Array, y::Array, c::Real) = sqrt(sum((x.-y).^2.,1) .+ c^2.)

invmultiquad_kernel(x::Array, y::Array, c::Real) = 1./sqrt(sum((x.-y).^2.,1) .+ c^2.)

function circular_kernel(x::Array, y::Array, σ::Real)
	n = sqrt(sum((x.-y).^2.,1))./σ
	out = (2./π)*arccos(-n) - (2./π)*n*sqrt(1.0 .- n.^2)
	out(n.>=σ) = 0
	vec(out)
end

function spherical_kernel(x::Array, y::Array, σ::Real)
	n = sqrt(sum((x.-y).^2.,1))./σ
	out = 1 - 1.5.*n + .5.*n.^3
	out(n.>=σ) = 0
	vec(out)
end

function wave_kernel(x::Array, y::Array, θ::Real)
	n = sqrt(sum((x.-y).^2.,1))
	vec((θ./n).*sin(n./θ))
end

power_kernel(x::Array, y::Array, d::Real) = -sqrt(sum((x.-y).^2.,1)).^d

log_kernel(x::Array, y::Array, d::Real) = -log(sqrt(sum((x.-y).^2.,1)).^d + 1)

function spline_kernel(x::Array, y::Array, σ::Real)
	xy = x.*y
	mxy = broadcast(min, x,y)
    z = 1 .+ xy .+ xy.*mxy - ((x.+y)/2).*mxy.^2 + (1/3).*mxy.^3
    vec(prod(z,1))
end

cauchy_kernel(x::Array, y::Array, σ::Real) = 1.0 ./ (1.0 .+ (sum((x.-y).^2.,1) ./ σ.^2.))

chisquare_kernel(x::Array, y::Array) = 1.0 .- sum((x.-y).^2 ./ (.5.*(x.+y)),1)

histintersect_kernel(x::Array, y::Array) = sum(broadcast(min, x,y),1)

genhistintersect_kernel(x::Array, y::Array, α::Real, β::Real) = sum(broadcast(min, abs(x).^α,abs(y).^β),1)

gentstudent_kernel(x::Array, y::Array, d::Real) = 1 ./ (1.0 .+ sqrt(sum((x.-y).^2.,1)).^d)

function fourier_kernel(x::Array, y::Array, a::Real)
    dist = x.-y;
    dist[dist==0] = eps(eltype(x))
    z = sin(a + 1/2)*(dist(i))./sin(dist(i)/2);
    vec(prod(z,1))
end

sigmoid_kernel(x::Array, y::Array; α::Real=1, c::Real=0) = tanh(α*x'*y + c)

motherwavelet(x::Array) = cos(1.75.*x).*exp(-(x.^2)./2)

wavelet_kernel(x::Array, y::Array, a::Real, c::Real) = vec(prod(motherwavelet((x.-c)./a).*motherwavelet((y.-c)./a),1))

transinvwavelet_kernel(x::Array, y::Array, a::Real) = vec(prod(motherwavelet((x.-y)./a),1))




