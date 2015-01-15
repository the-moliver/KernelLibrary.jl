module KernelLibrary

export linear_kernel, polynomial_kernel, laplacian_kernel, gaussian_kernel, anova_kernel, rbf_kernel, fractrbf_kernel, rationalquad_kernel, multiquad_kernel, invmultiquad_kernel, circular_kernel, spherical_kernel, wave_kernel
export power_kernel, log_kernel, spline_kernel, cauchy_kernel, chisquare_kernel, histintersect_kernel, genhistintersect_kernel, fourier_kernel, sigmoid_kernel, wavelet_kernel, transinvwavelet_kernel
# Julia implementation of most of the kernel functions from
# crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html

function linear_kernel{T}(x::Array{T}, y::Array{T}; c::Real=0.)
	c=convert(T,c)
	(y'*x .+ c)
end

function polynomial_kernel{T}(x::Array{T}, y::Array{T}, d::Real; α::Real=1., c::Real=1.)
	d=convert(T,d); α=convert(T,α); c=convert(T,c);
	(α.*(y'*x) .+ c).^ d
end

function laplacian_kernel{T}(x::Array{T}, y::Array{T}, σ::Real)
	σ=convert(T, σ)
	exp(-sum((x.-y).^2.,1).^.5 ./ σ)
end

function gaussian_kernel{T}(x::Array{T}, y::Array{T}, σ::Real)
	σ=convert(T, σ)
	exp(-sum((x.-y).^2,1) ./ 2*σ.^ 2)
end

function anova_kernel{T}(x::Array{T}, y::Array{T}, σ::Real, d::Real)
	d=convert(T,d); σ=convert(T, σ)
	sum(exp(-σ*(x.-y).^2).^d,1)
end

rbf_kernel(x::Array, y::Array, σ::Real) = exp(-sum(abs(x.-y),1) ./ 2*σ.^ 2)

fractrbf_kernel(x::Array, y::Array, σ::Real, fract::Real) = exp(-sum(abs(x .- y).^fract,1).^(1/fract) ./ 2*σ.^ 2)

function rationalquad_kernel{T}(x::Array{T}, y::Array{T}, c::Real)
	c = convert(T, c)
	n = sum((x.-y).^2,1)
	1 .- n./(n .+ c)
end

multiquad_kernel(x::Array, y::Array, c::Real) = sqrt(sum((x.-y).^2.,1) .+ c^2.)

invmultiquad_kernel(x::Array, y::Array, c::Real) = 1./sqrt(sum((x.-y).^2.,1) .+ c^2.)

function circular_kernel(x::Array, y::Array, σ::Real)
	n = sqrt(sum((x.-y).^2.,1))./σ
	out = (2./π)*arccos(-n) - (2./π)*n*sqrt(1.0 .- n.^2)
	out[n.>=σ] = 0
	vec(out)
end

function spherical_kernel(x::Array, y::Array, σ::Real)
	n = sqrt(sum((x.-y).^2.,1))./σ
	out = 1 - 1.5.*n + .5.*n.^3
	out[n.>=σ] = 0
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






end # module
