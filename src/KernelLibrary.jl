module KernelLibrary

export linear_kernel, polynomial_kernel, laplacian_kernel, gaussian_kernel, anova_kernel, rbf_kernel, fractrbf_kernel, rationalquad_kernel, multiquad_kernel, invmultiquad_kernel, circular_kernel, spherical_kernel, wave_kernel
export power_kernel, log_kernel, spline_kernel, cauchy_kernel, chisquare_kernel, histintersect_kernel, genhistintersect_kernel, fourier_kernel, sigmoid_kernel, wavelet_kernel, transinvwavelet_kernel
# Julia implementation of most of the kernel functions from
# crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html

function linear_kernel{T}(x::Array{T}, y::Vector{T}; c::Real=0.)
	c=convert(T,c)
	(y'*x .+ c)
end

function polynomial_kernel{T}(x::Array{T}, y::Vector{T}, d::Real; α::Real=1., c::Real=1.)
	d=convert(T,d); α=convert(T,α); c=convert(T,c);
	(α.*(y'*x) .+ c).^ d
end

function laplacian_kernel{T}(x::Array{T}, y::Vector{T}, σ::Real)
	σ=convert(T, σ)
	exp(-sum((x.-y).^2.,1).^.5 ./ σ)
end

function gaussian_kernel{T}(x::Array{T}, y::Vector{T}, σ::Real)
	σ=convert(T, σ)
	exp(-sum((x.-y).^2,1) ./ 2*σ.^ 2)
end

function anova_kernel{T}(x::Array{T}, y::Vector{T}, σ::Real, d::Real)
	d=convert(T,d); σ=convert(T, σ)
	sum(exp(-σ*(x.-y).^2).^d,1)
end

function rbf_kernel{T}(x::Array{T}, y::Vector{T}, σ::Real)
	σ=convert(T, σ)
	exp(-sum(abs(x.-y),1) ./ 2*σ.^ 2)
end

function fractrbf_kernel{T}(x::Array{T}, y::Vector{T}, σ::Real, fract::Real)
	σ=convert(T, σ); fract=convert(T, fract); 
	exp(-sum(abs(x .- y).^fract,1).^(1/fract) ./ 2*σ.^ 2)
end

function rationalquad_kernel{T}(x::Array{T}, y::Vector{T}, c::Real)
	c = convert(T, c)
	n = sum((x.-y).^2,1)
	1 .- n./(n .+ c)
end

function multiquad_kernel(x::Array, y::Vector, c::Real) = sqrt(sum((x.-y).^2.,1) .+ c^2.)

function invmultiquad_kernel{T}(x::Array{T}, y::Vector{T}, c::Real)
	c = convert(T, c)
	1 ./ sqrt(sum((x.-y).^2,1) .+ c^2)
end

function circular_kernel(x::Array, y::Vector, σ::Real)
	n = sqrt(sum((x.-y).^2.,1))./σ
	out = (2./π)*arccos(-n) - (2./π)*n*sqrt(1.0 .- n.^2)
	out[n.>=σ] = 0
	out
end

function spherical_kernel(x::Array, y::Vector, σ::Real)
	n = sqrt(sum((x.-y).^2.,1))./σ
	out = 1 - 1.5.*n + .5.*n.^3
	out[n.>=σ] = 0
	out
end

function wave_kernel(x::Array, y::Vector, θ::Real)
	n = sqrt(sum((x.-y).^2.,1))
	vec((θ./n).*sin(n./θ))
end

power_kernel(x::Array, y::Vector, d::Real) = -sqrt(sum((x.-y).^2.,1)).^d

log_kernel(x::Array, y::Vector, d::Real) = -log(sqrt(sum((x.-y).^2.,1)).^d + 1)

function spline_kernel(x::Array, y::Vector, σ::Real)
	xy = x.*y
	mxy = broadcast(min, x,y)
    z = 1 .+ xy .+ xy.*mxy - ((x.+y)/2).*mxy.^2 + (1/3).*mxy.^3
    vec(prod(z,1))
end

cauchy_kernel(x::Array, y::Vector, σ::Real) = 1.0 ./ (1.0 .+ (sum((x.-y).^2.,1) ./ σ.^2.))

chisquare_kernel(x::Array, y::Vector) = 1.0 .- sum((x.-y).^2 ./ (.5.*(x.+y)),1)

histintersect_kernel(x::Array, y::Vector) = sum(broadcast(min, x,y),1)

genhistintersect_kernel(x::Array, y::Vector, α::Real, β::Real) = sum(broadcast(min, abs(x).^α,abs(y).^β),1)

gentstudent_kernel(x::Array, y::Vector, d::Real) = 1 ./ (1.0 .+ sqrt(sum((x.-y).^2.,1)).^d)

function fourier_kernel(x::Array, y::Vector, a::Real)
    dist = x.-y;
    dist[dist==0] = eps(eltype(x))
    z = sin(a + 1/2)*(dist(i))./sin(dist(i)/2);
    vec(prod(z,1))
end

sigmoid_kernel(x::Array, y::Vector; α::Real=1, c::Real=0) = tanh(α*x'*y + c)

motherwavelet(x::Array) = cos(1.75.*x).*exp(-(x.^2)./2)

wavelet_kernel(x::Array, y::Vector, a::Real, c::Real) = vec(prod(motherwavelet((x.-c)./a).*motherwavelet((y.-c)./a),1))

transinvwavelet_kernel(x::Array, y::Vector, a::Real) = vec(prod(motherwavelet((x.-y)./a),1))






end # module
