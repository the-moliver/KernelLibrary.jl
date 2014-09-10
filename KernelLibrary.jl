linear(x::Array, y::Vector) = (x' * y)

polynomial(x::Array, y::Vector, d::Real; a::Real=1.) = (x' * y .+ a).^ d

laplacian(x::Array, y::Vector, sigma::Real) = exp(-sum((x.-y).^2.,1).^.5 ./ 2*sigma.^ 2.)

gaussian(x::Array, y::Vector, sigma::Real) = exp(-sum((x.-y).^2.,1) ./ 2*sigma.^ 2.)

rbf(x::Array, y::Vector, sigma::Real) = exp(-sum(abs(x.-y),1) ./ 2*sigma.^ 2.)

fractrbf(x::Array, y::Vector, sigma::Real, fract::Real) = exp(-sum(abs(x .- y).^fract,1).^.(1/fract) ./ 2*sigma.^ 2.)

multiquad(x::Array, y::Vector, sigma::Real) = sqrt(sum((x.-y).^2.,1) .+ sigma^2.)

invmultiquad(x::Array, y::Vector, sigma::Real) = 1./sqrt(sum((x.-y).^2.,1) .+ sigma^2.)

function spline(x::Array, y::Vector, sigma::Real)
	xy = x.*y
    z = 1 .+ xy .+ (1/2).*xy.*broadcast(min, x,y) - (1/6).*broadcast(min, x,y).^3
    vec(prod(z,1))
end

tdist(x::Array, y::Vector, a::Real) = (gamma((a+1)/2)/(sqrt(a*pi)*gamma(a/2)))*(1+(sum((x.-y).^2,1)/a)).^(-((a+1)/2))

function fourier(x::Array, y::Vector, a::Real)
    dist = x.-y;
    dist[dist==0] = eps(eltype(x))
    z = sin(a + 1/2)*(dist(i))./sin(dist(i)/2);
    vec(prod(z,1))
end

logistic(x::Array, y::Vector, a::Real) = 1 ./ (1 + exp(sqrt(sum((x.-y).^2,1)) / a))
	case 'sigmoid'
        K = tanh(params.a*x1*x2'./size(x1,2) + params.a);


