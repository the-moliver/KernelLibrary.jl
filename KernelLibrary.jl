linear(x::Array, y::Vector) = (x' * y)

polynomial(x::Array, y::Vector, d::Real; a::Real=1.) = (x' * y .+ a).^ d

laplacian(x::Array, y::Vector, sigma::Real) = exp(-sum((x .- y).^2,1).^.5 ./ 2*sigma.^ 2.)

gaussian(x::Array, y::Vector, sigma::Real) = exp(-sum((x .- y).^2,1) ./ 2*sigma.^ 2.)

rbf(x::Array, y::Vector, sigma::Real) = exp(-sum(abs(x .- y),1) ./ 2*sigma.^ 2.)


	case 'tdist'
		dist = bsxfun(@minus, x1, x2);
		K = (gamma((params.a+1)/2)/(sqrt(params.a*pi)*gamma(params.a/2)))*(1+(sum(dist.^2,2)/params.a)).^(-((params.a+1)/2));

	case 'fractrbf'
		dist = bsxfun(@minus, x1, x2);
		K = exp(- (sum(abs(dist).^params.d,2).^(1/params.d)) / (2*params.a^2));
	case 'spline'
		x12 = bsxfun(@times, x1, x2);
	    z = 1 + x12 + (1/2)*bsxfun(@times, x12, bsxfun(@min, x1,x2)) - (1/6)*bsxfun(@min, x1, x2).^3;
	    K = prod(z,2);
	case 'logistic'
		dist = bsxfun(@minus, x1, x2);
		K = 1 ./ (1 + exp(sqrt(sum(dist.^2,2)) / params.a));
	case 'sigmoid'
        K = tanh(params.a*x1*x2'./size(x1,2) + params.a);
    case 'fourier'
        z = sin(params.a + 1/2)*2*ones(size(x1,1),size(x1,2));
        dist = bsxfun(@minus, x1, x2);
        i = find(dist);
        z(i) = sin(params.a + 1/2)*(dist(i))./sin(dist(i)/2);
        K = prod(z,2);
    case 'multiquad'
    	dist = bsxfun(@minus, x1, x2);
    	K = sqrt(sum(dist.^2,2) + params.a^2);
    case 'invmultiquad'
    	dist = bsxfun(@minus, x1, x2);
    	K = 1./sqrt(sum(dist.^2,2) + params.a^2);
