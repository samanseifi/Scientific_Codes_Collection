function [ u_0 ] = initial( x, nx, vis )
    
    phi = zeros(1, nx);
    dphi = zeros(1, nx);
    
    u = zeros(1, nx);
    
    for i = 1:nx
    	phi(i) = exp(-0.25*(x(i)*x(i))/vis) + exp(-0.25*((2*pi - x(i))^2)/vis);
    	dphi(i) = (-0.5*x(i)/vis)*exp(-0.25*(x(i)*x(i))/vis) + (0.5*((2*pi) - x(i))/vis)*exp(-0.25*((2*pi - x(i))^2)/vis);
    end
    for i = 1:nx
    	u(i) = (-2.0*vis*(dphi(i)/phi(i))) + 4;
    end
    
    u_0 = u;
end

