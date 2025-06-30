function [ u,v,p ] = cavity_flow( nx, ny, nt, nit, c, rho, nu, dt )
    %[u,v,p] = cavity_flow(41, 41, 500, 50, 1, 1, 0.1, 0.001)
    
    dx = 2/(nx - 1);
    dy = 2/(ny - 1);
    
    x = linspace(0,2,nx);
    y = linspace(0,2,ny);
    [X,Y] = meshgrid(x,y);
    
    u = zeros(nx,ny); v = zeros(nx,ny);
    p = zeros(nx,ny); b = zeros(nx,ny);
    
    un = zeros(nx, ny);
    vn = zeros(nx, ny);
    
    for i = 1:nt
        un = u;
        vn = v;
        
        b = buildUpB(rho, dt, u, v, dx ,dy);
        p = presPoisson(p, dx, dy, b, nit);
        
        u(2:end-1, 2:end-1) = un(2:end-1, 2:end-1) - ...
            un(2:end-1, 2:end-1)*(dt/dx).*(un(2:end-1, 2:end-1) - un(2:end-1, 1:end-2)) - ...
            vn(2:end-1, 2:end-1)*(dt/dy).*(un(2:end-1, 2:end-1) - un(1:end-2, 2:end-1)) - ...
            dt/(2*rho*dx)*(p(2:end-1, 3:end) - p(2:end-1, 1:end-2)) + ...
            nu*((dt/dx^2)*(un(2:end-1, 3:end) - 2*un(2:end-1, 2:end-1) + un(1:end-2, 2:end-1)) + ...
            (dt/dy^2*(un(3:end, 2:end-1) - 2*un(2:end-1, 2:end-1) + un(1:end-2, 2:end-1))));
        
        v(2:end-1, 2:end-1) = vn(2:end-1, 2:end-1) - ...
            un(2:end-1, 2:end-1)*(dt/dx).*(vn(2:end-1, 2:end-1) - vn(2:end-1, 1:end-2)) - ...
            vn(2:end-1, 2:end-1)*(dt/dy).*(vn(2:end-1, 2:end-1) - vn(1:end-2, 2:end-1)) - ...
            dt/(2*rho*dy)*(p(3:end, 2:end-1) - p(1:end-2, 2:end-1)) + ...
            nu*((dt/dx^2)*(vn(2:end-1, 3:end) - 2*vn(2:end-1, 2:end-1) + vn(2:end-1, 1:end-2)) + ...
            (dt/dy^2*(vn(3:end, 2:end-1) - 2*vn(2:end-1, 2:end-1) + vn(1:end-2, 2:end-1))));
        
        u(1,:) = 0;
        u(:,1) = 0;
        u(:, end-1) = 0;
        u(end-1, :) = 1;
                
        v(1,:) = 0;
        v(end-1, :) = 0;
        v(:, 1) = 0;
        v(:, end-1) = 0;

    end
end

function [b] = buildUpB( rho, dt, u, v, dx, dy)
    [nx, ny] = size(u);
    b = zeros(nx, ny);
    b(2:end-1, 2:end-1) = rho*((1/dt)*((u(2:end-1, 3:end) - u(2:end-1, 1:end-2))/(2*dx) + ...
        (v(3:end, 2:end-1) - v(1:end-2, 2:end-1))/(2*dy)) - ...
        ((u(2:end-1, 3:end) - u(2:end-1, 1:end-2))/(2*dx)).^2 - ...
        2*((u(1:end-2, 2:end-1) - u(1:end-2, 2:end-1))/(2*dy).*(v(2:end-1, 3:end) - v(2:end-1, 1:end-2))/(2*dx)) - ...
        ((v(3:end, 2:end-1) - v(1:end-2, 2:end-1))/(2*dy)).^2);
end

function [p] = presPoisson(p, dx, dy, b, nit)
    [nx, ny] = size(p);
    pn = zeros(nx, ny);
    for q = 1:nit
        p(2:end-1, 2:end-1) = ((pn(2:end-1, 3:end) + pn(2:end-1, 1:end-2))*dy^2 + ...
            (pn(3:end, 2:end-1) + pn(1:end-2, 2:end-1))*dx^2)./(2*(dx^2 + dy^2)) - ...
            (dx^2)*(dy^2)/(2*(dx^2 + dy^2))*b(2:end-1, 2:end-1);
    
        p(:, end-1) = p(:, end-2);
        p(1, :) = p(2, :);
        p(:, 1) = p(:, 2);
        p(end-1, :) = 0;
    end
    
end


