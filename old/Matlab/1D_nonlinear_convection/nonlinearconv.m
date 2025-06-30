function [x, u] = nonlinearconv(scheme, L, nx, nt, dt, initial)
% Solving the nonlinear convection equation in 1D using different numerical
% schemes: 
%
%   1) Lax-Wendroff, 
%   2) Lax-Friedrichs,
%   3) MacCormak
%
% choices for 'scheme': 'LaxWendroffTwoStep', 'LaxWendroff', 'LaxFriedrichs', 'MacCormak'
%
% nonlinear convection:     0 <= x <= L
%
%   du/dt = u * du/dx
%
% with periodic BC: u(0) = u(L) and initial condition of 
%
% Example run:
%
%   nonlinearconv('LaxWendroff', 2*pi, 201, 200, 0.0016)
%

% Discritized domain spatial
x = linspace(0, L, nx);
dx = L/(nx - 1);

% Initial Condition
u_0 = initial;
fprintf("Initial condition loading...")

% Check the initial condition
if size(u_0) == size(x)
    fprintf("OK\n")
else
    fprintf("Failed \n")
    fprintf("The initial condition doesn't match with the domain size! \n")
end


% Setup the 1D stencil accounting for periodic BC
[ip, im, i] = stencil(nx);

u = u_0;

u_new = zeros(size(u_0));

if strcmp(scheme, 'LaxWendroffTwoStep')
    for t = 1:nt  
        u1 = zeros(1, nx);
        u2 = zeros(1, nx);
        
        % Lax-Wendroff
        % First Step
        u1(i) = 0.5.*(u(i) + u(ip(i))) - 0.25.*(dt/dx).*(u(ip(i)).^2 - u(i).^2);
            
        % Second Step
        u2(i) = 0.5.*(u(im(i)) + u(i)) - 0.25.*(dt/dx).*(u(i).^2 - u(im(i)).^2);                              
            
        % Update the solution!
        u_new(i) = u(i) - 0.5.*(dt/dx).*(u1(i).^2 - u2(i).^2); 
        u = u_new;
    end

elseif strcmp(scheme, 'LaxWendroff')
    for t = 1:nt  
        u1 = zeros(1, nx);
        u2 = zeros(1, nx);
        
        % Lax-Wendroff
        u1(i) = (u(i) + (u(i) + u(ip(i))).*0.5).*0.5;
        u2(i) = (u(i) + (u(im(i)) + u(i)).*0.5).*0.5;
        u_new(i) = u(i) - 0.25.*(dt/dx).*(u(ip(i)).^2 - u(im(i)).^2) + 0.5.*(dt^2/dx^2).*(u1(i).*(u(ip(i)).^2 - u(i).^2) - u2(i).*(u(i).^2 - u(im(i)).^2));

        u = u_new;
    end

elseif strcmp(scheme, 'LaxFriedrichs')
    for t = 1:nt  
        % Lax–Friedrichs
        u_new(i) = 0.5.*(u(ip(i)) + u(im(i))) - 0.25.*(dt/dx).*(u(ip(i)).^2 - u(im(i)).^2);
        
        u = u_new;
    end

elseif strcmp(scheme, 'MacCormak')
    u_star = u;
    for t = 1:nt      

        % Mac Cormak
        % Predictor
        u_star(i) = u(i) - (dt/dx).*0.5.*(u(ip(i)).^2 - u(i).^2); 
            
        % Corrector
        u_new(i) = 0.5.*(u(i) + u_star(i)) - 0.5.*(dt/dx).*0.5.*(u_star(i).^2 - u_star(im(i)).^2); 

        u = u_new;
    end
end   

% Defining the stencil
function [ip, im, i] = stencil(nx) 
    ip = zeros(1, nx);
    im = zeros(1, nx);
    i  = zeros(1, nx);
    for k = 1:nx
        ip(k) = k+1;
        im(k) = k-1;
        i(k) = k;
    end
    ip(nx) = 1;
    im(1) = nx;
    
