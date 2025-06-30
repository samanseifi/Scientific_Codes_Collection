% Solving the diffusion problem in 1D:
%
%       du/dt = c d^2 u/dx^2
%
% for 0 <= x <= L. And with periodic BC:
%       u(0) = u(L)
%
% with IC @ t = 0 with u(x, 0) = u_0(x)
% where u_0(x): a simple step function.

clear 
close all

% Input parameter
L = 1;        % Length of the domain
N = 100;        % Number of grid points

t_final = 10;   % Final time of simulation!
nt = 100;      % Number of time steps

% Wave speed (The speed of wave in the given domain)
c = 0.1;

% Discritized domain spatial and temporal
x = linspace(0, L, N);
dx = L/(N - 1);     % Alternatively dx = x(2) - x(1)
dt = t_final/nt;

% Generate the shape of the intial condition
u_0 = -heaviside(0.1*L -x) + heaviside(0.2*L-x);
u_0 = u_0';

% Setup the visualization
h = figure;
%axis tight manual % this ensures that getframe() returns a consistent size
filename = '1D_diffusion.gif';

ip = zeros(1, N);
im = zeros(1, N);
i  = zeros(1, N);
for k = 1:N
    ip(k) = k + 1;
    im(k) = k - 1;    
    i(k)  = k;
end
ip(N) = 1;
im(1) = N;

% Preallocating the solution u_new
u_new = zeros(N, 1);

u = sparse(u_0);  % Apply the IC to the system

t = dt;    % Initialize time

while t < t_final
		
    u_new(i) = u(i) + (c.*dt/(dx.*dx)).*(u(ip) - 2.*u(i) + u(im));

	u = u_new;

end