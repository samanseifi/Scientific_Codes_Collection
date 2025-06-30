% Solving the wave equation in 1D:
%
%       d^2 u/dt^2 = c^2 (d^2 u/dx^2)
%
% for 0 <= x <= L. And with fixed BC:
%       u(0) = u(L) = 0
%
% with IC @ t = 0 
%       u (x, 0) = f(x)
%       u'(x, 0) = g(x)
%
% where f(x): a simple Gaussian pulse and g(x) = 0

clear 
close all

% Input parameter
L = 4*pi;        % Length of the domain
N = 100;         % Number of grid points

t_final = 50;    % Final time of simulation!
nt = 300;        % Number of time steps

% Wave speed (The speed of wave in the given domain)
c = 0.5;

% Discritized domain spatial and temporal
x = linspace(0, L, N);
dx = L/(N - 1);     % Alternatively dx = x(2) - x(1)
dt = t_final/nt;

% Frist check the CFL codition (This is the same for all explicit schemes)

% Generate the shape of the intial condition
u_0 = 2*exp(-(x - L/2).^2);
u_0 = u_0';

% Setup the visualization
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = '1D_wave.gif';

% Preallocating the solution u_new
u1_new = zeros(N, 1);

% Solving the problem with matrices: [u]_k+1 = {A}[u]_k - {B}[u]_k-1
% i) Constructing matrix {A}:
% **<@
r = c*(dt/dx);
dm = r^2;               % ***> Off diagonal -1 values
d  = 2*(1 - r^2);       % ***> Diagonal values
dp = r^2;               % ***> Off diagonal +1 values

% Constructing the {A} matrix!
A = diag(dp * ones(1, N-1), 1) + diag(d * ones(1, N)) + diag(dm * ones(1, N-1), -1);
A(1, 1) = 1;            % ***> Account for fixed BC (left)
A(1, 2) = 0;
A(N, N) = 1;            % ***> Account for fixed BC (right)
A(N, N-1) = 0;
A = sparse(A);

% Constructing the {B} matrix!
B = diag(ones(1, N));
B = sparse(B);
% **@>

u0 = u_0;       % Apply the IC to the system u_0
u1 = 0.5*A*u0;  % Apply the IC to the system u_1

t = 0;          % Initialize time
n = 1;          % Gif generator counter!
% Marching in time!
while (t < t_final)
    
    % Simple forward in time!
    u1_new = A * u1 - B * u0;
    
    % Store new solutions
    u0 = u1;
    u1 = u1_new;
    
    
    % Step in time!
	t = t + dt;
    
    % Plotting live!
    plot(x, u1, '-o', 'LineWidth',2);
    
    % Setup the limits and labels
    xlim([0 L]) 
    ylim([-2 2])
    title(sprintf('1D Wave Equation. Time = %7.3f', t));
    xlabel('x', 'Interpreter', 'latex')
    ylabel('u(x)', 'Interpreter', 'latex')
	
    drawnow;
    
    % Creating a GIF!
    % **<@
    % Capture the plot as an image 
    frame = getframe(h); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    
    % Write to the GIF File 
    if n == 1 
        imwrite(imind,cm,filename,'gif', 'DelayTime',0.01,'Loopcount',inf); 
    else 
        imwrite(imind,cm,filename,'gif', 'DelayTime',0.01,'WriteMode','append'); 
    end 
    n = n +1; 
    % **@>
end