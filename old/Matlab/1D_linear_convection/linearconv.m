% Solving the Linear Convection problem in 1D:
%
%       du/dt + c du/dx = 0
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

% Frist check the CFL codition (This is the same for all explicit schemes)
if c <= dx/dt
    fprintf("NOTE: The explicit method is STABLE!\n")
else
    fprintf("WARNING: The explicit method is UNSTABLE!\n")
end

% Generate the shape of the intial condition
u_0 = -heaviside(0.1*L -x) + heaviside(0.2*L-x);
u_0 = u_0';

% Setup the visualization
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = '1D_linear_convection.gif';

% Preallocating the solution u_new
u1_new = zeros(N, 1);
u2_new = zeros(N, 1);

u1 = sparse(u_0);  % Apply the IC to the system
u2 = sparse(u_0);  % Apply the IC to the system

t = dt;    % Initialize time

% Solving the problem with matrices: [u]_k+1 = {A}[u]_k
% For explicit schemes matrix {A} is constant
% 1) Upwind Scheme
% **<@
dm = c * (dt/dx);           % ***> Off diagonal -1 values
d = 1 - c * (dt/dx);        % ***> Diagonal values

% Constructing the matrix!
A_upwind = diag(d * ones(1, N)) + diag(dm * ones(1, N-1), -1);
A_upwind(1, N) = dm;
A_upwind = sparse(A_upwind);
% **@>

% 2) Lax–Friedrichs method
% **<@
dm = 0.5*(1 + c*(dt/dx));   % ***> Off diagonal -1 values
dp = 0.5*(1 - c*(dt/dx));   % ***> Off diagonal +1 values

% Constructing the matrix!
A_LaxFr = diag(dm * ones(1, N-1), -1) + diag(dp * ones(1, N-1), 1);
A_LaxFr(1, N) = dm;
A_LaxFr(N, 1) = dp;
A_LaxFr = sparse(A_LaxFr);
% **@>

n = 1;          % Gif generator counter!
% Marching in time!
while (t < t_final)
    
    % Upwind scheme!
    u1_new = A_upwind * u1;
    
    % Lax–Friedrichs scheme!
    u2_new = A_LaxFr * u2;
    
    u3_new = (-heaviside(0.1*L - x + c*t) + heaviside(0.2*L - x + c*t))';
%      u3N = u3_new(N);
% %     u31 = u3_new(1);
%      u3_new(1) = u3N;
%     u3_new(N) = u31;
    
    % Store new solutions
    u1 = u1_new;
    u2 = u2_new;
    u3 = u3_new;
    
    % Step in time!
	t = t + dt;
    
    % Plotting live!
    plot(x, u1,'-o',  x, u2, 's', x, u3, '--k');
    
    % Setup the limits and labels
    xlim([0 L]) 
    ylim([-0.5 max(u_0)+0.5])
    title(sprintf('1D Linear Convection. Time = %3.1f', t));
    xlabel('x', 'Interpreter', 'latex')
    ylabel('u(x)', 'Interpreter', 'latex')
	legend('Upwind','Lax–Friedrichs')
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