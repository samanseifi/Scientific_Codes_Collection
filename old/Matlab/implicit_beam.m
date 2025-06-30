%1-D non linear convection in conservative form: Simulating a travelling
...shock wave(a Heaviside function)
... d(u)/dt+u*d(u)/dx=0 ->d(u)/dt=-0.5*d(u^2)/dx

%
clear
%Specifying parameters
nx=400;             %Number of steps in space(x)
nt=50;              %Number of time steps 
dt=0.01;            %Width of each time step
dx=4/(nx-1);        %Width of space step
x=0:dx:4;           %Range of x (0,4) and specifying the grid points
u=zeros(1,nx);      %Preallocating u
un=zeros(1,nx);     %Preallocating un
Eexp=0.11;          %Artificial viscosity

%
%Initial conditions
index=find(x>=2);
u(1:index(1))=1;    %The Heaviside step function
ustar=u;
b=zeros(nx,1);
I=speye(nx,nx);

%
i=2:nx-1;
%Calculating the velocity profile at every time step
for it=1:nt
    un=u;
    h=plot(x,u,'k');       %plotting the velocity profile
    axis([0 4 -1 2])
    title({'1-D Convection';['time(\itt) = ',num2str(dt*it)]})
    xlabel('Spatial co-ordinate (x) \rightarrow')
    ylabel('Transport property profile (u) \rightarrow')
    drawnow; 
    refreshdata(h)
    %Uncomment as necessary
    %---------------------------
    %Implicit methods:
    %Beam-Warming(implicit, second order) with artificial
    %...viscosity(explicit, fourth order)
    
    %E=(-0.25*dt/dx)*sparse(2:nx,1:nx-1,un(1:nx-1),nx,nx);
    %Et=(0.25*dt/dx)*sparse(2:nx,1:nx-1,un(2:nx),nx,nx);
    %D=E+Et'+I;
    %D(1,1)=1;D(1,2)=0;D(nx,nx)=1;D(nx,nx-1)=0;
    %b(3:nx-2)=un(3:nx-2)-Eexp*(un(5:nx)-4*un(4:nx-1)+6*un(3:nx-2)-4*un(2:nx-3)+un(1:nx-4));
    %b(1)=1;b(nx)=0;b(2)=1;b(nx-1)=0;
    %u=D\b;
    
    %Explicit methods:
    
    %Lax-Friedrichs(explicit, first order)
    %u(i)=0.5*(un(i+1)+un(i-1))-0.25*(dt/dx)*(un(i+1).^2-un(i-1).^2);
    
    %Lax-Wendroff(explicit, second order)
    
    %u(i)=un(i)-(0.25*dt/dx)*(un(i+1).^2-un(i-1).^2)+(dt^2/(8*dx^2))...
    %    *((un(i+1)+un(i)).*(un(i+1).^2-un(i-1).^2)-(un(i)+un(i-1)).*...
    %    (un(i).^2-un(i-1).^2));
    
    %MacCormack(explicit, second order)
    
    ustar(i)=un(i)-(0.5*dt/dx)*(un(i+1).^2-un(i).^2);
    u(i)=0.5*(un(i)+ustar(i)-(0.5*dt/dx)*(ustar(i).^2-ustar(i-1).^2));
    %}
    %---------------------------
    %}
end