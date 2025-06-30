% Solving the 2-D Laplace's equation by the Finite Difference
...Method 
% Numerical scheme used is a second order central difference in space
...(5-point difference)

%%
%Specifying parameters
nx=60;                           %Number of steps in space(x)
ny=60;                           %Number of steps in space(y)       
niter=10000;                     %Number of iterations 
dx=2/(nx-1);                     %Width of space step(x)
dy=2/(ny-1);                     %Width of space step(y)
x=0:dx:2;                        %Range of x(0,2) and specifying the grid points
y=0:dy:2;                        %Range of y(0,2) and specifying the grid points

%%
%Initial Conditions
p=zeros(ny,nx);                  %Preallocating p
pn=zeros(ny,nx);                 %Preallocating pn

%%
%Boundary conditions
p(:,1)=0;
p(:,nx)=y;
p(1,:)=p(2,:);                   %Neumann conditions
p(ny,:)=p(ny-1,:);               ...same as above

%%
%Explicit iterative scheme with C.D in space (5-point difference)
j=2:nx-1;
i=2:ny-1;
for it=1:niter
    pn=p;
    p(i,j)=((dy^2*(pn(i+1,j)+pn(i-1,j)))+(dx^2*(pn(i,j+1)+pn(i,j-1))))/(2*(dx^2+dy^2));
    %Boundary conditions (Neumann conditions)
    p(:,1)=0;
    p(:,nx)=y;
    p(1,:)=p(2,:);
    p(ny,:)=p(ny-1,:);   
end

%%
%Plotting the solution
surf(x,y,p,'EdgeColor','none');       
shading interp
title({'2-D Laplace''s equation';['{\itNumber of iterations} = ',num2str(it)]})
xlabel('Spatial co-ordinate (x) \rightarrow')
ylabel('{\leftarrow} Spatial co-ordinate (y)')
zlabel('Solution profile (P) \rightarrow')
