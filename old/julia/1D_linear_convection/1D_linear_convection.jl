using Gadfly
using DataFrames

nx = 41;
nt = 6;
l = 2.0;
dx = l/(nx - 1);
x = 0:dx:2;
c = 2.0;
dt = 0.01;
u_init = zeros(nx);
u = zeros(nx);
u_new = zeros(nx);

for i = 1:nx
   if x[i] >= 0.9 && x[i] <= 1.0            
        u_init[i] = 10.0*(x[i] - 0.9);
    elseif x[i] <= 1.1 && x[i] >= 1.0
        u_init[i] = 10.0*(1.1 - x[i]);
    else
        u_init[i] = 0.0;
    end
end
u = u_init;
df = DataFrame(x=x, u=u_init, Time="Initial");
for t = 1:nt
    for i = 2:nx-1
        u_new[i] = u[i] - c*dt*(u[i] - u[i-1])/dx
    end
    u_new[1] = 0.0;
    u_new[nx] = 0.0;
    u = u_new
    df1 = DataFrame(x=x, u=u, Time=dt*t);
    df = vcat(df, df1);
end
p = plot(df, x="x", y="u", color="Time", Geom.point, Geom.line)
draw(PDF("1D_convection.pdf", 6inch, 3inch), p)

