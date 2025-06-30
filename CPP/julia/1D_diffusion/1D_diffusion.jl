using Gadfly
using DataFrames

nx = 41;
nt = 6;
l = 2.0;
dx = l/(nx - 1);
x = 0:dx:l;
alpha = 0.1;
dt = 0.01;

u_init = zeros(nx);
u_new = zeros(nx);
u = zeros(nx);

for i = 1:nx
	if x[i] >= 0.5 && x[i] <= 1.0
		u_init[i] = 2.0;
	else
		u_init[i] = 1.0;
	end
end

u = u_init;
df = DataFrame(x=x, u=u_init, Time="Initial");
for t = 1:nt
	for i = 2:nx-1
		u_new[i] = u[i] + (alpha*dt/(dx*dx))*(u[i+1] - 2*u[i] + u[i-1]);
	end
	u_new[1] = 1.0;
	u_new[nx] = 1.0;
	u = u_new;
	df1 = DataFrame(x=x, u=u, Time=t*dt);
	df = vcat(df, df1);	
end

p = plot(df, x="x", y="u", color="Time", Geom.line)
draw(SVG("1D_diffusion.svg", 6inch, 3inch), p)

		
