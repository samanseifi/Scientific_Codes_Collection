using Gadfly
using DataFrames

l = 2;
nx, nt = 20, 10;
dt = 0.01;
dx = l/(nx - 1);
x = 0:dx:l;

u = zeros(nx);
un = zeros(nx);

vis = 0.1;

ip = zeros(nx);
im = zeros(nx);

phi = zeros(nx);
dphi = zeros(nx);

for i = 1:nx
	ip[i] = i + 1;
	im[i] = i - 1;
	phi[i] = exp(-0.25*(x[i]*x[i])/vis) + exp(-0.25((2*pi - x[i])^2)/vis);
	dphi[i] = (-0.5*x[i]/vis)*exp(-0.25*(x[i]*x[i])/vis) + (0.5*((2*pi) - x[i])/vis)*exp(-0.25*((2*pi - x[i])^2)/vis);
end
ip[nx] = 1;
im[1] = nx;

for i = 1:nx
	u[i] = (-2.0*vis*(dphi[i]/phi[i])) + 4;
end

df = DataFrame(x=x, u=u, Time="Initial");
for t = 1:nt
	un = u;
	for i = 1:nx
		u[i] = un[i] - (un[i]*dt*(un[i] - un[im[i]])/dx) + (vis*dt*(un[ip[i]] - 2*un[i] + un[im[i]])/(dx*dx));
		if x[i] == 2*pi
			u[x[i]] = u[1];
		end
	end
	df1 = DataFrame(x=x, u=u, Time=t*dt);
	df = vcat(df, df1);
end

p = plot(df, x="x", y="u", color="Time", Geom.line)
draw(PDF("1D_burgers.pdf", 6inch, 3inch), p)
