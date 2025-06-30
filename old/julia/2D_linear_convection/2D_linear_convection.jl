using PyPlot

ny = 40;
nx = 40;
nt = 10;
l = 2.0;
dx = l/(nx - 1);
dy = l/(ny - 1);
x = 0:dx:l;
y = 0:dy:l;
c = 2.0;
dt = 0.01;

u_init = zeros(nx, ny);
u_new = zeros(nx, ny);
u = zeros(nx, ny);

for i = 1:nx
	for j = 1:ny
		if x[i] >= 0.5 && x[i] <= 1.0 && y[j] >= 0.5 && y[j] <= 1.0
			u_init[i, j] = 2.0;
		else
			u_init[i, j] = 1.0;
		end
	end
end

u = u_init;
contourf(x, y , u)
colorbar()
for t = 1:nt
	for i = 2:nx-1
		for j = 2:nx-1
			u_new[i, j] = u[i, j] - c*dt*(u[i, j] - u[i-1, j])/dx - c*dt*(u[i, j] - u[i, j-1])/dy;
		end
	end
   u_new[1, :] = 1.0;  u_new[:, ny] = 1.0;
   u_new[nx, :] = 1.0; u_new[:, 1] = 1.0;
   u = u_new
	contourf(x, y, u, vmin=1.0, vmax=2.0);
	draw();
	sleep(0.5);	
end
