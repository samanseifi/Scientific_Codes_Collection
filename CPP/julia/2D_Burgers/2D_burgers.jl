using PyPlot

l = 2.0
nx, ny, nt = 30, 30, 60;
dt = 0.01;
dx = l/(nx - 1);
dy = l/(ny - 1);
x = 0:dx:l
y = 0:dy:l

vis = 0.01

u = zeros(nx, ny); v = zeros(nx, ny);
u_init = zeros(nx, ny); v_init = zeros(nx, ny);
u_new = zeros(nx, ny); v_new = zeros(nx, ny);

for i = 1:nx
	for j = 1:ny
		if (x[i] >= 0.5 && x[i] <= 1.0) && (y[j] >= 0.5 && y[j] <= 1.0)
			u_init[i, j] = 0.0;
			v_init[i, j] = 1.0;
		else
			u_init[i, j] = 1.0;
			v_init[i, j] = 0.0;
		end
	end
end
u_init[:, ny] = u_init[nx, :] = u_init[1, :] = u_init[:, 1] = 1.0;
v_init[:, ny] = v_init[nx, :] = v_init[1, :] = v_init[:, 1] = 0.0;

v = v_init;
u = u_init;

#quiver(x, y , u, v)
#contourf(x, y, u)
surf(x, y, u)
colorbar()
for t = 1:nt
	clf();
	for i = 2:nx-1
		for j = 2:ny-1
			u_new[i, j] = u[i, j] - (u[i, j]*dt*(u[i, j] - u[i-1, j])/dx) + (vis*dt*(u[i+1, j] - 2*u[i, j] + u[i-1, j])/(dx*dx)) - (u[i, j]*dt*(u[i, j] - u[i, j-1])/dy) + (vis*dt*(u[i, j+1] - 2*u[i, j] + u[i, j-1])/(dy*dy));
			v_new[i, j] = v[i, j] - (v[i, j]*dt*(v[i, j] - v[i-1, j])/dx) + (vis*dt*(v[i+1, j] - 2*v[i, j] + v[i-1, j])/(dx*dx)) - (v[i, j]*dt*(v[i, j] - v[i, j-1])/dy) + (vis*dt*(v[i, j+1] - 2*v[i, j] + v[i, j-1])/(dy*dy));
		end	
	end
	u_new[:, ny] = u_new[nx, :] = u_new[1, :] = u_new[:, 1] = 1.0;
	v_new[:, ny] = v_new[nx, :] = v_new[1, :] = v_new[:, 1] = 0.0;
	u = u_new;
	v = v_new;
#	plt.quiver(x, y, u, v)
#	contourf(x, y, u, vmin=0, vmax=1);
	surf(x, y, u);
	draw()
	pause(0.01)
end
