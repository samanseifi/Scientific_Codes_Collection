

Input; % Input values
Sod; % Sod shcok intial conditions

u = m./rho;
P = (gamma-1) * (e - 0.5*(rho.*(u.^2)));
a = sqrt((gamma*P)./rho);
dt = CFL * (dx / max(abs(u + a))); 

% Start Time Marching
for t=dt:dt:t_final  
    Q = [rho; rho.*u; e];
    E = [rho.*u; rho.*(u.^2) + P; u.*(e + P)];
    
    Q(1:3,2:n-1) = 0.5*(Q(1:3, 3:n) + Q(1:3,1:n-2)) - (dt/(2*dx)) * (E(1:3,3:n) - E(1:3,1:n-2));
    
    rho = Q(1,1:n);
    u = Q(2,1:n)./rho(1:n);
    e = Q(3,1:n);
    P = (gamma-1)*(e - 0.5*(rho.*u.^2));  
    %plot(x, u);
    %drawnow;
end
plot(rho)
%subplot(2,2,1),     	plot(x(1:10:end),rho(1:10:end) ,'o', xx, rhoexact, 'r')
%subplot(2,2,2),    	plot(x(1:10:end),u(1:10:end),'o', xx, uexact, 'r')
%subplot(2,2,3),    	plot(x(1:10:end),P(1:10:end), 'o', xx, pexact, 'r' )
%subplot(2,2,4),        plot(x(1:10:end),e(1:10:end), 'o',xx,  pexact./(gamma-1) + 0.5*rhoexact.*uexact.^2, 'r' )
