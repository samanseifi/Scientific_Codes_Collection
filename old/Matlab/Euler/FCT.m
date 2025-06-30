clear

Input;
Sod;

u = m./rho;
p = (gamma-1)*(e - 0.5*rho.*u.^2);
a = sqrt((gamma*p)./rho);
dt = CFL * dx/max(a);


% Q = [zeros(1,n); zeros(1,n); zeros(1,n)];
% E = [zeros(n,1); zeros(n,1); zeros(n,1)];
% E_low = [zeros(n,1); zeros(n,1); zeros(n,1)];
% E_high = [zeros(n,1); zeros(n,1); zeros(n,1)];

for t = dt:dt:5*dt
Q = [rho; rho.*u; e];
    
E = [rho.*u; rho.*u.^2+p; u.*(e+p)];
E_low = [rho.*u; rho.*u.^2+p; u.*(e+p)];
E_high(1:3,1:n-1) = (E(1:3,1:n-1) + (1 - dt/dx).*(E(1:3,2:n) - E(1:3,1:n-1))./2);
E_high(1:3,n) = E(1:3,n);

A(1:3,1:n) = E_high(1:3,1:n) - E_low(1:3,1:n);


Q_star(1:3,1) = Q(1:3,1) - (1/dx).*(E_low(1:3,1));
Q_star(1:3,2:n) = Q(1:3,2:n) - (1/dx).*(E_low(1:3,2:n) - E_low(1:3,1:n-1));

for j = 1:3
    for i = 1:n
        if i == 1
           A_c(j,i) = sign(A(j,i)).*(max(0, min(abs(A(j,i)), min(sign(A(j,i)).*(Q_star(j,i+2) - Q_star(j,i+1)).*dx, sign(A(j,i)).*(Q_star(j,i)).*dx))));
        elseif i == n-1
           A_c(j,i) = sign(A(j,i)).*(max(0, min(abs(A(j,i)), min(sign(A(j,i)).*(-Q_star(j,i+1)).*dx, sign(A(j,i)).*(Q_star(j,i) - Q_star(j,i-1)).*dx)))); 
        elseif i == n
           A_c(j,i) = sign(A(j,i)).*(max(0, min(abs(A(j,i)), sign(A(j,i)).*(Q_star(j,i) - Q_star(j,i-1)).*dx)));  
        else
           A_c(j,i) = sign(A(j,i)).*(max(0, min(abs(A(j,i)), min(sign(A(j,i)).*(Q_star(j,i+2) - Q_star(j,i+1)).*dx, sign(A(j,i)).*(Q_star(j,i) - Q_star(j,i-1)).*dx))));
        end
    end    
end

Q(1:3,1) = Q_star(1:3,1) - (1/dx).*(A_c(1:3,1));
Q(1:3,2:n-1) = Q_star(1:3,2:n-1) - (1/dx).*(A_c(1:3,2:n-1) - A_c(1:3,1:n-2));
Q(1:3,n) = Q_star(1:3,n) - (1/dx).*(A_c(1:3,n));
rho=Q(1,1:n); 
u=Q(2,1:n)./rho(1:n); 
e=Q(3,1:n); 
p=(gamma-1)*(e-0.5*rho.*u.^2);
end
