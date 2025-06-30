clear;

n = 601;
L = 10;
h = L / (n-1);
CFL = 0.30;
t_final = 3.9e-3;
x = 0:h:L;
gamma = 1.4;
alpha = 0.40;

m_1 = 0;
m_r = 0;
rho_1 = 1;
rho_r = 0.01;
e_1 = 2.5;
e_r = 0.125;

m(1:1:(n+1)/2) = m_1;
m((n+3)/2:1:n) = m_r;
rho(1:1:(n+1)/2) = rho_1;
rho((n+3)/2:1:n) = rho_r;
e(1:1:(n+1)/2) = e_1;
e((n+3)/2:1:n) = e_r;


u = m./rho;
P=(gamma-1)*(e-0.5*rho.*u.^2);

a = sqrt((gamma*P)./rho);
dt = CFL * h / max(a);

for t=dt:dt:t_final
    q = [rho; rho.*u; e];
    F = [rho.*u; rho.*u.^2 + P; u.*(e + P)];
    
    q_star(1:3, 1:n-1) = 0.5 * (q(1:3, 1:n-1) + q(1:3, 2:n)) - dt/(2 * h)*(F(1:3, 2:n) - F(1:3, 1:n-1));
    
    rho(1:n-1) = q_star(1, 1:n-1);
    u(1:n-1) = q_star(2, 1:n-1)./rho(1:n-1);
    e(1:n-1) = q_star(3, 1:n-1)./rho(1:n-1);
    P=(gamma-1)*(e(1:n-1)-0.5*rho(1:n-1).*u(1:n-1).^2)
    
    F_star(1:3, 1:n-1) = [rho(1:n-1).*u(1:n-1);...
        rho(1:n-1).*u(1:n-1).^2 + P(1:n-1);...
        u(1:n-1).*(rho(1:n-1).*e(1:n-1) + P(1:n-1))];
    
%     mx =[zeros(1,n-1) ; ones(1,n-1) ; u(1:n-1)];
%     for i=1:3
%        visc(i,1:n-1) = alpha*h^2*rho(1:n-1).*abs((u(2:n)-u(1:n-1))/h).*((u(2:n)-u(1:n-1))/h).*mx(i,1:n-1);
% 
%     end
%     
%     F_star = F_star - visc;
    q(1:3, 2:n-1) = q(1:3, 2:n-1) -dt/h*(F_star(1:3, 2:n-1) - F_star(1:3, 1:n-2));
    
    rho = q(1, 1:n);
    u = q(2, 1:n).*rho(1:n);
    e = q(3, 1:n)./rho;
    P = (gamma - 1)*rho.*(e - 0.5*(u.^2));
end
    