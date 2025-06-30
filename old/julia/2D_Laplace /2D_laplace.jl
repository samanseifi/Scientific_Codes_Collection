using PyPlot

lx = 2.0;
ly = 2.0;
nx = 60;                           
ny = 60;                                  
niter = 10000;                   
dx = lx/(nx-1);                    
dy = ly/(ny-1);                     
x = 0:dx:lx;                        
y = 0:dy:ly;                        

#Initial Conditions
p = zeros(ny,nx);                  
pn = zeros(ny,nx);                 


#Boundary conditions
p[:, 1] = 0;
p[:,nx] = y;
p[1, :] = p[2, :];                   
p[ny,:] = p[ny-1, :]; 


#Explicit iterative scheme with C.D in space (5-point difference)
j = 2:nx-1;
i = 2:ny-1;
for it = 1:niter
    pn = p;
    p[i, j]=((dy^2*(pn[i+1,j]+pn[i-1,j]))+(dx^2*(pn[i,j+1]+pn[i,j-1])))/(2*(dx^2+dy^2));
p[:, 1] = 0;
p[:,nx] = y;
p[1, :] = p[2, :];                   
p[ny,:] = p[ny-1, :]; 

end

