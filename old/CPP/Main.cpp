// compile with: g++ InitialShape.cpp Scheme.cpp Grid.cpp Main.cpp -o samp1
// run with: ./samp1 > u.txt

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>

#include "InitialShape.h"
#include "Scheme.h"
#include "Grid.h"

int main() {

        // Problem parameters
        int m = 100;
        int t_end = 10;

        std::cout << m     << std::endl;
        std::cout << t_end << std::endl;

        double c = 10.0;
        double dt = 0.001;

        // Create the grid: m is the number of nodal points
        Grid grid(m);
        double dx = grid.DeltaX();

        // Constructing initial conditions
        InitialShape IC(grid, 1);

        Eigen::VectorXd u_i;
        u_i = IC.u_init;

        // Calculating CFL number
        double CFL = c * (dt / dx);

        // Choose of discretization scheme!
        Scheme scheme(CFL);

        Eigen::VectorXd u_old, u_new;
        u_old = u_i;

        // Time Marching!
        for (int t = 0; t < t_end; t++) {
                std::cout << u_old << std::endl;
                if (t == 0) {
                        u_old = u_i;
                }
                u_new = scheme.MacCormack(u_old); // codename 1
                u_old = u_new;

        }

        return 0;

}
