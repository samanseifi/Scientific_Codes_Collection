#include <eigen3/Eigen/Dense>
#include <iostream>

#include "Scheme.h"


Scheme::Scheme() {}

Scheme::Scheme(double cfl){
        CFL = cfl;
}

Scheme::~Scheme() {}

Eigen::VectorXd Scheme::Upwind(Eigen::VectorXd u_old) {

        int m = u_old.size();
        Eigen::VectorXd u_n(m);

        // Upwind scheme stability criteria for CFL
        if (CFL > 1.0) {
                std::cout << "Warning: The upwind scheme is NOT stable!" << std::endl;
        }

        for (int i = 1; i < m; i++) {
                u_n(i) = u_old(i) - CFL * (u_old(i) - u_old(i - 1));
        }

        // Periodic boundary condition
        u_n(0) = u_n(m - 1);

        return u_n;


}

Eigen::VectorXd Scheme::MacCormack(Eigen::VectorXd u_old) {
        int m = u_old.size();
        Eigen::VectorXd u_n(m);

        for (int i = 1; i < m - 1; i++) {
                u_n(i) = u_old(i) - (CFL / 2) * (u_old(i+1) - u_old(i)) - CFL * (u_old(i) -  CFL * (u_old(i+1) - u_old(i)) - u_old(i-1) + CFL * (u_old(i) - u_old(i-1)));
        }

        // Fixed boundary condition
        u_n(0) = u_old(0);
        u_n(m - 1) = u_old(m - 1);

        return u_n;
}

Eigen::VectorXd Scheme::LaxWendroff(Eigen::VectorXd u_old) {

}

Eigen::VectorXd LaxFriedrichs(Eigen::VectorXd) {

}
