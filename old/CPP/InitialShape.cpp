#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>

#include "InitialShape.h"

InitialShape::InitialShape() {

}

InitialShape::InitialShape(Grid grid, int code): u_init(grid.num_nodes()) {
        // code = 1: Square Wave function
        // code = 2: Triangle Wave function
        // code = 3: Step function
        if (code == 1) {
                u_init = SquareWaveFunction(grid);
        } else if (code == 2) {
                u_init = TriangleWaveFunction(grid);
        } else if (code == 3) {
                u_init = StepFunction(grid);
        }
}

InitialShape::~InitialShape() {


}
/*
*         +------+
*         |      |
*         |      |
*         |      |
* --------+      +----------------
*/
Eigen::VectorXd InitialShape::SquareWaveFunction(Grid grid) {
        Eigen::VectorXd u_0(grid.num_nodes());
        for (int i = 0; i < grid.num_nodes(); i++) {
                if (grid.x_init(i) > 0.25 && grid.x_init(i) < 0.75) {
                        u_0(i) = 1.0;
                } else {
                        u_0(i) = 0.0;
                }
        }
        return u_0;
}
/*
*             /\
*            /  \
*           /    \
*          /      \
* --------+        +----------------
*/
Eigen::VectorXd InitialShape::TriangleWaveFunction(Grid grid) {
        Eigen::VectorXd u_0(grid.num_nodes());
        for (int i = 0; i < grid.num_nodes(); i++) {
                if (grid.x_init(i) >= 0.25 && grid.x_init(i) < 0.375) {
                        u_0(i) = 8.0*grid.x_init(i) - 2.0;
                } else if (grid.x_init(i) >= 0.375 && grid.x_init(i) <= 0.5) {
                        u_0(i) = -8.0*grid.x_init(i) + 4.0;
                } else {
                        u_0(i) = 0.0;
                }
        }
        return u_0;

}

/*
* -----------+
*            |
*            |
*            |
*            +-------------------
*/
Eigen::VectorXd InitialShape::StepFunction(Grid grid) {
        Eigen::VectorXd u_0(grid.num_nodes());
        for (int i = 0; i < grid.num_nodes(); i++) {
                if (grid.x_init(i) < 0.75) {
                        u_0(i) = 1.0;
                } else {
                        u_0(i) = 0.0;
                }
        }
        return u_0;
}

void InitialShape::Print() {
        //std::cout << u_init << std::endl;
}
