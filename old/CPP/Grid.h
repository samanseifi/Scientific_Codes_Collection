#ifndef GRID_H_
#define GRID_H_

#include <eigen3/Eigen/Dense>

class Grid {

public:
        Grid();
        Grid(int);
        virtual ~Grid();

        int num_nodes();
        int num_cells();
        double DeltaX();

        Eigen::VectorXd x_init;

private:

        int nnodes;
        int ncells;
        double dx;


};

#endif
