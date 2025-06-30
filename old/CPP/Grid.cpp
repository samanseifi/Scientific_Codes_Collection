#include <eigen3/Eigen/Dense>

#include "Grid.h"

Grid::Grid() {

}

// (Note: Eigen's vector and matrices initialization is done directly)
Grid::Grid(int m): x_init(m) {

        dx = 2.0/(m - 1);

        for (int i = 0; i < m; i++) {
                x_init(i) = i * dx;
        }
        nnodes = m;
        ncells = m + 1;
}

Grid::~Grid() {

}

int Grid::num_nodes() { return nnodes; }

int Grid::num_cells() { return ncells; }

double Grid::DeltaX() { return dx; }
