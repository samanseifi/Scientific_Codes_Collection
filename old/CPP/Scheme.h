#ifndef SCHEME_H_
#define SCHEME_H_

#include <eigen3/Eigen/Dense>

class Scheme {

public:
        Scheme();
        Scheme(double);
        virtual ~Scheme();

        Eigen::VectorXd Upwind(Eigen::VectorXd);
        Eigen::VectorXd MacCormack(Eigen::VectorXd);
        Eigen::VectorXd LaxWendroff(Eigen::VectorXd);
        Eigen::VectorXd LaxFriedrichs(Eigen::VectorXd);

        Eigen::VectorXd u_new;

private:
        double CFL;

};

#endif
