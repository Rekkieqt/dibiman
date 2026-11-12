#include "pinocchio/multibody.hpp"
#include <iostream>
#include <cmath>

#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "pinocchio/spatial.hpp"

const double a5 = 6;
const double a4 = -15;
const double a3 = 10;

static inline double s(double t, double Tf)
{

    double x = t / Tf;
    return a3 * pow(x, 3) + a4 * pow(x, 4) + a5 * pow(x, 5);
}

static inline double ds(double t, double Tf)
{
    double x = t / Tf;
    return (3 * a3 * pow(x, 2) + 4 * a4 * pow(x, 3) + 5 * a5 * pow(x, 4)) * 1.0/Tf;
}

int
main(int argc, char *argv[])
{
    double t = 9;
    double Tf = 10;
    if (argc > 1)
    {
        t = static_cast<double>(atoi(argv[1]));
        Tf = static_cast<double>(atoi(argv[2]));
    }

    using namespace Eigen;
    using namespace pinocchio;

    Model model;
    buildModels::humanoidRandom(model);

    Data data(model);

    model.lowerPositionLimit.head<3>().fill(-1.);
    model.upperPositionLimit.head<3>().fill(1.);

    VectorXd q = randomConfiguration(model);
    VectorXd v(VectorXd::Random(model.nv));
    VectorXd a(VectorXd::Random(model.nv));

    /* X(s) = Xstart * exp(log(X^{-1}start * Xend) * s) */
    /* Xdot(s) = X(s) * log(X^{-1}start * Xend) * sdot */
    /* Vs = Xdot(s) * X(s)^{-1} - Spatial Velocity */

    SE3 Xi(Matrix3d::Identity(), Vector3d(0, 0, 0));
    Matrix3d Rf = rpy::rpyToMatrix(Vector3d(M_PI/2.0, 0, 0));

    SE3 Xf(Rf, Vector3d(1, 1, 1));

    Motion nu = log6(Xi.actInv(Xf));
    SE3 Xs = Xi * exp6(nu * s(t, Tf));
    Motion Vs = Xs.act(nu) * ds(t, Tf);

    std::cout << Xs << std::endl;
    std::cout << Vs << std::endl;

}

