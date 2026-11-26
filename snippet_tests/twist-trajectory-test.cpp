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
main()
{
    double t = 0;
    double Tf = 10;
    double dt = 0.05;

    const int N = (int)Tf/dt;

    using namespace Eigen;
    using namespace pinocchio;

    /* X(s) = Xstart * exp(log(X^{-1}start * Xend) * s) */
    /* Xdot(s) = X(s) * log(X^{-1}start * Xend) * sdot */
    /* Vs = Xdot(s) * X(s)^{-1} - Spatial Velocity */

    SE3 Xi = SE3::Identity();
    SE3 Xf = SE3::Random();
    SE3 X = Xi;
    SE3 X_next;

    for (int k = 0; k < N; ++k)
    {
        Motion nu = log6(Xi.actInv(Xf));
        SE3 Xs = Xi * exp6(nu * s(k * dt, Tf));
        Motion Vs = Xs.act(nu) * ds(k * dt, Tf);
        X_next = exp6(Vs * dt) * X;
        X = X_next;
    }
    std::cout << log6(X.actInv(Xf));
    std::cout << X;
    std::cout << Xf;
}


