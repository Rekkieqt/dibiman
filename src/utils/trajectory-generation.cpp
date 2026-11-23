#include "dibiman/utils/trajectory-generation.hpp"

using namespace dibiman;

TrajectoryGenerator::TrajectoryGenerator()
{
    initial_X.setIdentity();
    final_X.setIdentity();
    final_T = 0;
}
inline double
TrajectoryGenerator::st(const double& t)
{
    double x = t / final_T;
    double x3 = x * x * x;
    return a3 * x3 + a4 * x3 * x + a5 * x3 * x * x;

}

inline double
TrajectoryGenerator::d_st(const double& t)
{
    double x = t / final_T;
    double x2 = x * x;
    double res = 3 * a3 * x2 + 4 * a4 * x2 * x + 5 * a5 * x2 * x2;
    return res * (1.0)/final_T;
}

pinocchio::Motion
TrajectoryGenerator::s_twist(const double& t)
{
    pinocchio::Motion dif = pinocchio::log6(initial_X.actInv(final_X));
    pinocchio::SE3 Xs = initial_X * pinocchio::exp6(dif * st(t));
    return Xs.act(dif) * d_st(t);
}

void
TrajectoryGenerator::setData(const pinocchio::SE3 & iX, const pinocchio::SE3 & fX, const int & fT)
{
    initial_X = iX;
    final_X = fX;
    final_T = fT;
}
