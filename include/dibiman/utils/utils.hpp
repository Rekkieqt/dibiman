#ifndef __dibiman_utils__
#define __dibiman_utils__

#include <casadi/casadi.hpp>

namespace dibiman {
    inline casadi::DM eigenToDM(const Eigen::VectorXd& vec)
    {
        casadi::DM dm = casadi::DM::zeros(vec.size());
        std::memcpy(dm.ptr(), vec.data(), sizeof(double) * vec.size());
        return dm;
    }
}
#endif //__dibiman_utils__
