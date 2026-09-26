#ifndef __dibiman_utils_utils__
#define __dibiman_utils_utils__

#include <casadi/casadi.hpp>
#include <Eigen/Dense>

namespace dibiman {

    struct armTargetParams {
        Eigen::VectorXd x0;
        Eigen::VectorXd xref;
        Eigen::VectorXd uref;

        armTargetParams() = default;

        armTargetParams(int nv)
            : x0(nv + nv),
            xref(nv + nv),
            uref(nv)

        {
            // x0.setZero();
            // xref.setZero();
            // uref.setZero();
        }

        // double* q0 = nullptr;
        // double* v0 = nullptr;
        // double* uref = nullptr;
        // double* qref = nullptr;
        // double* vref = nullptr;
        
        // size_t n_dim = 0; /* keep track of alloc size */

        // ``~armTargetParams() {
        // ``    delete[] q0, v0, uref, qref, vref;
        // ``}
    };

    inline casadi::DM eigenToDM(const Eigen::VectorXd& vec)
    {
        casadi::DM dm = casadi::DM::zeros(vec.size());
        std::memcpy(dm.ptr(), vec.data(), sizeof(double) * vec.size());
        return dm;
    }
}

/*

void logData(std::fstream & logfile, Eigen::Ref<Eigen::VectorXd> tau, Eigen::Ref<Eigen::VectorXd> qread, int size) {
  for (int i=0; i < size; ++i) {
    logfile << qread[i] << ",";
  }
  for (int i=0; i < size; ++i) {
    logfile << tau[i] << ",";
  }
  logfile << std::endl;
}

void logHeader(std::fstream & logfile, int size) {
  for (int i=0; i < size; ++i) {
    logfile << "joint_meas" << i << ",";
  }
  for (int i=0; i < size; ++i) {
    logfile << "torque " << i << ",";
  }
  logfile << std::endl;
}

void logRef(std::fstream & reffile, Eigen::VectorXd& qref, int size) {
  for (int i=0; i < size; ++i) {
    reffile << "joint_ref" << i << ",";
  }
  reffile << std::endl;
  for (int i=0; i < size; ++i) {
    reffile << qref[i] << ",";
  }
  reffile << std::endl;
}

*/
#endif //__dibiman_utils_utils__
