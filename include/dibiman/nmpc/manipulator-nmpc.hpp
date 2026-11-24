#pragma once

#ifndef __dibiman_manipulator_nmpc__
#define __dibiman_manipulator_nmpc__

#include <casadi/casadi.hpp>

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "dibiman/utils/utils.hpp"
#include "dibiman/fatrop/decentralized-fatrop-helper.hpp"

namespace dibiman {
    class ManipulatorNMPC : public DecentralizedFatropHelper {

        casadi::Function solver = casadi::Function();

        public:
            ManipulatorNMPC(const pinocchio::Model &, const std::string &, const int &);

            void setCosts(const OcpCosts &);
            /* set per stage and terminal costs
             */

            void set_dt(const double &);
            /* set delta t
             */

            void setRef(const Eigen::VectorXd &);
            /* set q reference for point-to-point control
             * also calculaters tau ref to be gravity compensating at q ref
             */

            void create(void);

            Eigen::VectorXd solve(const Eigen::VectorXd &);
    };
};
  
#endif //__dibiman_manipulator_nmpc__
