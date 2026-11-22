#pragma once

#ifndef __dibiman_base_nmpc__
#define __dibiman_base_nmpc__

#include <casadi/casadi.hpp>

#include <fatrop/fatrop.hpp>

#include "pinocchio/algorithm/model.hpp"

#include "dibiman/utils/utils.hpp"
#include "dibiman/fatrop/static-decentralized-ocp.hpp"
  
namespace dibiman {

    using namespace fatrop;

    typedef OcpAbstractTpl<dibiman::DecentralizedManipulatorStatic> DecentralizedOcp;

    class DecentralizedNMPC {
        std::shared_ptr<DecentralizedOcp> ocp;

        // fatrop::IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcpTpl<DecentralizedOcp>>(ocp));
        std::shared_ptr<IpData<OcpType>> ip_data;

        std::shared_ptr<IpAlgorithm<OcpType>> ip_alg; // = builder.build(); // algo

        public:
            
            DecentralizedNMPC(const pinocchio::Model &, const std::string &, const int &, const int &);
            /* 1st initialized the ocp + data
             * 2nd builder(ocp)
             * 3rd algo = builder.build()
             */

            void setCosts(const OcpCosts &);
            /* set per stage and terminal costs
             */

            void set_dt(const double &);
            /* set delta t
             */

            Eigen::VectorXd solve(const Eigen::VectorXd &);
            /* set x0
             * set xref (calc uref auto mbe?)
             * warm_start
             * algo->optimize()
             * return sol.value()
             */

            void setObjectFrame(const pinocchio::SE3 &); // rethink this

    };
};
#endif //__dibiman_base_nmpc__
