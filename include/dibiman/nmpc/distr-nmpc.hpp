#pragma once

#ifndef __dibiman_base_nmpc__
#define __dibiman_base_nmpc__

#include <casadi/casadi.hpp>

#include <fatrop/fatrop.hpp>

#include "pinocchio/algorithm/model.hpp"

#include "dibiman/utils/utils.hpp"
#include "dibiman/utils/trajectory-generation.hpp"
#include "dibiman/fatrop/static-decentralized-ocp.hpp"
  
namespace dibiman {

    using namespace fatrop;

    typedef OcpAbstractTpl<dibiman::DecentralizedManipulatorStatic> DecentralizedOcp;

    class DecentralizedNMPC {
        /* standard point-to-point */
        std::shared_ptr<DecentralizedOcp> ocp;
        std::shared_ptr<IpData<OcpType>> ip_data;
        std::shared_ptr<IpAlgorithm<OcpType>> ip_alg; // = builder.build(); // algo
        // fatrop::IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcpTpl<DecentralizedOcp>>(ocp));

        /* trajectory generator class */
        std::shared_ptr<TrajectoryGenerator> traj;

        public:
            
            DecentralizedNMPC(const pinocchio::Model &, const std::string &, const int &, const int &); // , const bool& = false);
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

            void setRef(const Eigen::VectorXd &);
            /* set q reference for point-to-point control
             * also calculaters tau ref to be gravity compensating at q ref
             */
            void setWarm(void);
            /* fill the get_initial_* with data from previous solve
             */

            Eigen::VectorXd solve(const Eigen::VectorXd &);
            /* set x0
             * warm_start
             * algo->optimize()
             * return sol.value()
             */

            void setObjectFrame(const pinocchio::SE3 &); // rethink this

    };
};
#endif //__dibiman_base_nmpc__
