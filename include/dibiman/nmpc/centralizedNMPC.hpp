#ifndef __dibiman_base_nmpc__
#define __dibiman_base_nmpc__

#include <casadi/casadi.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  

#include "dibiman/nmpc/baseNMPC.hpp"

namespace dibiman {
    class centralizedNMPC: public baseNMPC {
        auto optimizer = casadi::Opti();

        /* right arm variables */
        std::vector<casadi::MX> Xr;
        std::vector<casadi::MX> Ar;
        std::vector<casadi::MX> Ur;

        /* right hand parameters */
        casadi::MX x0_r = optimizer.parameter();
        casadi::MX xref_r = optimizer.parameter();
        casadi::MX uref_r = optimizer.parameter();

        /* left hand variables */
        std::vector<casadi::MX> Xl;
        std::vector<casadi::MX> Al;
        std::vector<casadi::MX> Ul;

        /* left hand parameters */
        casadi::MX x0_l = optimizer.parameter();
        casadi::MX xref_l = optimizer.parameter();
        casadi::MX uref_l = optimizer.parameter();

        public:
            centralizedNMPC(
                    const std::string& model_path,
                    const std::vector<std::string>& joints_to_use,
                    const std::vector<std::string>& arm_prefixes
                    );

            void createOCP(void) {};

            void solve(void) {};
    };
};
  
#endif //__dibiman_base_nmpc__
