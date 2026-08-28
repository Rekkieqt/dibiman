#ifndef __dibiman_arm__
#define __dibiman_arm__

#include <casadi/casadi.hpp>
  
namespace dibiman 
{
    class icubArm 
    {
        public:
            std::string prefix;

            pinocchio::Model model;
            pinocchio::Data data;

            pinocchio::FrameIndex hand_id;
            pinocchio::FrameIndex object_id;

            casadi::Function rnea_h = casadi::Function();
            casadi::Function inv_dyn_f = casadi::Function();
            casadi::Function for_dyn_f = casadi::Function();
            casadi::Function jac_h = casadi::Function();
    };

    /*
    struct icubArm {
        casadi::Function rnea_h;
        casadi::Function inv_dyn_f;
        casadi::Function for_dyn_f;
        casadi::Function arm_jac_h;
    }
    */
};

#endif //__dibiman_arm__
