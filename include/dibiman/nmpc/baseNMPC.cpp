#ifndef __base_nmpc__
#define __base_nmpc__

#include <casadi/casadi.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
  
class baseNMPC {
    public:
        pinocchio::Model::ConfigVectorType inverseKinematics(
            const pinocchio::Model & model,
            const pinocchio::Model::ConfigVectorType & q0,
            const pinocchio::SE3 & Href,
            const pinocchio::FrameIndex frame_id,
            const std::string & target = "full");

        pinocchio::Model getArmModel(
            const std::string & modelpath,  
            const std::string & armPrefix,  
            const std::vector<std::string> & jointsParam);

        casadi::Function rnea(void);

        casadi::Function inverseModel(int nq, double dt);

        casadi::Function forwardModel(/* cmodel, cdata via casadi-pinocchio C++ API */, double dt) {  
}
#endif //__base_nmpc__
