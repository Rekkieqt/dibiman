#ifndef __gen_nmpc__
#define __gen_nmpc__

#include <iostream>
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
  
class genericNMPC {
    public:
      pinocchio::Model::ConfigVectorType inverseKinematics(
          const pinocchio::Model & model,
          const pinocchio::Model::ConfigVectorType & q0,
          const pinocchio::SE3 & Href,
          const pinocchio::FrameIndex frame_id,
          const std::string & target = "full")  
}
#endif //__gen_nmpc__
