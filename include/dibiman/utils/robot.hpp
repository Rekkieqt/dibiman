#ifndef __dibiman_utils_robot__
#define __dibiman_utils_robot__

#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  

#include "dibiman/utils/robot.hpp"

namespace dibiman {

    pinocchio::Model::ConfigVectorType inverseKinematics(
        const pinocchio::Model &,  
        pinocchio::Data &,  
        const pinocchio::Model::ConfigVectorType &,
        const pinocchio::SE3 &,
        const pinocchio::FrameIndex,
        const std::string &);

    pinocchio::Model getArmModel(
        const std::string &,
        const std::vector<std::string> &);

};
#endif //__dibiman_utils_robot__
