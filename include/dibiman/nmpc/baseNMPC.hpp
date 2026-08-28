#ifndef __dibiman_base_nmpc__
#define __dibiman_base_nmpc__

#include <casadi/casadi.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  

#include "dibiman/arm/icub-arm.hpp"
  
namespace dibiman {
    class baseNMPC {
        /* ocp horizon */
        int H;

        /* dimensions of state variables */
        int nx;
        int nu;
        int na;

        /* discretization step */
        double dt;

        std::vector<icubArm> armList;

        public:
            pinocchio::Model::ConfigVectorType inverseKinematics(
                const std::string & prefix,
                const pinocchio::Model::ConfigVectorType & q0,
                const pinocchio::SE3 & Href,
                const pinocchio::FrameIndex frame_id,
                const std::string & target = "full");

            void getArmModel(
                const std::string & modelpath,
                const std::string & prefix,
                const std::vector<std::string> & joints,
                icubArm _arm
                );

            void rnea(icubArm _arm);

            void inverseModel(icubArm _arm);

            void forwardModel(icubArm _arm);

            void armJacobian(icubArm _arm);

            void addArmToList(
                    const std::string & modelpath, 
                    const std::vector<std::string> & joints_to_use, 
                    const std::string & prefix
                    );

            virtual casadi::Opti createOCP(void) = 0;

            virtual casadi::Opti solve(void) = 0;
    };
};
#endif //__dibiman_base_nmpc__
