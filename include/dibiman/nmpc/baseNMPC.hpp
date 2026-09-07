#ifndef __dibiman_base_nmpc__
#define __dibiman_base_nmpc__

#include <casadi/casadi.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  

#include "dibiman/arm/icub-arm.hpp"
  
namespace dibiman {
    class baseNMPC {
        protected:
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
                const pinocchio::Model & model,  
                const pinocchio::Model::ConfigVectorType & q0,
                const pinocchio::SE3 & Href,
                const pinocchio::FrameIndex frame_id,
                const std::string & target);

            void getArmModel(
                const std::string & modelpath,
                const std::vector<std::string> & joints_to_use,
                const std::string & prefix,
                icubArm _arm
                );

            void rnea(icubArm _arm);

            void inverseModel(icubArm _arm);

            /* void forwardModel(icubArm _arm); */

            void armJacobian(icubArm _arm);

            void addArmToList(
                    const std::string & modelpath, 
                    const std::vector<std::string> & joints_to_use, 
                    const std::string & prefix
                    );

            void getNewData(void);

            virtual void createOCP(void) = 0;

            virtual void solve(const std::map<std::string, Eigen::VectorXd>& initial_and_ref_values) = 0;
    };
};
#endif //__dibiman_base_nmpc__
