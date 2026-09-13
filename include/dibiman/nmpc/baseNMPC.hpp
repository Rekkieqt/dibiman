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

            /* list of manipulator data type */
            std::vector<icubArm> armList;

        public:
            pinocchio::Model::ConfigVectorType inverseKinematics(
                const pinocchio::Model & model,  
                pinocchio::Data & data,  
                const pinocchio::Model::ConfigVectorType & q0,
                const pinocchio::SE3 & Href,
                const pinocchio::FrameIndex frame_id,
                const std::string & target);

            pinocchio::Model getArmModel(
                const std::string & modelpath,
                const std::vector<std::string> & list_joints_to_use);

            casadi::Function rnea(const icubArm & _arm);

            casadi::Function inverseModel(const int n_dim);

            /* void forwardModel(icubArm _arm); */

            casadi::Function armJacobian(const icubArm & _arm);

            void addArmToList(
                    const std::string & modelpath, 
                    const std::vector<std::string> & joints_to_use, 
                    const std::string & name_ee_frame,
                    const std::string & id
                    );

            void getNewData(void);

            void verifyManipulatorList(void);

            virtual void createOCP(void) = 0;

            virtual void solve(const std::map<std::string, Eigen::VectorXd>& initial_and_ref_values) = 0;
    };
};
#endif //__dibiman_base_nmpc__
