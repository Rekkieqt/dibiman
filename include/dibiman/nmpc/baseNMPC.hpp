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

            /* casadi nlp solver function */
            casadi::Function solver = casadi::Function();

        public:
            /* list of manipulator data type */
            std::vector<icubArm> armList;

            struct armData {
                double* x0 = nullptr;
                Eigen::VectorXd uref;
                Eigen::VectorXd xref;
                size_t size = 0; /* keep track of alloc size */

                ~armData() {
                    delete[] x0;
                }
            };

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

            casadi::Function forwardModel(const icubArm & _arm);

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

            virtual std::map<std::string, casadi::DM> solve(const std::map<std::string, armData>&) = 0;

    };
};
#endif //__dibiman_base_nmpc__
