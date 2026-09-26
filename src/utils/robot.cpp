#include <Eigen/Dense>

#include "dibiman/utils/robot.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
#include "pinocchio/algorithm/frames.hpp"  
#include "pinocchio/algorithm/model.hpp"  

#include "pinocchio/parsers/urdf.hpp"  

#include "pinocchio/utils/cast.hpp"
#include "pinocchio/utils/check.hpp"

#include <casadi/casadi.hpp>
  
namespace dibiman {

    pinocchio::Model::ConfigVectorType 
    inverseKinematics(  
        const pinocchio::Model & model,  
        pinocchio::Data & data,  
        const pinocchio::Model::ConfigVectorType & q0,  
        const pinocchio::SE3 & Href,  
        const pinocchio::FrameIndex frame_id,  
        const std::string & target = "full")  
    {  
        using namespace pinocchio;

        const double eps = 1e-4;  
        const int IT_MAX = 4000;  
        const double DT = 1e-1;  
        const double damp = 1e-6;  
        
        Eigen::VectorXd q = q0;
        Eigen::VectorXd err(target == "full" ? 6 : 3);

        Eigen::VectorXd v(model.nv);
        Eigen::MatrixXd J(6, model.nv);  
        J.setZero();
        bool success = false;
        int i = 0; 
        
        for (;;)  
        {  
            forwardKinematics(model, data, q);  
            updateFramePlacement(model, data, frame_id);  
        
          if (target == "full")  
          {  
              const SE3 iMd = data.oMf[frame_id].actInv(Href);  
              err = pinocchio::log6(iMd).toVector();  
            
              computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL, J);  
              Data::Matrix6 Jlog;
              Jlog6(iMd.inverse(), Jlog);
              J = -Jlog * J;
              pinocchio::Data::Matrix6 JJt;
              JJt.noalias() = J * J.transpose();
              JJt.diagonal().array() += damp;
              v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
              J = -Jlog6(iMd.inverse()) * J;  
              q = integrate(model, q, v * DT);  
          }  
          else  
          {  
              err = data.oMf[frame_id].translation() - Href.translation();  
            
              Eigen::MatrixXd J6(6, model.nv);  
              J6.setZero();  
              computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J6);  
              Eigen::MatrixXd J3 = J6.topRows<3>();  
            
              Eigen::VectorXd v = -J3.transpose() * (J3 * J3.transpose() + damp * Eigen::MatrixXd::Identity(3, 3))  
                                       .ldlt().solve(err);  
              q = integrate(model, q, v * DT);  
          }  
        
          if (err.norm() < eps) { success = true; break; }  
          if (i >= IT_MAX) { success = false; break; }  
          ++i;  
        }  
        
        if (success)  
            std::cout << "Convergence achieved for " << model.frames[frame_id].name << "!" << std::endl;  
        else  
            std::cout << "\nWarning: the iterative algorithm has not reached convergence to the desired precision "  
                     << model.frames[frame_id].name << std::endl;  
        
        return q;  
    };
      
    pinocchio::Model
    getArmModel(const std::string & modelpath,  
                          const std::vector<std::string> & list_joints_to_use)
    {  
        using namespace pinocchio;

        Model model;  
        pinocchio::urdf::buildModel(modelpath, model);  

        std::vector<JointIndex> list_joints_to_use_id;

        /* Inverted "Joints to Use" List and its indexes */
        std::vector<std::string> list_joints_to_lock;
        std::vector<JointIndex> list_joints_to_lock_id;

        for (std::vector<std::string>::const_iterator it = list_joints_to_use.begin();
                it != list_joints_to_use.end(); 
                ++it)
        {
        const std::string & joint_name = *it;
        if (model.existJointName(joint_name))
            list_joints_to_use_id.push_back(model.getJointId(joint_name));
        }

        for (JointIndex joint_id = 1; joint_id < model.joints.size(); ++joint_id)
        {
            const std::string joint_name = model.names[joint_id];
            auto is_in_vector = std::find(list_joints_to_use.begin(), list_joints_to_use.end(), joint_name);
            if (is_in_vector != list_joints_to_use.end())
                continue;
            else 
            { 
                list_joints_to_lock_id.push_back(joint_id); 
            }
        }

        Eigen::VectorXd q_neutral = neutral(model);

        Model arm_model = buildReducedModel(model, list_joints_to_lock_id, q_neutral);

        return arm_model;
    };
}

