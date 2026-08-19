#include "dibiman/nmpc/GenericNMPC.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
  
pinocchio::Model::ConfigVectorType genericNMPC::inverseKinematics(  
    const pinocchio::Model & model,  
    const pinocchio::Model::ConfigVectorType & q0,  
    const pinocchio::SE3 & Href,  
    const pinocchio::FrameIndex frame_id,  
    const std::string & target = "full")  
  {  
    pinocchio::Data data(model);  
    const double eps = 1e-6;  
    const int IT_MAX = 4000;  
    const double DT = 1e-1;  
    const double damp = 1e-12;  
    
    Eigen::VectorXd q = q0;  
    Eigen::VectorXd err(target == "full" ? 6 : 3);  
    Eigen::MatrixXd J(6, model.nv);  
    bool success = false;  
    int i = 0;  
    
    for (;;)  
    {  
      pinocchio::forwardKinematics(model, data, q);  
      pinocchio::updateFramePlacement(model, data, frame_id);  
    
      if (target == "full")  
      {  
        const pinocchio::SE3 iMd = data.oMf[frame_id].actInv(Href);  
        err = pinocchio::log6(iMd).toVector();  
    
        J.setZero();  
        pinocchio::computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL, J);  
        J = -pinocchio::Jlog6(iMd.inverse()) * J;  
    
        Eigen::VectorXd v = -J.transpose() * (J * J.transpose() + damp * Eigen::MatrixXd::Identity(6, 6))  
                                 .ldlt().solve(err);  
        q = pinocchio::integrate(model, q, v * DT);  
      }  
      else  
      {  
        err = data.oMf[frame_id].translation() - Href.translation();  
    
        Eigen::MatrixXd J6(6, model.nv);  
        J6.setZero();  
        pinocchio::computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J6);  
        Eigen::MatrixXd J3 = J6.topRows<3>();  
    
        Eigen::VectorXd v = -J3.transpose() * (J3 * J3.transpose() + damp * Eigen::MatrixXd::Identity(3, 3))  
                                 .ldlt().solve(err);  
        q = pinocchio::integrate(model, q, v * DT);  
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
  }
