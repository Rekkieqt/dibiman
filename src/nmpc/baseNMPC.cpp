#include "dibiman/nmpc/baseNMPC.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
#include "pinocchio/parsers/urdf.hpp"  
#include "pinocchio/algorithm/model.hpp"  
  
pinocchio::Model::ConfigVectorType 
baseNMPC::inverseKinematics(  
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
  };
  
pinocchio::Model 
baseNMPC::getArmModel(const std::string & modelpath,  const std::string & armPrefix,  const std::vector<std::string> & jointsParam)  
  {  
      std::vector<std::string> jointsToFree;  
      for (const auto & jnt : jointsParam)  
        jointsToFree.push_back(armPrefix + jnt);  
      jointsToFree.push_back("universe");  
      
      pinocchio::Model fullModel;  
      pinocchio::urdf::buildModel(modelpath, fullModel);  
      
      std::vector<pinocchio::JointIndex> jointsToLockIDs;  
      for (const auto & jn : fullModel.names)  
      {  
        if (std::find(jointsToFree.begin(), jointsToFree.end(), jn) == jointsToFree.end())  
          jointsToLockIDs.push_back(fullModel.getJointId(jn));  
      }  
      
      Eigen::VectorXd initConf = Eigen::VectorXd::Zero(fullModel.nq);  
      
      pinocchio::Model model = pinocchio::buildReducedModel(fullModel, jointsToLockIDs, initConf);  
      pinocchio::Data data(model);  
      
      return model;  
  };

casadi::Function 
baseNMPC::inverseModel(int nq, double dt) {  
    using namespace casadi;  
  
    // Dynamic variables  
    SX q = SX::sym("q", nq);  
    SX v = SX::sym("v", nq);  
    SX a = SX::sym("a", nq);  
  
    // Simplified dynamics  
    SX x = vertcat(q, v);  
    SX dx = vertcat(v, a);  
  
    SX qk = q + dt * v;  
    SX vk = v + dt * a;  
    SX xk = vertcat(qk, vk);  
  
    return Function("Fk", {x, a}, {xk}, {"x", "a"}, {"xk"}).expand();  
};

casadi::Function 
baseNMPC::rnea(void) {  
    /* model/cmodel/cdata bindings via casadi-pinocchio C++ API */
    using namespace casadi;  
  
    SX q = SX::sym("q", cmodel.nq);  
    SX v = SX::sym("v", cmodel.nv);  
    SX a = SX::sym("a", cmodel.nv);  
  
    // RNEA + derivatives (via casadi-pinocchio C++ bindings, cpin equivalent)  
    SX tau = cpin::rnea(cmodel, cdata, q, v, a);  
    cpin::computeRNEADerivatives(cmodel, cdata, q, v, a);  
  
    SX du_dq = cdata.dtau_dq;  
    SX du_dv = cdata.dtau_dv;  
    SX du_da = cdata.M;  
  
    SX x = vertcat(q, v);  
  
    // Correctly-shaped custom Jacobian: placeholder "out_tau" input,  
    // separately named jac_tau_x / jac_tau_a outputs  
    Function rneaJac("jac_rnea",  
        {x, a, MX(1,1)==SX(1,1) ? SX(1,1) : SX(1,1)}, // placeholder for tau output  
        {horzcat(du_dq, du_dv), du_da},  
        {"x", "a", "out_tau"},  
        {"jac_tau_x", "jac_tau_a"});  
  
    return Function("rnea", {x, a}, {tau}, {"x", "a"}, {"tau"},  
        Dict{{"custom_jacobian", rneaJac}, {"jac_penalty", 0}}).expand();  
}

casadi::Function 
baseNMPC::forwardModel(/* cmodel, cdata via casadi-pinocchio C++ API */, double dt) {  
    using namespace casadi;  
  
    SX u = SX::sym("tau", cmodel.nv);  
    SX q = SX::sym("q", cmodel.nq);  
    SX v = SX::sym("v", cmodel.nv);  
  
    // ABA  
    SX ddq = cpin::aba(cmodel, cdata, q, v, u);  
  
    Function dx_f = Function("dx_f", {x, u}, {dx}, {"x", "u"}, {"dx"})
  
    SX xk = x + dt * dx_f(SXVector{x, u}).at(0);  
  
    return Function("Fk", {x, u}, {xk}, {"x0", "u"}, {"xf"}).expand();  
}
