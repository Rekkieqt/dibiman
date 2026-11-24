#include <Eigen/Dense>

#include "dibiman/utils/robot.hpp"
#include "dibiman/utils/utils.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
#include "pinocchio/algorithm/frames.hpp"  
#include "pinocchio/algorithm/rnea.hpp"  
#include "pinocchio/algorithm/model.hpp"  
#include "pinocchio/algorithm/default-check.hpp"
#include "pinocchio/autodiff/casadi.hpp"

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
        const pinocchio::FrameIndex & frame_id,  
        const std::string & target)  
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
    returnReducedModel(const std::string & modelpath,  
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

        Model reduced_model = buildReducedModel(model, list_joints_to_lock_id, q_neutral);

        return reduced_model;
    };

    casadi::Function
    descreteABA(const pinocchio::Model & model, const double & dt)
    {
        using namespace pinocchio;

        typedef double Scalar;
        typedef ::casadi::SX ADScalar;

        typedef ModelTpl<Scalar> Model;

        typedef ModelTpl<ADScalar> ADModel;
        typedef ADModel::Data ADData;

        // Pick up random configuration, velocity and acceleration vectors.
        Eigen::VectorXd q(model.nq);
        q = randomConfiguration(model);
        Eigen::VectorXd v(Eigen::VectorXd::Random(model.nv));
        Eigen::VectorXd tau(Eigen::VectorXd::Random(model.nv));

        // Create CasADi model and data from model
        typedef ADModel::ConfigVectorType ConfigVectorAD;
        typedef ADModel::TangentVectorType TangentVectorAD;
        ADModel ad_model = model.cast<ADScalar>();
        ADData ad_data(ad_model);

        // Create symbolic CasADi vectors
        ::casadi::SX cs_q = ::casadi::SX::sym("q", model.nq);
        ConfigVectorAD q_ad(model.nq);
        q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADScalar>>(cs_q).data(), model.nq, 1);

        ::casadi::SX cs_v = ::casadi::SX::sym("v", model.nv);
        TangentVectorAD v_ad(model.nv);
        v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADScalar>>(cs_v).data(), model.nv, 1);

        ::casadi::SX cs_tau = ::casadi::SX::sym("tau", model.nv);
        TangentVectorAD tau_ad(model.nv);
        tau_ad =
        Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADScalar>>(cs_tau).data(), model.nv, 1);

        // Build CasADi function
        aba(ad_model, ad_data, q_ad, v_ad, tau_ad);
        ::casadi::SX a_ad(model.nv, 1);

        for (Eigen::Index k = 0; k < model.nv; ++k) a_ad(k) = ad_data.ddq[k];

        // ::casadi::Function eval_aba("eval_aba", ::casadi::SXVector{cs_q, cs_v, cs_tau}, ::casadi::SXVector{a_ad});
        ::casadi::SX cs_x = ::casadi::SX::vertcat({cs_q, cs_v});
        ::casadi::Function eval_aba("aba", ::casadi::SXVector{cs_x, cs_tau}, ::casadi::SXVector{a_ad});

        ::casadi::SX dx = ::casadi::SX::vertcat({cs_v, a_ad});

        ::casadi::SXDict dae = {{"x", cs_x}, {"p", cs_tau}, {"ode", dx}};

        ::casadi::Dict integrator_opts;
        integrator_opts["simplify"] = true;
        integrator_opts["number_of_finite_elements"] = 1;
        
        ::casadi::Function descrete_aba = ::casadi::integrator("discr_aba", "rk", dae, 0, dt, integrator_opts);
        
        return descrete_aba;
    };

    Eigen::VectorXd descreteForwardDynamics(const pinocchio::Model & model, const Eigen::VectorXd & xk, const Eigen::VectorXd & uk, const double & dt)
    {
        using namespace casadi;
        static casadi::Function aba_dt = descreteABA(model, dt);
        DM _xk = DM::zeros(xk.size());
        std::memcpy(_xk.ptr(), xk.data(), sizeof(double) * xk.size());
        DM _uk = DM::zeros(uk.size());
        std::memcpy(_uk.ptr(), uk.data(), sizeof(double) * uk.size());
        DMDict args = {{"x0", _xk}, {"p", _uk}};
        // std::vector <casadi::DM> args = {_xk, _uk};
        DM _xk_next = aba_dt(args).at("xf");
        // Eigen::VectorXd xk_next(_xk_next.ptr(), _xk_next.size());
        Eigen::VectorXd xk_next = Eigen::VectorXd::Zero(xk.size());
        std::copy(_xk_next.ptr(), _xk_next.ptr() + _xk_next.numel(), xk_next.data());
        return xk_next;
    }

    casadi::Function
    linearDynamics(const int n, const double dt)
    {
        using namespace casadi;

        SX q = SX::sym("q", n);
        SX v = SX::sym("v", n);
        SX a = SX::sym("a", n);
      
        SX x = vertcat(q, v);
        SX dx = vertcat(v, a);
      
        SX qk = q + dt * v;
        SX vk = v + dt * a;
        SX xk = vertcat(qk, vk);
      
        return Function("Linear_joint_dynamics", {x, a}, {xk}, {"x", "a"}, {"xk"}).expand();
    };

    casadi::Function
    rneaSymbolic(const pinocchio::Model & model)
    {  
        using namespace pinocchio;
        typedef ::casadi::SX ADScalar;
        const int nv = model.nv;
        
        typedef pinocchio::ModelTpl<ADScalar> ADModel;  
        typedef ADModel::Data ADData;  
        
        ADModel ad_model = model.cast<ADScalar>();  
        ADData ad_data(ad_model);  
        
        typedef ADModel::ConfigVectorType ConfigVectorAD;  
        typedef ADModel::TangentVectorType TangentVectorAD;  
        
        ::casadi::SX cs_q = ::casadi::SX::sym("q", nv);  
        ConfigVectorAD q_ad(nv);  
        q_ad = Eigen::Map<ConfigVectorAD>(  
          static_cast<std::vector<ADScalar>>(cs_q).data(), nv, 1);  
        
        ::casadi::SX cs_v = ::casadi::SX::sym("v", nv);  
        TangentVectorAD v_ad(nv);
        v_ad = Eigen::Map<TangentVectorAD>(  
          static_cast<std::vector<ADScalar>>(cs_v).data(), nv, 1);  
        
        ::casadi::SX cs_a = ::casadi::SX::sym("a", nv);  
        TangentVectorAD a_ad(nv);  
        a_ad = Eigen::Map<TangentVectorAD>(  
          static_cast<std::vector<ADScalar>>(cs_a).data(), nv, 1);  
        
        pinocchio::rnea(ad_model, ad_data, q_ad, v_ad, a_ad);  
        
        ::casadi::SX tau_ad(nv, 1);  
        for (Eigen::Index k = 0; k < nv; ++k)  
          tau_ad(k) = ad_data.tau[k];  
        
        ::casadi::SX cs_x = ::casadi::SX::vertcat({cs_q, cs_v});
        return ::casadi::Function("rnea", ::casadi::SXVector{cs_x, cs_a}, ::casadi::SXVector{tau_ad});  
    };
}

