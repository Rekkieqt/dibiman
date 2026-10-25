const Eigen::Index nv = model.nv;  
  
// View the tensor as nv x nv^2: column i = vec( d2tau_dqdq(i,:,:) )  
Eigen::Map<const Eigen::MatrixXd> T_qq(_data.d2tau_dqdq.data(), nv, nv * nv);  
Eigen::Map<const Eigen::MatrixXd> T_vv(_data.d2tau_dvdv.data(), nv, nv * nv);  
Eigen::Map<const Eigen::MatrixXd> T_qv(_data.d2tau_dqdv.data(), nv, nv * nv);  
Eigen::Map<const Eigen::MatrixXd> T_aq(_data.d2tau_dadq.data(), nv, nv * nv);  
  
// Contract: result is 1 x nv^2, layout (j,k) with j fastest -> remap as nv x nv  
Eigen::VectorXd flat_qq = _lam_eq.transpose() * T_qq;          // 1 x nv^2  
dtau_dux._mat_qq = Eigen::Map<Eigen::MatrixXd>(flat_qq.data(), nv, nv);  
  
Eigen::VectorXd flat_vv = _lam_eq.transpose() * T_vv;  
dtau_dux._mat_vv = Eigen::Map<Eigen::MatrixXd>(flat_vv.data(), nv, nv);  
  
Eigen::VectorXd flat_qv = _lam_eq.transpose() * T_qv;          // comes out as (v,q)  
dtau_dux._mat_qv = Eigen::Map<Eigen::MatrixXd>(flat_qv.data(), nv, nv).transpose();  
  
Eigen::VectorXd flat_aq = _lam_eq.transpose() * T_aq;  
dtau_dux._mat_aq = Eigen::Map<Eigen::MatrixXd>(flat_aq.data(), nv, nv);

Eigen::Map<Eigen::RowVectorXd>(dtau_dux._mat_qq.data(), nv*nv) = _lam_eq.transpose() * T_qq;
