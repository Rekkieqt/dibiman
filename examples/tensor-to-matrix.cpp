// Source - https://stackoverflow.com/a/54536187
// Posted by thclark
// Retrieved 2026-10-21, License - CC BY-SA 4.0

Eigen::Tensor<double, 3> my_tens(2, 3, 4);
my_tens.setRandom();
Eigen::array<Eigen::Index, 3> dims = my_tens.dimensions();
Eigen::array<Eigen::Index, 3> offsets = {0, 1, 0};
Eigen::array<Eigen::Index, 3> extents = {dims[0], 1, 1};

// Compute the offset, correctly this time!
auto page_offset = offsets[2] * dims[0] * dims[1];
auto col_offset = offsets[1] * dims[0];
auto elements_offset = page_offset + col_offset;

// Map into the array
Eigen::Map<Eigen::VectorXd> my_mapped_vec(my_tens.data() + elements_offset, dims[0]);

// Compare the two:
std::cout << my_tens.slice(offsets, extents) << std::endl;
std::cout << my_mapped_vec.transpose() << std::endl;

// deep wiki
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
Eigen::VectorXd flat_qq = _lam_eq.transpose() * T_qq;                 // length nv*nv, flattened (j fastest)  
Eigen::Map<Eigen::MatrixXd> Hqq(flat_qq.data(), nv, nv);               // Hqq(j,k) = sum_i lam_i * d2tau_dqdq(i,j,k)  
// coulc call blasfeo_pack_mat(...)
dtau_dux._mat_qq = Hqq;      
