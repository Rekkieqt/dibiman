#include <Eigen/Dense>  
#include <blasfeo.h>  
  
int m = 10, n = 8;  
Eigen::MatrixXd Ae = Eigen::MatrixXd::Random(m, n);  
Eigen::VectorXd xe = Eigen::VectorXd::Random(m);  
  
struct blasfeo_dmat sA;  
struct blasfeo_dvec sx;  
  
blasfeo_allocate_dmat(m, n, &sA);  
blasfeo_allocate_dvec(m, &sx);  
  
// Eigen -> blasfeo  
blasfeo_pack_dmat(m, n, Ae.data(), Ae.outerStride(), &sA, 0, 0);  
blasfeo_pack_dvec(m, xe.data(), xe.innerStride(), &sx, 0);  
  
// ... blasfeo computations ...  
  
// blasfeo -> Eigen  
Eigen::MatrixXd Be(m, n);  
blasfeo_unpack_dmat(m, n, &sA, 0, 0, Be.data(), Be.outerStride());  
  
blasfeo_free_dmat(&sA);  
blasfeo_free_dvec(&sx);
