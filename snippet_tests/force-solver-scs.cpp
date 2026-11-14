#include <iostream>

#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "scs.h"
// #include "glbopts.h"  
// #include "scs_matrix.h"  
// #include "util.h"  

static const char *friction_socp(void);

int
main()
{
    using namespace Eigen;
    using namespace pinocchio;

    Model model;
    buildModels::humanoidRandom(model);

    Data data(model);

    model.lowerPositionLimit.head<3>().fill(-1.);
    model.upperPositionLimit.head<3>().fill(1.);

    VectorXd q = randomConfiguration(model);
    VectorXd v(VectorXd::Random(model.nv));
    VectorXd a(VectorXd::Random(model.nv));

    std::cout << friction_socp() << std::endl;
}
  
/*  
 * minimize  Fmax  
 * s.t.      A_grasp f + w_ext = 0        (6 equality rows -> zero cone)  
 *           Fmax >= f^(i)_3, i = 1,2     (2 rows -> positive orthant)  
 *           f^(i) in K_i,  K_i: sqrt(f1^2+f2^2) <= mu_i f3   (SOC_3 each)  
 *  
 * Variables x = [f1x f1y f1z | f2x f2y f2z | Fmax]   (n = 7)  
 * Contacts at r1 = (1,0,0), r2 = (-1,0,0); mu = 0.5; w_ext = (0,0,-10,0,0,0)  
 *  
 * SCS form: s = b - A x in K, so cone rows carry NEGATED coefficients,  
 * and each SOC block is ordered (mu*fz, fx, fy) to match proj_soc's (t, x).  
 */  
static const char *friction_socp(void) {  
  ScsCone *k = (ScsCone *)calloc(1, sizeof(ScsCone));  
  ScsData *d = (ScsData *)calloc(1, sizeof(ScsData));  
  ScsSettings *stgs = (ScsSettings *)calloc(1, sizeof(ScsSettings));  
  ScsSolution *sol = (ScsSolution *)calloc(1, sizeof(ScsSolution));  
  ScsInfo info = {0};  
  
  /* A (14 x 7, CSC). Rows:  
   *  0-5   zero cone:  A_grasp f = -w_ext  
   *  6-7   linear:     Fmax - f_i3 >= 0  
   *  8-10  SOC_3 #1:   (0.5 f1z, f1x, f1y)  
   *  11-13 SOC_3 #2:   (0.5 f2z, f2x, f2y)  
   */  
  scs_float Ax[] = {  
      /* f1x */ 1., -1.,  
      /* f1y */ 1., 1., -1.,  
      /* f1z */ 1., -1., -1., -0.5,  
      /* f2x */ 1., -1.,  
      /* f2y */ 1., -1., -1.,  
      /* f2z */ 1., 1., -1., -0.5,  
      /* Fmax*/ 1., 1.};  
  scs_int Ai[] = {0, 9, 1, 5, 10, 2, 4, 6, 8, 0, 12, 1, 5, 13, 2, 4, 7, 11, 6, 7};  
  scs_int Ap[] = {0, 2, 5, 9, 11, 14, 18, 20};  
  
  scs_float b[] = {0., 0., 10., 0., 0., 0.,   /* -w_ext */  
                   0., 0.,                    /* Fmax rows */  
                   0., 0., 0., 0., 0., 0.};   /* SOC rows */  
  scs_float c[] = {0., 0., 0., 0., 0., 0., 1.}; /* minimize Fmax */  
  
  scs_int q[] = {3, 3};  
  scs_int m = 14, n = 7;  
  
  d->m = m; d->n = n; d->b = b; d->c = c;  
  d->A = (ScsMatrix *)calloc(1, sizeof(ScsMatrix));  
  d->A->m = m; d->A->n = n;  
  d->A->x = Ax; d->A->i = Ai; d->A->p = Ap;  
  
  k->z = 6;        /* equality rows first */  
  k->l = 2;        /* then linear rows   */  
  k->q = q;        /* then SOC blocks    */  
  k->qsize = 2;  
  
  scs_set_default_settings(stgs);  
  stgs->eps_abs = 1e-7;  
  stgs->eps_rel = 1e-7;  
  
  scs_int exitflag = scs(d, k, stgs, sol, &info);  
  /* expected: f1z = f2z = 5, Fmax = 5 */  
  
  free(sol);  
  free(d->A); 
  free(k); 
  free(stgs); 
  free(d);  
  return exitflag == SCS_SOLVED ? 0 : "solve failed";  
}

