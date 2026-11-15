#include <iostream>
#include <filesystem>

#include "pinocchio/parsers/urdf.hpp"  

#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/algorithm/model.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "scs.h"

static const char *friction_socp(void);
static const char *friction_qp(void);

int
main()
{
    using namespace Eigen;
    using namespace pinocchio;

    const std::string urdf_path = static_cast<std::string>(std::filesystem::current_path()) + "/../../conf/model.urdf";

    Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    Data data(model);

    VectorXd q = neutral(model);
    forwardKinematics(model, data, q);
    updateFramePlacements(model, data);

    friction_socp();
    friction_qp();

    return 0;
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
        /* f1z */ 1., -1., 1., -0.5,  
        /* f2x */ 1., -1.,  
        /* f2y */ 1., -1., -1.,  
        /* f2z */ 1., 1., 1., -0.5,  
        /* Fmax*/ -1., -1.};  
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
    stgs->verbose = 1;
    
    scs_int exitflag = scs(d, k, stgs, sol, &info);  
    /* expected: f1z = f2z = 5, Fmax = 5 */  
    printf("status: %s (val %d), iters: %d\n", info.status, (int)info.status_val,  
         (int)info.iter);  
    printf("pobj: %.6f  dobj: %.6f  gap: %.2e\n", info.pobj, info.dobj, info.gap);  
    printf("res_pri: %.2e  res_dual: %.2e\n", info.res_pri, info.res_dual);  
    printf("res_pri: %.2e  res_dual: %.2e\n", info.res_pri, info.res_dual);  
      
    if (exitflag == SCS_SOLVED || exitflag == SCS_SOLVED_INACCURATE) {  
        for (int i = 0; i < 2; ++i)
        {
            printf("f%d = (%.4f, %.4f, %.4f)\n", i + 1,  sol->x[3*i], sol->x[3*i+1], sol->x[3*i+2]);  
            printf("Fmax = %.4f\n", sol->x[6]);  
    
        }
    }
    
    free(sol->x);
    free(sol->y);
    free(sol->s);
    free(sol);  
    free(d->A); 
    free(k); 
    free(stgs); 
    free(d); 

    return exitflag == SCS_SOLVED ? 0 : "solve failed";  
}

/*  
 * minimize  sum_i ||f^(i)||^2        ->  1/2 x'Px with P = 2*I_6  
 * s.t.      A_grasp f + w_ext = 0    (6 zero-cone rows)  
 *           f^(i) in K_i             (SOC_3 each)  
 *  
 * Variables x = [f1x f1y f1z | f2x f2y f2z]   (n = 6, m = 12)  
 */  
static const char *friction_qp(void) {  
    ScsCone *k = (ScsCone *)calloc(1, sizeof(ScsCone));  
    ScsData *d = (ScsData *)calloc(1, sizeof(ScsData));  
    ScsSettings *stgs = (ScsSettings *)calloc(1, sizeof(ScsSettings));  
    ScsSolution *sol = (ScsSolution *)calloc(1, sizeof(ScsSolution));  
    ScsInfo info = {0};  
    
    /* A (12 x 6, CSC): rows 0-5 grasp equality, 6-8 SOC#1, 9-11 SOC#2 */  
    scs_float Ax[] = {  
        /* f1x */ 1., -1.,  
        /* f1y */ 1., 1., -1.,  
        /* f1z */ 1., -0.5,  
        /* f2x */ 1., -1.,  
        /* f2y */ 1., -1., -1.,  
        /* f2z */ 1., -0.5};  
    scs_int Ai[] = {0, 7, 1, 5, 8, 2, 6, 0, 10, 1, 5, 11, 2, 9};  
    scs_int Ap[] = {0, 2, 5, 7, 9, 12, 14};  
    
    /* P = 2*I_6, upper triangle only -> just the diagonal */  
    scs_float Px[] = {2., 2., 2., 2., 2., 2.};  
    scs_int Pi[]   = {0, 1, 2, 3, 4, 5};  
    scs_int Pp[]   = {0, 1, 2, 3, 4, 5, 6};  
    
    scs_float b[] = {0., 0., 10., 0., 0., 0.,   /* -w_ext */  
                     0., 0., 0., 0., 0., 0.};   /* SOC rows */  
    scs_float c[] = {0., 0., 0., 0., 0., 0.};  
    
    scs_int q[] = {3, 3};  
    scs_int m = 12, n = 6;  
    
    d->m = m; d->n = n; d->b = b; d->c = c;  
    d->A = (ScsMatrix *)calloc(1, sizeof(ScsMatrix));  
    d->A->m = m; d->A->n = n;  
    d->A->x = Ax; d->A->i = Ai; d->A->p = Ap;  
    d->P = (ScsMatrix *)calloc(1, sizeof(ScsMatrix));  
    d->P->m = n; d->P->n = n;  
    d->P->x = Px; d->P->i = Pi; d->P->p = Pp;  
    
    k->z = 6;  
    k->q = q;  
    k->qsize = 2;  
    
    scs_set_default_settings(stgs);  
    stgs->eps_abs = 1e-7;  
    stgs->eps_rel = 1e-7;  
    
    scs_int exitflag = scs(d, k, stgs, sol, &info);  
    /* expected: f1 = f2 = (0,0,5), objective = 25 + 25 = 50 */  
    
    free(sol->x); free(sol->y); free(sol->s); free(sol);  
    free(d->A); free(d->P); free(k); free(stgs); free(d);  
    return exitflag == SCS_SOLVED ? 0 : "solve failed";  
}
