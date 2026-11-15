
**************************************************************************
   This program contains Fatrop 1.2.99, a nonlinear optimization solver
                    for optimal control and robotics.
                 Copyright (c) Lander Vanroye, KU Leuven.
         Fatrop is dual licensed under BSD-2-Clause and EPL-2.0.
     For more information visit https://github.com/meco-group/fatrop
**************************************************************************

iter    objective   inf_pr   inf_du lg(mu)    ||d||   lg(rg)   alpha_du   alpha_pr ls
   0  0.00000000e+00 1.88e-01 0.00e+00   -1.0 0.00e+00      -  0.00e+00   0.00e+00x  1
   1  1.73198783e+00 1.52e-03 6.93e-03   -2.5 0.00e+00      -  1.00e+00   1.00e+00h  1
   2  1.73099178e+00 4.68e-05 6.45e-04   -3.8 1.87e-01      -  1.00e+00   1.00e+00h  1
   3  1.73094421e+00 5.17e-07 5.50e-06   -5.7 2.14e-03      -  1.00e+00   1.00e+00h  1
   4  1.73094416e+00 3.56e-08 5.22e-07   -8.6 6.76e-05      -  1.00e+00   1.00e+00h  1
   5  1.73094412e+00 3.08e-10 4.53e-09   -8.6 1.56e-06      -  1.00e+00   1.00e+00h  1
Return flag: 0
Return flag == success: 1
operation                               time (s)   share
--------------------------------------------------------
fatrop                                  0.002251   70.1%
  initialization                        0.000478
  compute search dir                    0.001394
  compute obj value                     0.000000
  compute obj gradient                  0.000000
  compute constr viol                   0.000007
  compute hessian                       0.000001
  compute jacobian                      0.000000
  compute barrier value                 0.000003
  compute barrier gradient              0.000000
  compute delta lower                   0.000000
  compute delta upper                   0.000001
  compute dual infeasibility x          0.000117
  compute dual infeasibility s          0.000004
  compute complementarity l             0.000001
  compute complementarity u             0.000001
  compute relaxed compl l               0.000001
  compute relaxed compl u               0.000002
  compute primal damping                0.000003
  compute linear decrease obj           0.000012
  compute linear decrease bar           0.000001
  compute e_mu                          0.000011
  compute max step primal               0.000000
  compute max step dual                 0.000000
  compute modify dual bounds            0.000001
  rest                                  0.000213
function evaluation                     0.000961   29.9%
  eval objective                        0.000020
  eval gradient                         0.000019
  eval constraint violation             0.000488
  eval hessian                          0.000063
  eval jacobian                         0.000371
--------------------------------------------------------
total                                   0.003212

