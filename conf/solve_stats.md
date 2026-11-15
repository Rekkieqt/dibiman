``` without openmp 
operation                               time (s)   share
--------------------------------------------------------
fatrop                                  0.011584   57.0%
  initialization                        0.000518
  compute search dir                    0.009679
  compute obj value                     0.000000
  compute obj gradient                  0.000002
  compute constr viol                   0.000035
  compute hessian                       0.000001
  compute jacobian                      0.000001
  compute barrier value                 0.000005
  compute barrier gradient              0.000001
  compute delta lower                   0.000000
  compute delta upper                   0.000003
  compute dual infeasibility x          0.000362
  compute dual infeasibility s          0.000010
  compute complementarity l             0.000000
  compute complementarity u             0.000001
  compute relaxed compl l               0.000007
  compute relaxed compl u               0.000003
  compute primal damping                0.000010
  compute linear decrease obj           0.000075
  compute linear decrease bar           0.000001
  compute e_mu                          0.000044
  compute max step primal               0.000002
  compute max step dual                 0.000000
  compute modify dual bounds            0.000005
  rest                                  0.000819
function evaluation                     0.008741   43.0%
  eval objective                        0.000101
  eval gradient                         0.000097
  eval constraint violation             0.000915
  eval hessian                          0.000323
  eval jacobian                         0.007305
--------------------------------------------------------
total                                   0.020325

```
``` openmp on h(uk, xk) = 0
operation                               time (s)   share
--------------------------------------------------------
fatrop                                  0.009063   53.2%
  initialization                        0.000419
  compute search dir                    0.007445
  compute obj value                     0.000002
  compute obj gradient                  0.000000
  compute constr viol                   0.000034
  compute hessian                       0.000000
  compute jacobian                      0.000000
  compute barrier value                 0.000005
  compute barrier gradient              0.000002
  compute delta lower                   0.000008
  compute delta upper                   0.000002
  compute dual infeasibility x          0.000282
  compute dual infeasibility s          0.000009
  compute complementarity l             0.000002
  compute complementarity u             0.000001
  compute relaxed compl l               0.000004
  compute relaxed compl u               0.000001
  compute primal damping                0.000009
  compute linear decrease obj           0.000062
  compute linear decrease bar           0.000002
  compute e_mu                          0.000041
  compute max step primal               0.000002
  compute max step dual                 0.000001
  compute modify dual bounds            0.000002
  rest                                  0.000728
function evaluation                     0.007988   46.8%
  eval objective                        0.000075
  eval gradient                         0.000086
  eval constraint violation             0.001948
  eval hessian                          0.000273
  eval jacobian                         0.005606
--------------------------------------------------------
total                                   0.017051
```

``` open mp on h(uk, xk) and dh/d(uk, xk)
operation                               time (s)   share
--------------------------------------------------------
fatrop                                  0.011932   80.5%
  initialization                        0.000490
  compute search dir                    0.009784
  compute obj value                     0.000001
  compute obj gradient                  0.000001
  compute constr viol                   0.000033
  compute hessian                       0.000001
  compute jacobian                      0.000003
  compute barrier value                 0.000005
  compute barrier gradient              0.000003
  compute delta lower                   0.000000
  compute delta upper                   0.000000
  compute dual infeasibility x          0.000592
  compute dual infeasibility s          0.000015
  compute complementarity l             0.000001
  compute complementarity u             0.000000
  compute relaxed compl l               0.000004
  compute relaxed compl u               0.000002
  compute primal damping                0.000009
  compute linear decrease obj           0.000074
  compute linear decrease bar           0.000005
  compute e_mu                          0.000054
  compute max step primal               0.000004
  compute max step dual                 0.000001
  compute modify dual bounds            0.000001
  rest                                  0.000849
function evaluation                     0.002885   19.5%
  eval objective                        0.000104
  eval gradient                         0.000098
  eval constraint violation             0.000811
  eval hessian                          0.000321
  eval jacobian                         0.001551
--------------------------------------------------------
total                                   0.014817

```
