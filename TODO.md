# cub-bim
appending hand to object frame systematically + update kinematics -> done

# gz-cub-bim
making the yarp api access cleaner
try inbuilt Ikin
try inbuilt position control directly to achieve initial grasp

# fatrop
compile simple euclidian joint tracking -> done
obtaining rnea derivatives 1st and 2nd -> done
obtaining jac derivatives 1st and 2nd(possibly not existant) -> done first derivative
wrapping Eigen in blasfeo -> in examples/eigen_to_blasfeo.cpp -> done
make use of the inverse kinematics example (its rather quick)
implementing spatial twist tracking -> done

# gazebo
find out if there is API to get c.o.m. of object and spatial info... -> done through gz-sim-yarp-plugins

# force optimal problem
using scs convex socp solver, basic example -> done (50 ms with cone objective, 20 ms with quadratic objective) 
implement stephen boyd specialized newton fop

# dibiman library
add runtime new reference via yarp
add single arm controller gazebo for testing object interactions
add higher level state machine
add consecutive point tracking

# pinocchio Lie Algebra tests 
screw theory reference spatial velocity calculation -> done
