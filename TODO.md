# cub-bim
appending hand to object frame systematically + update kinematics -> done

# gz-cub-bim
making the yarp api access cleaner
try inbuilt Ikin
try inbuilt position control directly to achieve initial grasp

# fatrop
compile simple euclidian joint tracking -> done
obtaining rnea derrivatives 1st and 2nd -> done
obtaining jac derrivatives 1st and 2nd(possibly not existant) -> semi done for 1st derivative, 2nd derivative needs to be symbolic/auto diff
wrapping Eigen in blasfeo -> in examples/eigen_to_blasfeo.cpp -> done
make use of the inverse kinematics example (its rather quick)

# gazebo
find out if there is API to get c.o.m. of object and spatial info... -> done through gz-sim-yarp-plugins

# distributed theory
apply object impedance with reduced polygonal cone force constraints in order to obtain common object velocity
possibly extend existing impedance with tangent space rotational error

# dibiman library
add runtime new reference via yarp

# pinocchio Lie Algebra tests 
screw theory reference spatial velocity calculation
