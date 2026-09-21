cmake_minimum_required(VERSION 3.14)  
project(fatrop_robotics_ocp CXX)  
  
find_package(fatrop REQUIRED)  
find_package(blasfeo REQUIRED)  
find_package(Eigen3 REQUIRED)  
find_package(pinocchio REQUIRED)  

add_executable(simple_nmpc single_manip_nmpc.cpp)  
target_link_libraries(simple_nmpc PRIVATE fatrop blasfeo pinocchio::pinocchio)
# target_include_directories(ocp_impl_example_static PRIVATE /usr/local/include/fatrop)

