cmake_minimum_required(VERSION 3.14)  
project(fatrop_pinocchio_ocp CXX)
  
find_package(fatrop REQUIRED)  
find_package(blasfeo REQUIRED)  
find_package(Eigen3 REQUIRED)  
find_package(pinocchio REQUIRED)  

add_executable(nmpc_test nmpc_test.cpp)  
add_executable(fatrop_ik robot_ik_example.cpp)  
target_link_libraries(nmpc_test PRIVATE fatrop blasfeo pinocchio::pinocchio Eigen3::Eigen)
target_link_libraries(fatrop_ik PRIVATE fatrop blasfeo)
# target_include_directories(ocp_impl_example_static PRIVATE /usr/local/include/fatrop)

