cmake_minimum_required(VERSION 3.14)  
project(fatrop_pinocchio_ocp CXX)
  
find_package(fatrop REQUIRED)  
find_package(blasfeo REQUIRED)  
find_package(Eigen3 REQUIRED)  
find_package(pinocchio REQUIRED)  
find_package(OpenMP REQUIRED)
find_package(scs REQUIRED)

add_executable(nmpc_test nmpc_test.cpp)
add_executable(fatrop_ik robot_ik_example.cpp)
add_executable(traj-test trajectory-gen.cpp)
add_executable(fop-test force-solver-scs.cpp)

set(DIBIMAN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_include_directories(nmpc_test PRIVATE
    ${DIBIMAN_ROOT}/include/utils/robot.hpp) 

file(GLOB DIBIMAN_UTILS_SRC ${DIBIMAN_ROOT}/src/utils/robot.cpp)

target_sources(nmpc_test PRIVATE ${DIBIMAN_UTILS_SRC})

target_link_libraries(nmpc_test PRIVATE fatrop blasfeo pinocchio::pinocchio Eigen3::Eigen OpenMP::OpenMP_CXX)
target_link_libraries(fatrop_ik PRIVATE fatrop blasfeo)
target_link_libraries(traj-test PRIVATE pinocchio::pinocchio Eigen3::Eigen)
target_link_libraries(fop-test PRIVATE pinocchio::pinocchio Eigen3::Eigen scs::scsdir)
# target_include_directories(ocp_impl_example_static PRIVATE /usr/local/include/fatrop)
