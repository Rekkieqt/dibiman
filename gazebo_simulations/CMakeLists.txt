# find_package(Eigen3 REQUIRED)
# find_package(pinocchio REQUIRED)

# set(SIMULATIONS_INSTALL_DIR )

find_package(YARP REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(pinocchio REQUIRED)

find_package(gz-sim REQUIRED)
find_package(gz-msgs REQUIRED)
find_package(gz-transport REQUIRED)

add_executable(gz-cub-bim gz-cub-bim.cpp)
add_executable(gz-cub-single gz-cub-single.cpp)
add_executable(cub-bim cub-bim.cpp)

target_link_libraries(gz-cub-bim PRIVATE
  ${YARP_LIBRARIES} dibiman pinocchio::pinocchio Eigen3::Eigen gz-sim::gz-sim gz-transport::gz-transport gz-msgs::gz-msgs)

target_link_libraries(gz-cub-single PRIVATE
  ${YARP_LIBRARIES} dibiman pinocchio::pinocchio Eigen3::Eigen gz-sim::gz-sim gz-transport::gz-transport gz-msgs::gz-msgs)

target_link_libraries(cub-bim PRIVATE
  dibiman pinocchio::pinocchio Eigen3::Eigen)
