cmake_minimum_required(VERSION 3.20)
project(iCubBiManual CXX)

# set(SOURCES src/icub_bim.cpp src/nmpc/*.cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(YARP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(casadi REQUIRED)

# simulator exe name and path
set(EXECUTABLE_NAME cub-bim)

# testing exe name and path
set(GAZEBO_EXECUTABLE_NAME gz-cub-bim)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE COMMON_SOURCES src/*.cpp)

list(REMOVE_ITEM COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/gz-cub-bim.cpp
    ${CMAKE_SOURCE_DIR}/src/cub-bim.cpp
)

# =============== Gazebo Simulator Executable =====================
add_executable(${GAZEBO_EXECUTABLE_NAME} src/gz-cub-bim.cpp ${COMMON_SOURCES})

# Include directories
target_include_directories(${GAZEBO_EXECUTABLE_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(${GAZEBO_EXECUTABLE_NAME} PUBLIC -Wall -O0)

target_link_libraries(${GAZEBO_EXECUTABLE_NAME} PRIVATE
  ${YARP_LIBRARIES} casadi::casadi pinocchio::pinocchio Eigen3::Eigen)

# =============== Normal Testing Executable =====================
add_executable(${EXECUTABLE_NAME} src/cub-bim.cpp ${COMMON_SOURCES})

# Include directories
target_include_directories(${EXECUTABLE_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(${EXECUTABLE_NAME} PUBLIC -Wall -O0)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE
  casadi::casadi pinocchio::pinocchio Eigen3::Eigen)

# For debugging libraries
# message(STATUS "CASADI_INCLUDE_DIRS = ${CASADI_INCLUDE_DIRS}")  
# get_target_property(CASADI_LOC casadi::casadi LOCATION)  
# message(STATUS "casadi::casadi library = ${CASADI_LOC}")
# target_compile_options(${EXECUTABLENAME} PRIVATE ${PINLIB_CFLAGS_OTHER})
# target_link_options(${EXECUTABLENAME} PRIVATE ${PINLIB_LDFLAGS})

