cmake_minimum_required(VERSION 3.16)
project(iCubBiManual CXX)

# file(GLOB_RECURSE SOURCES src/main/*.cpp src/main/*.h)
set(SOURCES src/icub_bim.cpp)

find_package(YARP)
# find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

SET(EXECUTABLENAME biman)

include_directories(${CMAKE_SOURCE_DIR}/include)

# target_sources(main PRIVATE icub_bim.cpp)

add_executable(${EXECUTABLENAME} ${SOURCES})

# Include directories
target_include_directories(${EXECUTABLENAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${PINOCCHIO_INCLUDE_DIRS}
    ${YARP_INCLUDE_DIRS}
)

target_compile_options(${EXECUTABLENAME} PUBLIC -Wall -O0)

target_link_libraries(${EXECUTABLENAME} PRIVATE
  ${YARP_LIBRARIES} pinocchio::pinocchio Eigen3::Eigen)

# target_compile_options(${EXECUTABLENAME} PRIVATE ${PINLIB_CFLAGS_OTHER})
# target_link_options(${EXECUTABLENAME} PRIVATE ${PINLIB_LDFLAGS})

