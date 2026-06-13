# In order to change binary output directory...
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

cmake_minimum_required(VERSION 3.16)
SET(EXECUTABLENAME main)
project(iCubBimanual)

find_package(YARP)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PINLIB REQUIRED pinocchio)

set(SOURCES 
    icub_bim.cpp 
    )

source_group("Source Files" FILES ${SOURCES})

add_executable(${EXECUTABLENAME})

target_sources(main PRIVATE icub_bim.cpp)
target_include_directories(${EXECUTABLENAME} PRIVATE ${PINLIB_INCLUDE_DIRS} ${YARP_LIBRARIES} ../include/)
# header lib for nmpc
target_include_directories(${EXECUTABLENAME} PRIVATE ../include/)


target_link_libraries(${EXECUTABLENAME} PRIVATE
  ${YARP_LIBRARIES} ${PINLIB_LIBRARIES})

target_compile_options(${EXECUTABLENAME} PRIVATE ${PINLIB_CFLAGS_OTHER})
target_link_options(${EXECUTABLENAME} PRIVATE ${PINLIB_LDFLAGS})
