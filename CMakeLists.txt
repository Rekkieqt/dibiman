cmake_minimum_required(VERSION 3.16)
project(dibiman VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generate position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(GAZEBO_SIMULATIONS "Build gazebo simulations" ON)
option(SNIPPET_TESTS "Build snippet tests" ON)

# Define installation directories
set(LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
set(ARCHIVE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
set(RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dibimanConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dibimanConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dibimanConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/dibimanConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/dibimanConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DIR}"
)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(casadi REQUIRED)
find_package(fatrop REQUIRED)
find_package(OpenMP REQUIRED)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# dibiman source directory
add_subdirectory(src)

add_library(dibiman::dibiman ALIAS dibiman)

# target_link_libraries(dibiman
#   PUBLIC
#     Eigen3::Eigen
#     pinocchio::pinocchio
#     fatrop::fatrop
#     OpenMP::OpenMP_CXX
# )

# compiler warnings
# target_compile_options(dibiman PRIVATE
#   $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
#   $<$<CXX_COMPILER_ID:MSVC>:/W4>
# )

# ----------------------------------------------------------------------
# Gazebo Simulations
# ----------------------------------------------------------------------
if(GAZEBO_SIMULATIONS)
    add_subdirectory(gazebo_simulations)
endif()

# ----------------------------------------------------------------------
# snippet tests folder - short scripts to test new libraries or procedures
# ----------------------------------------------------------------------
if(SNIPPET_TESTS)
    add_subdirectory(snippet_tests)
endif()

