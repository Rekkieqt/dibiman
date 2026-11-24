# Define source files
set(DIBIMAN_SOURCES
    nmpc/baseNMPC.cpp
    nmpc/centralizedNMPC.cpp
    nmpc/distr-nmpc.cpp
    nmpc/manipulator-nmpc.cpp
    fatrop/decentralized-fatrop-helper.cpp
    fatrop/static-decentralized-ocp.cpp
    utils/robot.cpp
    utils/trajectory-generation.cpp
    utils/utils.cpp
)

# Create the fatrop library (shared or static, per BUILD_SHARED_LIBS)
add_library(dibiman ${DIBIMAN_SOURCES})
target_compile_definitions(dibiman PRIVATE DIBIMAN_VERSION="${PROJECT_VERSION}")

# Set properties for the fatrop target
set_target_properties(dibiman PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Link fatrop with blasfeo
target_link_libraries(dibiman PUBLIC fatrop::fatrop)
target_link_libraries(dibiman PUBLIC pinocchio::pinocchio)
target_link_libraries(dibiman PUBLIC Eigen3::Eigen)
target_link_libraries(dibiman PUBLIC casadi::casadi)
target_link_libraries(dibiman PUBLIC OpenMP::OpenMP_CXX)

# Specify include directories for the target
target_include_directories(dibiman
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


# Define install targets
set(INSTALL_TARGETS dibiman)

# Export the target
install(TARGETS ${INSTALL_TARGETS}
    EXPORT dibimanTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install header files (only .hpp)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/dibiman
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

# Install the export
install(EXPORT dibimanTargets
        FILE dibimanTargets.cmake
        NAMESPACE dibiman::
        DESTINATION ${CMAKE_INSTALL_DIR})

