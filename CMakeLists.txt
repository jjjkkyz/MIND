# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
add_subdirectory(pybind11)

add_subdirectory(src)
# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
# include_directories("E:\\WN_sep\\MLDF\\src\\third\\polyscope")

pybind11_add_module(_core src/my_module.cpp)
target_link_libraries(_core PRIVATE WN_API)
# target_compile_features(_core PRIVATE cxx_std_20)
# target_link_libraries(_core PRIVATE polyscope)
# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION mldf)