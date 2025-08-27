cmake_minimum_required(VERSION 3.18)
project(fastcarbs_core LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
# Header-only Eigen; fetch if not available
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

pybind11_add_module(fastcarbs_core fastcarbs_core.cpp)
target_link_libraries(fastcarbs_core PRIVATE Eigen3::Eigen)
target_compile_features(fastcarbs_core PRIVATE cxx_std_17)
# Install the extension as fastcarbs.fastcarbs_core so it ships inside the package
set_target_properties(fastcarbs_core PROPERTIES SKBUILD_MODULE_NAME fastcarbs.fastcarbs_core)

# Ensure the extension target is installed for both normal and editable installs
install(TARGETS fastcarbs_core LIBRARY DESTINATION .)

# FTZ is spooky
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND _fast_local_flags -mno-daz-ftz)
endif()

target_compile_options(fastcarbs_core PRIVATE -O3 -ffast-math -march=native -fno-unsafe-math-optimizations)
