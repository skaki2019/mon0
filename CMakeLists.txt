# Set the minimum CMake version
cmake_minimum_required(VERSION 3.15...3.27)

# Set up the project, specifying C++ as the language
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Warn users if they try to invoke CMake directly
if (NOT SKBUILD)
  message(WARNING "This project is designed to be built with scikit-build-core. Please run `pip install .` from the project root.")
endif()

# Find Python and the required development components
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# Find the installed nanobind package
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR
)

find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
# find_package(OpenMP REQUIRED)

# --- tsl-robin-map (header-only, may lack CMake config in conda) ---
find_package(tsl_robin_map CONFIG QUIET)
if(NOT robin_map_FOUND)
    # Try conda's include directory directly
    if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/tsl/robin_map.h")
        message(STATUS "Using tsl-robin-map from conda: $ENV{CONDA_PREFIX}/include")
        add_library(tsl_robin_map INTERFACE IMPORTED)
        target_include_directories(tsl_robin_map INTERFACE $ENV{CONDA_PREFIX}/include)
        # Create the expected target alias for consistency
        if(NOT TARGET tsl::robin_map)
            add_library(tsl::robin_map ALIAS tsl_robin_map)
        endif()
    # else()
    #     # Fallback to FetchContent for CI/CD
    #     message(STATUS "tsl-robin-map not found, fetching from GitHub...")
    #     include(FetchContent)
    #     FetchContent_Declare(tsl_robin_map
    #         GIT_REPOSITORY https://github.com/Tessil/robin-map.git
    #         GIT_TAG        v1.3.0
    #     )
    #     FetchContent_MakeAvailable(tsl_robin_map)
    endif()
endif()

# Create the nanobind extension module
nanobind_add_module(
  _impl          # Name of the module
  # STABLE_ABI          # Uncomment to build Stable ABI wheels for Python 3.12+
  src/crs_dataset.hpp       # Source file
  src/crs_dataset.cpp
  src/transition_matrix.hpp
  src/transition_matrix.cpp
  src/multi_order_network.hpp
  src/multi_order_network.cpp
  src/utilities.hpp
  src/utilities.cpp
)

# 5. Link dependencies
target_link_libraries(_impl PRIVATE 
    Eigen3::Eigen
)

# Install the compiled library into the Python package directory
# install(TARGETS _impl LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# Install the compiled library into the Python package directory
install(TARGETS _impl LIBRARY DESTINATION mon0)

# Copy Python files - add this to your CMakeLists.txt
install(DIRECTORY src/mon0/ DESTINATION mon0)