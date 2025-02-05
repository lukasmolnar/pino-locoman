cmake_minimum_required(VERSION 3.10)
project(CompiledSolver)

# Set C and C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_library(FATROP_LIB fatrop REQUIRED)
find_library(BLASFEO_LIB blasfeo REQUIRED)

# Add blasfeo include directory
include_directories(/opt/blasfeo/include)

# Add the source file
set(SOURCE_FILE compiled_solver.c)

# Create the shared library
add_library(compiled_solver SHARED ${SOURCE_FILE})

# Link the libraries
target_link_libraries(compiled_solver PRIVATE ${FATROP_LIB} ${BLASFEO_LIB})

# Compilation flags
target_compile_options(compiled_solver PRIVATE -O3 -fPIC)

# Install the shared library
install(TARGETS compiled_solver LIBRARY DESTINATION lib)
