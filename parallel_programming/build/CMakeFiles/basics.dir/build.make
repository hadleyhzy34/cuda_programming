# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hadley/Developments/cuda_programming/parallel_programming

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hadley/Developments/cuda_programming/parallel_programming/build

# Include any dependencies generated for this target.
include CMakeFiles/basics.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/basics.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/basics.dir/flags.make

CMakeFiles/basics.dir/src/julia_set.cu.o: CMakeFiles/basics.dir/flags.make
CMakeFiles/basics.dir/src/julia_set.cu.o: ../src/julia_set.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hadley/Developments/cuda_programming/parallel_programming/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/basics.dir/src/julia_set.cu.o"
	/usr/local/cuda-12.6/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/hadley/Developments/cuda_programming/parallel_programming/src/julia_set.cu -o CMakeFiles/basics.dir/src/julia_set.cu.o

CMakeFiles/basics.dir/src/julia_set.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/basics.dir/src/julia_set.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/basics.dir/src/julia_set.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/basics.dir/src/julia_set.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target basics
basics_OBJECTS = \
"CMakeFiles/basics.dir/src/julia_set.cu.o"

# External object files for target basics
basics_EXTERNAL_OBJECTS =

basics: CMakeFiles/basics.dir/src/julia_set.cu.o
basics: CMakeFiles/basics.dir/build.make
basics: CMakeFiles/basics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hadley/Developments/cuda_programming/parallel_programming/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable basics"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/basics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/basics.dir/build: basics

.PHONY : CMakeFiles/basics.dir/build

CMakeFiles/basics.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/basics.dir/cmake_clean.cmake
.PHONY : CMakeFiles/basics.dir/clean

CMakeFiles/basics.dir/depend:
	cd /home/hadley/Developments/cuda_programming/parallel_programming/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hadley/Developments/cuda_programming/parallel_programming /home/hadley/Developments/cuda_programming/parallel_programming /home/hadley/Developments/cuda_programming/parallel_programming/build /home/hadley/Developments/cuda_programming/parallel_programming/build /home/hadley/Developments/cuda_programming/parallel_programming/build/CMakeFiles/basics.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/basics.dir/depend

