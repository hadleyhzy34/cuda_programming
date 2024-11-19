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
CMAKE_SOURCE_DIR = /home/hadley/Developments/cuda_programming/basic_elements

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hadley/Developments/cuda_programming/basic_elements/build

# Include any dependencies generated for this target.
include CMakeFiles/utility.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/utility.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/utility.dir/flags.make

CMakeFiles/utility.dir/src/utility.cu.o: CMakeFiles/utility.dir/flags.make
CMakeFiles/utility.dir/src/utility.cu.o: ../src/utility.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hadley/Developments/cuda_programming/basic_elements/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/utility.dir/src/utility.cu.o"
	/usr/local/cuda-12.6/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/hadley/Developments/cuda_programming/basic_elements/src/utility.cu -o CMakeFiles/utility.dir/src/utility.cu.o

CMakeFiles/utility.dir/src/utility.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/utility.dir/src/utility.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/utility.dir/src/utility.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/utility.dir/src/utility.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target utility
utility_OBJECTS = \
"CMakeFiles/utility.dir/src/utility.cu.o"

# External object files for target utility
utility_EXTERNAL_OBJECTS =

libutility.a: CMakeFiles/utility.dir/src/utility.cu.o
libutility.a: CMakeFiles/utility.dir/build.make
libutility.a: CMakeFiles/utility.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hadley/Developments/cuda_programming/basic_elements/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library libutility.a"
	$(CMAKE_COMMAND) -P CMakeFiles/utility.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/utility.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/utility.dir/build: libutility.a

.PHONY : CMakeFiles/utility.dir/build

CMakeFiles/utility.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/utility.dir/cmake_clean.cmake
.PHONY : CMakeFiles/utility.dir/clean

CMakeFiles/utility.dir/depend:
	cd /home/hadley/Developments/cuda_programming/basic_elements/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hadley/Developments/cuda_programming/basic_elements /home/hadley/Developments/cuda_programming/basic_elements /home/hadley/Developments/cuda_programming/basic_elements/build /home/hadley/Developments/cuda_programming/basic_elements/build /home/hadley/Developments/cuda_programming/basic_elements/build/CMakeFiles/utility.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/utility.dir/depend
