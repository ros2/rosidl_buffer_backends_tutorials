# Copyright 2026 Open Source Robotics Foundation, Inc.
# Copyright 2026 Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.28)
project(executorch_policy_demo LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
endif()

find_package(ament_cmake REQUIRED)
# The linters and the unit test need neither CUDA nor ExecuTorch, so they run even
# when the package skips.
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_input_set test/test_input_set.cpp)
  if(TARGET test_input_set)
    target_include_directories(test_input_set PRIVATE src)
  endif()
endif()
# Without a CUDA ExecuTorch for this toolkit, the vendor reports itself not found
# with the reason, and this package builds no nodes.
find_package(executorch_cuda_vendor QUIET)
if(EXECUTORCH_CUDA_VENDOR_SKIPPED)
  message(WARNING "${PROJECT_NAME}: skipping: ${EXECUTORCH_CUDA_VENDOR_SKIPPED}.")
  ament_package()
  return()
endif()
find_package(executorch_cuda_vendor REQUIRED)
# nvcc is not on PATH everywhere, for example on Jetson, so use the toolkit's
# unless the user chose one. A chosen compiler picks the toolkit instead, once
# CUDA is enabled, so the kernel and CUDA::cudart come from the same release.
if(NOT CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()
# Machine code for every GPU generation the toolkit knows, rather than the
# compiler's oldest one, unless the user chose.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(executorch ${EXECUTORCH_VENDOR_VERSION_RANGE} REQUIRED CONFIG COMPONENTS backend_cuda extension_cuda)
find_package(executorch_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(tensor_msgs REQUIRED)

# Not a component: its exit status reports the check, and a container cannot.
add_executable(camera_node src/camera_node.cpp src/frame_kernel.cu)
# One CUDA runtime, the shared one that CUDA::cudart and the libraries use.
set_target_properties(camera_node PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(camera_node
  executorch_conversions::executorch_conversions
  CUDA::cudart
  rclcpp::rclcpp
  ${tensor_msgs_TARGETS})

add_library(${PROJECT_NAME}_policy SHARED src/policy_node.cpp)
target_link_libraries(${PROJECT_NAME}_policy
  executorch_conversions::executorch_conversions
  executorch::extension_cuda
  CUDA::cudart
  rclcpp::rclcpp
  rclcpp_components::component
  ${tensor_msgs_TARGETS}
  # The node references no symbol in these libraries. They register operators and
  # delegates when loaded, and the ExecuTorch package keeps them linked for that.
  executorch::kernels_optimized
  executorch::backend_cuda)
rclcpp_components_register_node(${PROJECT_NAME}_policy
  PLUGIN "PolicyNode"
  EXECUTABLE policy_node)

set_target_properties(${PROJECT_NAME}_policy camera_node PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(
  TARGETS ${PROJECT_NAME}_policy
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS camera_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch scripts DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_ros REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_policy_demo_launch.py TIMEOUT 240
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
endif()

ament_package()
