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
# The linters need neither CUDA nor ExecuTorch, so they run even when the package skips.
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()
find_package(executorch_cuda_vendor REQUIRED)
if(EXECUTORCH_CUDA_VENDOR_SKIPPED)
  message(WARNING "${PROJECT_NAME}: skipping, because executorch_cuda_vendor installed no "
    "ExecuTorch: ${EXECUTORCH_CUDA_VENDOR_SKIPPED}.")
  ament_package()
  return()
endif()
find_package(CUDAToolkit REQUIRED)
# nvcc is not on PATH everywhere, for example on Jetson, so use the toolkit's
# unless the user chose one.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()
# Machine code for every GPU generation the toolkit knows, rather than the
# compiler's oldest one, unless the user chose.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
enable_language(CUDA)
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
  ${CMAKE_DL_LIBS}
  CUDA::cudart
  rclcpp::rclcpp
  rclcpp_components::component
  ${tensor_msgs_TARGETS}
  # The node references no symbol in these libraries. They register operators and
  # delegates when loaded, and the ExecuTorch package keeps them linked for that.
  # Other delegates load at run time from backend_libraries.
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

ament_package()
