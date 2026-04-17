cmake_minimum_required(VERSION 3.20)
project(robot_arm_demo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(torch_buffer REQUIRED)
find_package(torch_buffer_backend REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(rclcpp_components REQUIRED)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(X11 REQUIRED)

# Renderer node (publisher)
add_library(renderer_component SHARED
  src/renderer_node.cpp
  src/robot_arm.cpp
  src/font.cpp
)
target_link_libraries(renderer_component
  ${TORCH_LIBRARIES}
  torch_buffer::torch_buffer
  torch_buffer_backend::torch_buffer_backend
  rclcpp::rclcpp
  rclcpp_components::component
  ${sensor_msgs_TARGETS}
  CUDA::cudart
  CUDA::cuda_driver
)
target_compile_options(renderer_component PRIVATE ${TORCH_CXX_FLAGS})
rclcpp_components_register_node(renderer_component
  PLUGIN "RendererNode"
  EXECUTABLE renderer_node
)
install(TARGETS renderer_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Display node (subscriber, SDL2/GLEW/OpenGL + CUDA-GL interop)
add_library(display_component SHARED
  src/display_node.cpp
  src/display.cpp
  src/font.cpp
)
target_compile_definitions(display_component PRIVATE USE_CUDA_GL_INTEROP)
target_link_libraries(display_component
  ${TORCH_LIBRARIES}
  torch_buffer::torch_buffer
  torch_buffer_backend::torch_buffer_backend
  rclcpp::rclcpp
  rclcpp_components::component
  ${sensor_msgs_TARGETS}
  CUDA::cudart
  CUDA::cuda_driver
  OpenGL::GL
  SDL2::SDL2
  GLEW::GLEW
  X11::X11
)
target_compile_options(display_component PRIVATE ${TORCH_CXX_FLAGS})
rclcpp_components_register_node(display_component
  PLUGIN "DisplayNode"
  EXECUTABLE display_node
)
install(TARGETS display_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
