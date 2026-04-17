// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <chrono>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "torch_buffer/torch_buffer_api.hpp"
#include "robot_arm.hpp"

class RendererNode : public rclcpp::Node
{
public:
  explicit RendererNode(const rclcpp::NodeOptions & options)
  : Node("renderer", options)
  {
    this->declare_parameter<bool>("use_cuda", true);
    this->declare_parameter<int>("image_width", 1920);
    this->declare_parameter<int>("image_height", 1080);
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    width_ = this->get_parameter("image_width").as_int();
    height_ = this->get_parameter("image_height").as_int();

    renderer_ = std::make_unique<RobotArmRenderer>(width_, height_, torch::kCUDA);

    auto qos = rclcpp::QoS(1).reliable();
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image", qos);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1),
      std::bind(&RendererNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(),
      "Robot arm renderer started (%dx%d, %.1f MB, transport=%s)",
      width_, height_, width_ * height_ * 4 / 1e6,
      use_cuda_ ? "cuda" : "cpu");
  }

private:
  void timer_callback()
  {
    auto guard = torch_buffer_backend::set_stream();

    renderer_->update();
    at::Tensor frame = renderer_->render_frame();

    sensor_msgs::msg::Image msg;
    if (use_cuda_) {
      msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
        {height_, width_, 4}, torch::kByte);
      torch_buffer_backend::to_buffer(msg.data, frame);
    } else {
      size_t nbytes = static_cast<size_t>(height_) * width_ * 4;
      msg.data.resize(nbytes);
      cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
      cudaMemcpyAsync(msg.data.data(), frame.data_ptr(), nbytes,
        cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    }

    msg.header.frame_id = "render";
    msg.height = height_;
    msg.width = width_;
    msg.encoding = "bgra8";
    msg.step = width_ * 4;
    msg.is_bigendian = 0;

    publisher_->publish(msg);
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool use_cuda_;
  int width_, height_;
  std::unique_ptr<RobotArmRenderer> renderer_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(RendererNode)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RendererNode>(rclcpp::NodeOptions());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
