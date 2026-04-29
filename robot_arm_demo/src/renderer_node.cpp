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
#include <torch/torch.h>

#include <chrono>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/tensor.hpp"
#include "robot_arm.hpp"

class RendererNode : public rclcpp::Node
{
public:
  explicit RendererNode(const rclcpp::NodeOptions & options)
  : Node("renderer", options)
  {
    this->declare_parameter<int>("image_width", 1920);
    this->declare_parameter<int>("image_height", 1080);
    width_ = this->get_parameter("image_width").as_int();
    height_ = this->get_parameter("image_height").as_int();

    renderer_ = std::make_unique<RobotArmRenderer>(width_, height_, torch::kCUDA);

    auto qos = rclcpp::QoS(1).reliable();
    publisher_ = this->create_publisher<tensor_msgs::msg::Tensor>("image", qos);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1),
      std::bind(&RendererNode::timer_callback, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Robot arm renderer started (%dx%d, %.1f MB, transport=torch_conversions)",
      width_, height_, width_ * height_ * 4 / 1e6);
  }

private:
  void timer_callback()
  {
    auto guard = torch_conversions::set_stream();

    renderer_->update();
    at::Tensor frame = renderer_->render_frame();

    tensor_msgs::msg::Tensor msg = torch_conversions::allocate_tensor_msg(
      {height_, width_, 4}, torch::kByte);
    torch_conversions::to_tensor_msg(msg, frame);

    publisher_->publish(msg);
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  rclcpp::Publisher<tensor_msgs::msg::Tensor>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
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
