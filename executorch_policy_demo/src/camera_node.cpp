// Copyright 2026 Meta Platforms, Inc. and affiliates.
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

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "executorch_conversions/executorch_conversions.hpp"
#include "frame_kernel.hpp"
#include "rclcpp/rclcpp.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;
using executorch_policy_demo::FrameFormat;
using TensorMsg = tensor_msgs::msg::ExperimentalTensor;
using Clock = std::chrono::steady_clock;

namespace
{

void check_cuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("camera_node: ") + cudaGetErrorString(result));
  }
}

double rss_mb()
{
  std::ifstream statm("/proc/self/statm");
  int64_t pages = 0;
  int64_t resident = 0;
  statm >> pages >> resident;
  return static_cast<double>(resident * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
}

}  // namespace

class CameraNode : public rclcpp::Node
{
public:
  explicit CameraNode(const rclcpp::NodeOptions & options)
  : Node("camera", options)
  {
    const auto width = declare_parameter<int>("width", 640);
    const auto height = declare_parameter<int>("height", 480);
    const auto channels = declare_parameter<int>("channels", 3);
    if (width <= 0 || height <= 0 || channels <= 0) {
      throw std::invalid_argument(
              "width, height and channels must be positive, got " + std::to_string(width) +
              ", " + std::to_string(height) + " and " + std::to_string(channels));
    }
    const auto format = declare_parameter<std::string>("format", "float_chw");
    const auto transport = declare_parameter<std::string>("transport", "cuda");
    state_shape_ = declare_parameter<std::vector<int64_t>>("state_shape", {1, 14});
    state_count_ = 1;
    for (const auto dimension : state_shape_) {
      // ExecuTorch sizes are 32-bit.
      if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max() ||
        (dimension > 0 &&
        state_count_ > static_cast<size_t>(std::numeric_limits<int32_t>::max() / dimension)))
      {
        throw std::invalid_argument("state_shape must not be negative or exceed 32 bits");
      }
      state_count_ *= static_cast<size_t>(dimension);
    }
    // The launch file takes only [0] to mean no state, so another empty shape would leave
    // the policy waiting for a state that never comes.
    if (state_count_ == 0 && state_shape_ != std::vector<int64_t>{0}) {
      throw std::invalid_argument("state_shape must be [0] for no state, or have no zero size");
    }
    frames_ = declare_parameter<int>("frames", 1000);
    warmup_ = declare_parameter<int>("warmup", 50);
    if (frames_ <= 0 || warmup_ < 0) {
      throw std::invalid_argument("frames must be positive and warmup not negative");
    }
    const auto expected_file = declare_parameter<std::string>("expected_file_path", "");
    if (!expected_file.empty()) {
      std::ifstream file(expected_file, std::ios::binary | std::ios::ate);
      if (!file) {
        throw std::invalid_argument("cannot read " + expected_file);
      }
      const auto bytes = static_cast<size_t>(file.tellg());
      // One float output block per frame of the period, as the export script writes it.
      if (bytes == 0 || bytes % (executorch_policy_demo::kFramePeriod * sizeof(float)) != 0) {
        throw std::invalid_argument(expected_file + " is not an expected outputs file");
      }
      expected_.resize(bytes / sizeof(float));
      file.seekg(0);
      file.read(reinterpret_cast<char *>(expected_.data()), expected_.size() * sizeof(float));
      // An infinite reference would make the tolerance infinite too.
      if (!file || !std::all_of(
          expected_.begin(), expected_.end(), [](float value) {return std::isfinite(value);}))
      {
        throw std::invalid_argument(expected_file + " is not an expected outputs file");
      }
    }

    if (format == "float_chw") {
      format_ = FrameFormat::kFloatChw;
      shape_ = {1, channels, height, width};
      dtype_ = ScalarType::Float;
    } else if (format == "byte_hwc") {
      format_ = FrameFormat::kByteHwc;
      shape_ = {height, width, channels};
      dtype_ = ScalarType::Byte;
    } else {
      throw std::invalid_argument("format must be float_chw or byte_hwc, got " + format);
    }
    if (transport != "cuda" && transport != "cpu") {
      throw std::invalid_argument("transport must be cuda or cpu, got " + transport);
    }
    transport_ = transport;
    device_ = transport == "cuda" ? Device(DeviceType::CUDA) : Device(DeviceType::CPU);
    plane_ = static_cast<size_t>(width) * static_cast<size_t>(height);
    count_ = plane_ * static_cast<size_t>(channels);

    image_publisher_ = create_publisher<TensorMsg>("image", rclcpp::QoS(1));
    state_publisher_ = create_publisher<TensorMsg>("state", rclcpp::QoS(1));
    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.acceptable_buffer_backends = "any";
    action_subscription_ = create_subscription<TensorMsg>(
      "action", rclcpp::QoS(1),
      [this](TensorMsg::ConstSharedPtr action) {receive_action(*action);},
      subscription_options);

    started_at_ = Clock::now();
    start_timer_ = create_wall_timer(std::chrono::milliseconds(200), [this]() {
          const bool image = image_publisher_->get_subscription_count() > 0;
          const bool state = state_count_ == 0 || state_publisher_->get_subscription_count() > 0;
          const bool action = action_subscription_->get_publisher_count() > 0;
          if (!image || !state || !action) {
            // Loading a large program takes seconds, not minutes.
            if (Clock::now() - started_at_ > std::chrono::seconds(60)) {
              RCLCPP_ERROR(get_logger(), "camera: no policy connected in 60 s, stopping");
              rclcpp::shutdown();
              return;
            }
            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 5000, "camera: waiting for%s%s%s",
              image ? "" : " an image subscriber", state ? "" : " a state subscriber",
              action ? "" : " an action publisher");
            return;
          }
          start_timer_->cancel();
          RCLCPP_INFO(
            get_logger(), "camera: policy connected, transport=%s, frame %zu elements",
            transport_.c_str(), count_);
          publish_frame();
          // A frame is not sent again: messages carry no frame number, so a late
          // action would then be paired with the wrong frame.
          stall_timer_ = create_wall_timer(std::chrono::seconds(2), [this]() {
            const auto waited = Clock::now() - sent_at_;
            if (waited > std::chrono::seconds(10)) {
              RCLCPP_ERROR(get_logger(), "camera: no action for frame %u, stopping", index_);
              rclcpp::shutdown();
            } else if (waited > std::chrono::seconds(2)) {
              RCLCPP_WARN(get_logger(), "camera: no action for frame %u yet", index_);
            }
            });
        });

    // Last, so a constructor that throws earlier leaks no GPU memory.
    if (transport_ == "cpu") {
      // A GPU camera with plain CPU messages renders on the GPU and then copies
      // each frame to host memory for the middleware.
      check_cuda(cudaMalloc(&scratch_, count_ * executorch::runtime::elementSize(dtype_)));
    }
  }

  ~CameraNode() override
  {
    if (scratch_ != nullptr) {
      cudaFree(scratch_);
    }
  }

  /// True only when every measured frame got its action and every checked action
  /// was right, so a run that was stopped early or never started fails.
  bool passed() const
  {
    return round_trips_ms_.size() >= static_cast<size_t>(frames_) && wrong_actions_ == 0;
  }

private:
  void publish_frame()
  {
    sent_at_ = Clock::now();
    waiting_ = true;
    // Both messages are ready before either is sent, so a failure here cannot leave
    // the policy holding a state whose image never comes.
    std::unique_ptr<TensorMsg> state_msg;
    if (state_count_ > 0) {
      std::vector<float> state(state_count_);
      for (size_t i = 0; i < state_count_; ++i) {
        state[i] = executorch_policy_demo::robot_state(index_, i);
      }
      auto host = make_tensor_ptr(
        std::vector<int32_t>(state_shape_.begin(), state_shape_.end()), std::move(state));
      state_msg = executorch_conversions::allocate_tensor_msg(
        state_shape_, ScalarType::Float, device_);
      executorch_conversions::to_tensor_msg(*state_msg, *host);
    }

    auto msg = executorch_conversions::allocate_tensor_msg(shape_, dtype_, device_);
    if (transport_ == "cuda") {
      auto frame = executorch_conversions::from_output_tensor_msg(*msg);
      check_cuda(executorch_policy_demo::render_frame(
          frame->mutable_data_ptr(), format_, count_, plane_, index_, cudaStreamPerThread));
    } else {
      check_cuda(executorch_policy_demo::render_frame(
          scratch_, format_, count_, plane_, index_, cudaStreamPerThread));
      auto frame = make_tensor_ptr(
        std::vector<int32_t>(shape_.begin(), shape_.end()), scratch_, dtype_,
        Device(DeviceType::CUDA));
      executorch_conversions::to_tensor_msg(*msg, *frame);
    }
    if (state_msg) {
      state_publisher_->publish(std::move(state_msg));
    }
    image_publisher_->publish(std::move(msg));
  }

  void receive_action(const TensorMsg & action)
  {
    // Before the first frame or after the last one, no action is expected.
    if (!waiting_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000, "camera: ignoring an action sent before its frame");
      return;
    }
    bool correct = false;
    try {
      correct = check_action(action);
    } catch (const std::exception & error) {
      // The policy answers each frame once, so an action that cannot be checked counts as wrong.
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000, "camera: cannot check the action: %s", error.what());
    }
    waiting_ = false;
    if (!correct) {
      ++wrong_actions_;
    }
    const auto now = Clock::now();

    if (index_ == warmup_) {
      measured_since_ = sent_at_;
    }
    if (index_ >= warmup_) {
      round_trips_ms_.push_back(
        std::chrono::duration<double, std::milli>(now - sent_at_).count());
    }
    ++index_;

    if (round_trips_ms_.size() >= static_cast<size_t>(frames_)) {
      stall_timer_->cancel();
      report(now, action.data.get_backend_type());
      rclcpp::shutdown();
      return;
    }
    publish_frame();
  }

  // A controller runs on the CPU, so the action is read back to host memory.
  // Throws for an action that cannot be read or compared.
  bool check_action(const TensorMsg & action)
  {
    auto view = executorch_conversions::from_input_tensor_msg(action);
    if (!view) {
      throw std::invalid_argument("the action has no data");
    }
    std::vector<int64_t> shape(view->sizes().begin(), view->sizes().end());
    if (!action_host_ || action_host_->shape != shape ||
      executorch_conversions::scalar_type(*action_host_) != view->scalar_type())
    {
      action_host_ = executorch_conversions::allocate_tensor_msg(
        shape, view->scalar_type(), Device(DeviceType::CPU));
    }
    executorch_conversions::to_tensor_msg(*action_host_, *view);
    if (expected_.empty()) {
      return true;
    }
    const auto count = static_cast<size_t>(view->numel());
    if (view->scalar_type() != ScalarType::Float ||
      expected_.size() != executorch_policy_demo::kFramePeriod * count)
    {
      throw std::invalid_argument("the action does not match the expected outputs file");
    }
    const float * values =
      executorch_conversions::from_input_tensor_msg(*action_host_)->const_data_ptr<float>();
    const float * reference =
      expected_.data() + (index_ % executorch_policy_demo::kFramePeriod) * count;
    bool correct = true;
    for (size_t i = 0; i < count; ++i) {
      const float error = std::abs(values[i] - reference[i]);
      // A NaN error compares false with everything, so once seen it stays the maximum.
      if (!std::isnan(max_error_) && !(error <= max_error_)) {
        max_error_ = error;
      }
      // Above the rounding differences of the backends, below the change in
      // the tiny policy's output from one frame to the next.
      correct = correct && error <= 2e-4f * (1.0f + std::abs(reference[i]));
    }
    return correct;
  }

  void report(Clock::time_point now, const std::string & action_backend)
  {
    auto sorted = round_trips_ms_;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        return sorted[std::min(sorted.size() - 1, static_cast<size_t>(p * sorted.size()))];
      };
    double sum = 0;
    for (const auto value : sorted) {
      sum += value;
    }
    const double seconds = std::chrono::duration<double>(now - measured_since_).count();
    RCLCPP_INFO(
      get_logger(),
      "SUMMARY transport=%s action_backend=%s frames=%zu rtt_p50_ms=%.3f rtt_p90_ms=%.3f "
      "rtt_p99_ms=%.3f rtt_max_ms=%.3f rtt_mean_ms=%.3f loops_per_s=%.1f camera_rss_mb=%.1f "
      "checked=%s wrong_actions=%zu max_error=%.3g",
      transport_.c_str(), action_backend.c_str(), sorted.size(), percentile(0.5),
      percentile(0.9), percentile(0.99), sorted.back(), sum / sorted.size(),
      sorted.size() / seconds, rss_mb(), expected_.empty() ? "no" : "yes", wrong_actions_,
      max_error_);
  }

  rclcpp::Publisher<TensorMsg>::SharedPtr image_publisher_;
  rclcpp::Publisher<TensorMsg>::SharedPtr state_publisher_;
  rclcpp::Subscription<TensorMsg>::SharedPtr action_subscription_;
  Clock::time_point started_at_;
  rclcpp::TimerBase::SharedPtr start_timer_;
  rclcpp::TimerBase::SharedPtr stall_timer_;
  std::unique_ptr<TensorMsg> action_host_;
  std::vector<int64_t> shape_;
  ScalarType dtype_{ScalarType::Float};
  FrameFormat format_{FrameFormat::kFloatChw};
  Device device_{DeviceType::CUDA};
  std::string transport_;
  void * scratch_{nullptr};
  size_t count_{0};
  size_t plane_{0};
  std::vector<int64_t> state_shape_;
  size_t state_count_{0};
  int64_t frames_{0};
  int64_t warmup_{0};
  uint32_t index_{0};
  std::vector<float> expected_;
  size_t wrong_actions_{0};
  float max_error_{0};
  Clock::time_point sent_at_;
  bool waiting_{false};
  Clock::time_point measured_since_;
  std::vector<double> round_trips_ms_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  bool passed = false;
  try {
    auto node = std::make_shared<CameraNode>(rclcpp::NodeOptions());
    rclcpp::spin(node);
    passed = node->passed();
  } catch (const std::exception & error) {
    // Also reached when Ctrl-C ends the context while the node is being built.
    RCLCPP_ERROR(rclcpp::get_logger("camera"), "camera: stopping: %s", error.what());
  }
  rclcpp::shutdown();
  return passed ? 0 : 1;
}
