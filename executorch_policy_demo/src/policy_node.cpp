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

#include <cuda_runtime_api.h>
#include <executorch/extension/cuda/device_guard.h>
#include <executorch/extension/module/module.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "executorch_conversions/executorch_conversions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using TensorMsg = tensor_msgs::msg::ExperimentalTensor;
using Clock = std::chrono::steady_clock;

namespace
{

double rss_mb()
{
  std::ifstream statm("/proc/self/statm");
  int64_t pages = 0;
  int64_t resident = 0;
  statm >> pages >> resident;
  return static_cast<double>(resident * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
}

struct TensorSpec
{
  std::vector<int64_t> shape;
  ScalarType dtype;
  Device device{DeviceType::CPU};
};

}  // namespace

class PolicyNode : public rclcpp::Node
{
public:
  explicit PolicyNode(const rclcpp::NodeOptions & options)
  : Node("policy", options)
  {
    const auto model = declare_parameter<std::string>("model_file_path", "");
    const auto data_files = without_empty(declare_parameter<std::vector<std::string>>(
        "data_file_paths", std::vector<std::string>{}));
    method_ = declare_parameter<std::string>("method_name", "forward");
    const auto input_topics = declare_parameter<std::vector<std::string>>(
      "input_topics", std::vector<std::string>{"image", "state"});
    const auto output_topics = declare_parameter<std::vector<std::string>>(
      "output_topics", std::vector<std::string>{"action"});
    output_transport_ = declare_parameter<std::string>("output_transport", "auto");
    if (output_transport_ != "auto" && output_transport_ != "cpu") {
      throw std::invalid_argument("policy_node: output_transport must be auto or cpu");
    }
    report_every_ = declare_parameter<int>("report_every", 500);
    if (report_every_ <= 0) {
      throw std::invalid_argument("policy_node: report_every must be positive");
    }

    module_ = std::make_unique<Module>(model, data_files);
    const auto meta = module_->method_meta(method_);
    if (!meta.ok()) {
      throw std::runtime_error("policy_node: cannot read method " + method_ + " of " + model);
    }
    if (meta->num_inputs() != input_topics.size() || meta->num_outputs() != output_topics.size()) {
      throw std::runtime_error(
              "policy_node: " + std::to_string(input_topics.size()) + " input and " +
              std::to_string(output_topics.size()) + " output topics, but the program takes " +
              std::to_string(meta->num_inputs()) + " inputs and " +
              std::to_string(meta->num_outputs()) + " outputs");
    }
    // Also a program that copies host inputs to the GPU itself.
    uses_cuda_ = meta->uses_backend("CudaBackend");
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      inputs_.push_back(spec(meta->input_tensor_meta(i), "input " + std::to_string(i)));
      uses_cuda_ = uses_cuda_ || inputs_.back().device.type() == DeviceType::CUDA;
    }
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
      outputs_.push_back(spec(meta->output_tensor_meta(i), "output " + std::to_string(i)));
      uses_cuda_ = uses_cuda_ || outputs_.back().device.type() == DeviceType::CUDA;
    }
    // Also covers the subscriptions and publishers, since cuda_buffer puts its pool
    // on the GPU that is current when the first one is created.
    const auto guard = on_cuda0();
    if (module_->load_method(method_) != Error::Ok) {
      throw std::runtime_error("policy_node: cannot load method " + method_ + " of " + model);
    }

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.acceptable_buffer_backends = "any";
    latest_.resize(input_topics.size());
    for (size_t i = 0; i < input_topics.size(); ++i) {
      subscriptions_.push_back(create_subscription<TensorMsg>(
          input_topics[i], rclcpp::QoS(1),
          [this, i](TensorMsg::ConstSharedPtr msg) {receive(i, std::move(msg));},
          subscription_options));
    }
    for (const auto & topic : output_topics) {
      publishers_.push_back(create_publisher<TensorMsg>(topic, rclcpp::QoS(1)));
    }
    // The resolved names, so a wrong topic or namespace shows before the node waits on it.
    std::string topics;
    for (const auto & subscription : subscriptions_) {
      topics += (topics.empty() ? "" : ", ") + std::string(subscription->get_topic_name());
    }
    RCLCPP_INFO(
      get_logger(), "policy: loaded %s (%zu inputs on %s, %zu outputs, output_transport=%s)",
      model.c_str(), inputs_.size(), topics.c_str(), outputs_.size(), output_transport_.c_str());
  }

private:
  // Launch files cannot pass an empty list, so they pass [''] instead.
  static std::vector<std::string> without_empty(std::vector<std::string> values)
  {
    values.erase(std::remove(values.begin(), values.end(), std::string()), values.end());
    return values;
  }

  // The node hands message storage to the program, so it cannot run a program
  // that expects planned memory or another layout.
  static TensorSpec spec(
    const executorch::runtime::Result<executorch::runtime::TensorInfo> & meta,
    const std::string & name)
  {
    if (!meta.ok()) {
      throw std::runtime_error("policy_node: " + name + " is not a tensor");
    }
    const auto & info = meta.get();
    // An empty message has no storage to view.
    if (info.nbytes() == 0) {
      throw std::runtime_error("policy_node: " + name + " has no elements");
    }
    // cuda_buffer keeps one pool per process, so the node uses one GPU.
    if (info.device().type() == DeviceType::CUDA && info.device().index() > 0) {
      throw std::runtime_error(
              "policy_node: " + name + " is on cuda:" + std::to_string(info.device().index()) +
              ", and the node runs programs on cuda:0");
    }
    if (info.is_memory_planned()) {
      throw std::runtime_error(
              "policy_node: " + name + " is memory planned; export it with "
              "MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False)");
    }
    const auto dim_order = info.dim_order();
    for (size_t i = 0; i < dim_order.size(); ++i) {
      if (dim_order[i] != i) {
        throw std::runtime_error(
                "policy_node: " + name + " is not in the contiguous dimension order");
      }
    }
    TensorSpec result;
    result.shape.assign(info.sizes().begin(), info.sizes().end());
    result.dtype = info.scalar_type();
    result.device = info.device();
    return result;
  }

  static std::string describe(const std::vector<int64_t> & shape, const std::string & dtype)
  {
    std::string result = dtype + " [";
    for (size_t i = 0; i < shape.size(); ++i) {
      result += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return result + "]";
  }

  bool matches(size_t i, const TensorMsg & msg)
  {
    const auto & spec = inputs_[i];
    std::string dtype = "unsupported";
    bool same_dtype = false;
    try {
      const auto type = executorch_conversions::scalar_type(msg);
      dtype = executorch::runtime::toString(type);
      same_dtype = type == spec.dtype;
    } catch (const std::exception &) {
      // Reported below as an unsupported dtype.
    }
    if (same_dtype && msg.shape == spec.shape) {
      return true;
    }
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "policy: skipping a message on input %zu: got %s, the program takes %s", i,
      describe(msg.shape, dtype).c_str(),
      describe(spec.shape, executorch::runtime::toString(spec.dtype)).c_str());
    return false;
  }

  // A host message for a GPU program, or a GPU message for a CPU program, is copied
  // into a new message on the program's device, since a cuda_buffer message can be
  // written only once.
  TensorPtr input_view(
    size_t i, const TensorMsg & msg, std::vector<std::unique_ptr<TensorMsg>> & staging)
  {
    const auto & spec = inputs_[i];
    // The conversion checks the message's size, offset and strides against its storage.
    auto view = executorch_conversions::from_input_tensor_msg(msg);
    if (!view) {
      throw std::invalid_argument("input " + std::to_string(i) + " has no data");
    }
    if (view->device().type() == DeviceType::CUDA && spec.device.type() == DeviceType::CUDA &&
      view->device() != spec.device)
    {
      throw std::invalid_argument(
              "input " + std::to_string(i) + " is on another GPU than the program expects");
    }
    if (view->device() == spec.device) {
      return view;
    }
    ++staged_copies_;
    staging.push_back(
      executorch_conversions::allocate_tensor_msg(spec.shape, spec.dtype, spec.device));
    executorch_conversions::to_tensor_msg(*staging.back(), *view);
    return executorch_conversions::from_input_tensor_msg(*staging.back());
  }

  // A message with the wrong dtype or shape is dropped on arrival, so it does not
  // discard the messages held for the other inputs. Once every input has a message,
  // the set runs and is released, also when a message cannot be viewed. A held
  // message keeps the publisher's buffer, and the publisher waits for its buffers
  // when it exits.
  void receive(size_t i, TensorMsg::ConstSharedPtr msg)
  {
    const auto started = Clock::now();
    if (!matches(i, *msg)) {
      return;
    }
    latest_[i] = std::move(msg);
    if (std::any_of(latest_.begin(), latest_.end(), [](const auto & held) {return !held;})) {
      return;
    }
    try {
      step(started);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000, "policy: skipping a run: %s", error.what());
    }
    std::fill(latest_.begin(), latest_.end(), nullptr);
  }

  // ExecuTorch's CUDA backend loads and runs on the current GPU, which another
  // component in the same process can change.
  std::optional<executorch::extension::cuda::CUDAGuard> on_cuda0() const
  {
    if (!uses_cuda_) {
      return std::nullopt;
    }
    auto guard = executorch::extension::cuda::CUDAGuard::create(0);
    if (!guard.ok()) {
      throw std::runtime_error("policy_node: cannot select cuda:0");
    }
    return std::move(guard.get());
  }

  void step(Clock::time_point started)
  {
    std::vector<std::unique_ptr<TensorMsg>> replies;
    {
      const auto guard = on_cuda0();
      std::vector<std::unique_ptr<TensorMsg>> staging;
      std::vector<TensorPtr> views;
      std::vector<EValue> inputs;
      for (size_t i = 0; i < inputs_.size(); ++i) {
        views.push_back(input_view(i, *latest_[i], staging));
        inputs.emplace_back(*views.back());
      }
      std::vector<TensorPtr> output_views;
      std::vector<bool> staged_outputs;
      for (size_t i = 0; i < outputs_.size(); ++i) {
        const auto & spec = outputs_[i];
        const Device msg_device = output_transport_ == "cpu" ?
          Device(DeviceType::CPU) : spec.device;
        replies.push_back(
          executorch_conversions::allocate_tensor_msg(spec.shape, spec.dtype, msg_device));
        staged_outputs.push_back(msg_device != spec.device);
        if (!staged_outputs.back()) {
          output_views.push_back(executorch_conversions::from_output_tensor_msg(*replies.back()));
        } else {
          staging.push_back(
            executorch_conversions::allocate_tensor_msg(spec.shape, spec.dtype, spec.device));
          output_views.push_back(executorch_conversions::from_output_tensor_msg(*staging.back()));
        }
        if (module_->set_output(method_, *output_views.back(), i) != Error::Ok) {
          throw std::runtime_error(
                  "policy_node: set_output failed for output " + std::to_string(i));
        }
      }
      const auto result = module_->execute(method_, inputs);
      if (!result.ok()) {
        throw std::runtime_error(
                "policy_node: execute failed with error " +
                std::to_string(static_cast<int>(result.error())));
      }
      for (size_t i = 0; i < outputs_.size(); ++i) {
        const auto & output = result->at(i).toTensor();
        // An output that is another output or an input is not written into the
        // storage given to set_output.
        if (output.const_data_ptr() != output_views[i]->const_data_ptr()) {
          throw std::runtime_error(
                  "policy_node: output " + std::to_string(i) + " is not written to its reply");
        }
        if (staged_outputs[i]) {
          ++staged_copies_;
          executorch_conversions::to_tensor_msg(*replies[i], output);
        }
        // A bounded dynamic output can come back smaller than its largest shape.
        replies[i]->shape.assign(output.sizes().begin(), output.sizes().end());
        replies[i]->strides.assign(output.strides().begin(), output.strides().end());
      }
    }  // Release the views, which records the write events, before publishing.
    for (size_t i = 0; i < replies.size(); ++i) {
      publishers_[i]->publish(std::move(replies[i]));
    }
    durations_ms_.push_back(
      std::chrono::duration<double, std::milli>(Clock::now() - started).count());
    if (durations_ms_.size() == static_cast<size_t>(report_every_)) {
      report(latest_[0]->data.get_backend_type());
    }
  }

  void report(const std::string & input0_backend)
  {
    auto sorted = durations_ms_;
    std::sort(sorted.begin(), sorted.end());
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (uses_cuda_) {
      const auto guard = on_cuda0();
      if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        free_bytes = total_bytes = 0;
      }
    }
    RCLCPP_INFO(
      get_logger(),
      "POLICY runs=%zu callback_p50_ms=%.3f callback_p99_ms=%.3f staged_copies=%zu "
      "rss_mb=%.1f gpu_total_used_mb=%.1f input0_backend=%s",
      sorted.size(), sorted[sorted.size() / 2],
      sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)], staged_copies_, rss_mb(),
      static_cast<double>(total_bytes - free_bytes) / (1024.0 * 1024.0),
      input0_backend.c_str());
    durations_ms_.clear();
    staged_copies_ = 0;
  }

  std::unique_ptr<Module> module_;
  std::string method_;
  std::string output_transport_;
  int64_t report_every_{0};
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  bool uses_cuda_{false};
  std::vector<TensorMsg::ConstSharedPtr> latest_;
  std::vector<rclcpp::Subscription<TensorMsg>::SharedPtr> subscriptions_;
  std::vector<rclcpp::Publisher<TensorMsg>::SharedPtr> publishers_;
  std::vector<double> durations_ms_;
  size_t staged_copies_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(PolicyNode)
