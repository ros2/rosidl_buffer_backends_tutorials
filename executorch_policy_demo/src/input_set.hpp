// Copyright 2026 Open Source Robotics Foundation, Inc.
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

#ifndef INPUT_SET_HPP_
#define INPUT_SET_HPP_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace executorch_policy_demo
{

/// The latest message on each input, gathered into sets for the program.
/// A held message keeps the publisher's buffer, and the publisher waits for its
/// buffers when it exits. So a message the program cannot take counts as its
/// input's arrival, and the set it belongs to is dropped once every input has
/// arrived, in whatever order, instead of holding the others for a set that cannot
/// complete.
template<typename Message>
class InputSet
{
public:
  explicit InputSet(size_t inputs)
  : held_(inputs), rejected_(inputs) {}

  /// Returns true when every input holds a message that fits, so the set can run.
  /// Call clear() after the run. A complete set with a rejected message is cleared here.
  bool add(size_t input, Message message, bool fits)
  {
    rejected_[input] = !fits;
    held_[input] = fits ? std::move(message) : Message{};
    for (size_t i = 0; i < held_.size(); ++i) {
      if (!held_[i] && !rejected_[i]) {
        return false;
      }
    }
    if (std::find(rejected_.begin(), rejected_.end(), true) != rejected_.end()) {
      clear();
      return false;
    }
    return true;
  }

  const Message & operator[](size_t input) const {return held_[input];}

  void clear()
  {
    std::fill(held_.begin(), held_.end(), Message{});
    std::fill(rejected_.begin(), rejected_.end(), false);
  }

private:
  std::vector<Message> held_;
  std::vector<bool> rejected_;
};

}  // namespace executorch_policy_demo

#endif  // INPUT_SET_HPP_
