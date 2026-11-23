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

#include <gtest/gtest.h>

#include <memory>

#include "input_set.hpp"

using executorch_policy_demo::InputSet;
using Message = std::shared_ptr<int>;

// A use count of 1 means only the test holds the message, so its buffer would be free.
TEST(InputSet, RejectedMessageIsNotHeld)
{
  InputSet<Message> set(2);
  auto rejected = std::make_shared<int>(0);
  EXPECT_FALSE(set.add(0, rejected, false));
  EXPECT_EQ(rejected.use_count(), 1);
}

TEST(InputSet, RejectedFirstDropsTheMessageThatFollows)
{
  InputSet<Message> set(2);
  auto good = std::make_shared<int>(1);
  EXPECT_FALSE(set.add(1, std::make_shared<int>(0), false));
  EXPECT_FALSE(set.add(0, good, true));
  EXPECT_EQ(good.use_count(), 1);
}

TEST(InputSet, RejectedLastDropsTheMessageBeforeIt)
{
  InputSet<Message> set(2);
  auto good = std::make_shared<int>(1);
  EXPECT_FALSE(set.add(0, good, true));
  EXPECT_FALSE(set.add(1, std::make_shared<int>(0), false));
  EXPECT_EQ(good.use_count(), 1);
}

TEST(InputSet, CompleteSetRunsAndTheNextSetStartsEmpty)
{
  InputSet<Message> set(2);
  EXPECT_FALSE(set.add(1, std::make_shared<int>(0), false));
  EXPECT_FALSE(set.add(0, std::make_shared<int>(1), true));
  EXPECT_FALSE(set.add(0, std::make_shared<int>(2), true));
  EXPECT_TRUE(set.add(1, std::make_shared<int>(3), true));
  EXPECT_EQ(*set[0], 2);
  EXPECT_EQ(*set[1], 3);
  set.clear();
  EXPECT_FALSE(set[0]);
  EXPECT_FALSE(set.add(0, std::make_shared<int>(4), true));
}

TEST(InputSet, NewerMessageReleasesTheOlderOne)
{
  InputSet<Message> set(2);
  auto older = std::make_shared<int>(0);
  EXPECT_FALSE(set.add(0, older, true));
  EXPECT_FALSE(set.add(0, std::make_shared<int>(1), true));
  EXPECT_EQ(older.use_count(), 1);
}

TEST(InputSet, RejectedMessageReleasesTheOlderOne)
{
  InputSet<Message> set(2);
  auto older = std::make_shared<int>(0);
  EXPECT_FALSE(set.add(0, older, true));
  EXPECT_FALSE(set.add(0, std::make_shared<int>(1), false));
  EXPECT_EQ(older.use_count(), 1);
}

TEST(InputSet, NewerMessageReplacesARejectedOne)
{
  InputSet<Message> set(2);
  EXPECT_FALSE(set.add(0, std::make_shared<int>(0), false));
  EXPECT_FALSE(set.add(0, std::make_shared<int>(1), true));
  EXPECT_TRUE(set.add(1, std::make_shared<int>(2), true));
}

// A program with one input, such as frame statistics, runs on every message that fits.
TEST(InputSet, OneInputRunsOnEachMessageThatFits)
{
  InputSet<Message> set(1);
  EXPECT_TRUE(set.add(0, std::make_shared<int>(0), true));
  set.clear();
  EXPECT_FALSE(set.add(0, std::make_shared<int>(1), false));
  EXPECT_TRUE(set.add(0, std::make_shared<int>(2), true));
  EXPECT_EQ(*set[0], 2);
}
