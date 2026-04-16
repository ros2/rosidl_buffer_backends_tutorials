# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('width', default_value='1920'),
        DeclareLaunchArgument('height', default_value='1080'),
        DeclareLaunchArgument('use_cuda', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),

        Node(
            package='robot_arm_demo',
            executable='renderer_node',
            parameters=[{
                'image_width': LaunchConfiguration('width'),
                'image_height': LaunchConfiguration('height'),
                'use_cuda': LaunchConfiguration('use_cuda'),
            }],
        ),

        Node(
            package='robot_arm_demo',
            executable='display_node',
            parameters=[{
                'use_cuda': LaunchConfiguration('use_cuda'),
                'headless': LaunchConfiguration('headless'),
            }],
        ),
    ])
