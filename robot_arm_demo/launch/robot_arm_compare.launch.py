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
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('window_width', default_value='960'),
        DeclareLaunchArgument('window_height', default_value='540'),
        DeclareLaunchArgument('window_y', default_value='100'),

        Node(
            package='robot_arm_demo',
            executable='renderer_node',
            namespace='cuda',
            parameters=[{
                'image_width': LaunchConfiguration('width'),
                'image_height': LaunchConfiguration('height'),
                'use_cuda': True,
            }],
            arguments=['--log-level', 'WARN'],
        ),

        Node(
            package='robot_arm_demo',
            executable='renderer_node',
            namespace='cpu',
            parameters=[{
                'image_width': LaunchConfiguration('width'),
                'image_height': LaunchConfiguration('height'),
                'use_cuda': False,
            }],
            arguments=['--log-level', 'WARN'],
        ),

        Node(
            package='robot_arm_demo',
            executable='display_node',
            namespace='cuda',
            parameters=[{
                'use_cuda': True,
                'headless': LaunchConfiguration('headless'),
                'borderless': True,
                'window_x': 0,
                'window_y': LaunchConfiguration('window_y'),
                'max_window_width': LaunchConfiguration('window_width'),
                'max_window_height': LaunchConfiguration('window_height'),
            }],
        ),

        Node(
            package='robot_arm_demo',
            executable='display_node',
            namespace='cpu',
            parameters=[{
                'use_cuda': False,
                'headless': LaunchConfiguration('headless'),
                'borderless': True,
                'window_x': LaunchConfiguration('window_width'),
                'window_y': LaunchConfiguration('window_y'),
                'max_window_width': LaunchConfiguration('window_width'),
                'max_window_height': LaunchConfiguration('window_height'),
            }],
        ),
    ])
