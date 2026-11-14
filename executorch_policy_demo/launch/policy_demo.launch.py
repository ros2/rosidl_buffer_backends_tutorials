# Copyright 2026 Meta Platforms, Inc. and affiliates.
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

"""Camera and policy in 2 processes, over CUDA buffers or plain CPU messages."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import (
    EqualsSubstitution, IfElseSubstitution, LaunchConfiguration, PythonExpression)
from launch_ros.actions import Node


def generate_launch_description():
    def arg(name):
        return LaunchConfiguration(name)

    arguments = [
        DeclareLaunchArgument('model_file_path'),
        DeclareLaunchArgument('data_file_paths', default_value="['']"),
        DeclareLaunchArgument('backend_libraries', default_value="['']",
                              description='extra delegate libraries to load at run time'),
        DeclareLaunchArgument('expected_file_path', default_value='',
                              description='expected_outputs.bin, to check every action'),
        DeclareLaunchArgument('transport', default_value='cuda',
                              description='cuda: zero-copy CUDA buffers; cpu: plain messages'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('channels', default_value='3'),
        DeclareLaunchArgument('format', default_value='float_chw'),
        DeclareLaunchArgument('state_shape', default_value='[1, 14]',
                              description='robot state shape; [0] for a program without one'),
        DeclareLaunchArgument('frames', default_value='1000'),
        DeclareLaunchArgument('warmup', default_value='50'),
        DeclareLaunchArgument('report_every', default_value='500'),
    ]

    def on_policy_exit(event, context):
        if event.returncode != 0:
            return [Shutdown(reason='the policy failed')]
        return None

    policy = Node(
        package='executorch_policy_demo',
        executable='policy_node',
        name='policy',
        namespace='policy_demo',
        output='screen',
        parameters=[{
            'model_file_path': arg('model_file_path'),
            'data_file_paths': arg('data_file_paths'),
            'backend_libraries': arg('backend_libraries'),
            # With state_shape:='[0]' the camera sends no state, so the policy takes the image.
            # Compared as a list, so spellings such as '[ 0 ]' count too.
            'input_topics': PythonExpression([
                '"[\'image\']" if ', arg('state_shape'), ' == [0] else "[\'image\', \'state\']"']),
            'output_topics': ['action'],
            'output_transport': IfElseSubstitution(
                EqualsSubstitution(arg('transport'), 'cpu'), 'cpu', 'auto'),
            'report_every': arg('report_every'),
        }],
        on_exit=on_policy_exit,
    )
    camera = Node(
        package='executorch_policy_demo',
        executable='camera_node',
        name='camera',
        namespace='policy_demo',
        output='screen',
        parameters=[{
            'transport': arg('transport'),
            'width': arg('width'),
            'height': arg('height'),
            'channels': arg('channels'),
            'format': arg('format'),
            'state_shape': arg('state_shape'),
            'frames': arg('frames'),
            'warmup': arg('warmup'),
            'expected_file_path': arg('expected_file_path'),
        }],
        on_exit=Shutdown(),
    )

    return LaunchDescription(arguments + [policy, camera])
