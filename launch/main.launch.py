import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Declare the 'debug' global argument (Default is False)
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='If true, spawns GT cameras in CARLA, runs evaluation, and shows GT in dashboard.'
    )

    # 2. Extract the configuration value to pass to nodes
    debug_config = LaunchConfiguration('debug')

    # 3. Define the Nodes
    
    # Carla Node: Always runs. (You'll update it to check 'debug' and spawn 5 or 15 cameras)
    carla_node = Node(
        package='perception_pipeline',
        executable='carla_node',
        name='carla_node',
        output='screen',
        parameters=[{'debug': debug_config}]
    )

    # Inference Node: Always runs. Consumes RGB and outputs Pred Depth/Seg.
    inference_node = Node(
        package='perception_pipeline',
        executable='inference_node',
        name='inference_node',
        output='screen'
        # Doesn't strictly need debug unless you want to silence its prints
    )

    # Evaluate Node: Always launch, but its behavior changes with debug mode
    evaluate_node = Node(
        package='perception_pipeline',
        executable='evaluate_node',
        name='evaluate_node',
        output='screen',
        parameters=[{'debug': debug_config}],
    )

    # Dashboard Node: Always runs, but adapts RViz outputs based on debug
    dashboard_node = Node(
        package='perception_pipeline',
        executable='dashboard_node',
        name='dashboard_node',
        output='screen',
        parameters=[{'debug': debug_config}]
    )

    # Note: EngineBuilderNode is usually run once offline to build the cache, 
    # so we don't put it in the real-time launch file.

    # 4. Return the LaunchDescription
    return LaunchDescription([
        debug_arg,
        carla_node,
        inference_node,
        evaluate_node,
        dashboard_node
    ])