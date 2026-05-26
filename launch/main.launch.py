import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Declare the global arguments
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='If true, spawns GT cameras in CARLA, runs evaluation, and shows GT in dashboard.'
    )
    rviz_arg = DeclareLaunchArgument(
        'with_rviz',
        default_value='false',
        description='If true, launches RViz with the dashboard configuration.'
    )

    # 2. Extract configuration values to pass to nodes
    debug_config = LaunchConfiguration('debug')
    with_rviz = LaunchConfiguration('with_rviz')

    # 3. Define the Nodes
    carla_node = Node(
        package='perception_pipeline',
        executable='carla_node',
        name='carla_node',
        output='screen',
        parameters=[{'debug': debug_config}]
    )

    inference_node = Node(
        package='perception_pipeline',
        executable='infrence_node',
        name='inference_node',
        output='screen'
    )

    evaluate_node = Node(
        package='perception_pipeline',
        executable='evaluate_node',
        name='evaluate_node',
        output='screen',
        parameters=[{'debug': debug_config}],
    )

    dashboard_node = Node(
        package='perception_pipeline',
        executable='dashboard_node',
        name='dashboard_node',
        output='screen',
        parameters=[{'debug': debug_config}]
    )

    ekf_node = Node(
        package='perception_pipeline',
        executable='ekf_node',
        name='ekf_node',
        output='screen'
    )

    vo_node = Node(
        package='perception_pipeline',
        executable='visual_odometry_node',
        name='visual_odometry_node',
        output='screen',
        condition=IfCondition(debug_config) 
    )

    # --- RViz Configuration Logic ---
    eval_debug_rviz = os.path.join(
        get_package_share_directory('perception_pipeline'),
        'rviz',
        'eval_debug.rviz'
    )
    eval_inference_rviz = os.path.join(
        get_package_share_directory('perception_pipeline'),
        'rviz',
        'eval_inference.rviz'
    )

    # Keep PythonExpression here to select the correct file path based on debug mode
    rviz_config_file = PythonExpression([
        "'", debug_config, "' == 'true' and '", eval_debug_rviz, "' or '", eval_inference_rviz, "'"
    ])

    # FIXED: Strictly enforce the 'with_rviz' argument. 
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dashboard_rviz',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(with_rviz)
    )

    # 4. Return the LaunchDescription
    return LaunchDescription([
        debug_arg,
        rviz_arg,
        carla_node,
        inference_node,
        evaluate_node,
        dashboard_node,
        ekf_node,
        vo_node,
        rviz_node
    ])