from setuptools import find_packages, setup

package_name = 'perception_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'carla_node = perception_pipeline.carla_node:main',
            'dashboard_node = perception_pipeline.dashboard_node:main',
            'engine_builder_node = perception_pipeline.engine_builder_node:main',
            'evaluate_node = perception_pipeline.evaluate_node:main',
            'infrence_node = perception_pipeline.infrence_node:main',
        ],
    },
)
