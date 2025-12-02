from setuptools import setup
import os
from glob import glob

package_name = 'mapless_navigation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={package_name: 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Mapless DRL Forest Navigation using PPO',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_ppo = mapless_navigation.train_ppo:main',
            'navigation_node = mapless_navigation.navigation_node:main',
        ],
    },
)
