from setuptools import setup
import os
from glob import glob

package_name = 'mapless_navigation'

data_files_list = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]

for directory in ['mapless_navigation/models', 'mapless_navigation/worlds', 'mapless_navigation/params']:
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            dest_path = os.path.join('share', package_name, path.replace('mapless_navigation/', '', 1))
            data_files_list.append((dest_path, [os.path.join(path, filename)]))

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=data_files_list,
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
            'evaluate_policy = mapless_navigation.evaluate:main',
            'sabertooth_driver = mapless_navigation.sabertooth_driver:main',
            'bts7960_driver = mapless_navigation.bts7960_driver:main',
            'train_fast = mapless_navigation.train_fast:main',
        ],
    },
)
