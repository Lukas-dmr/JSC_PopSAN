#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['ros_fr3_control_service'],
    package_dir={'': 'scripts'},
    scripts=['scripts']
)

setup(**setup_args)
