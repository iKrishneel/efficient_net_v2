#! /usr/bin/env python

from setuptools import setup
from setuptools import find_packages


try:
    with open('README.md', 'r') as f:
        readme = f.read()
except Exception:
    readme = str('')


# write all package dependencies here
install_requires = [
    'coloredlogs',
    'numpy',
    'matplotlib',
    'opencv-python',
    'pillow',
    'torch',
    'torchvision',
    'tqdm',
    'scipy',
    'pytest'
]

setup(
    name='efficient_net_v2',
    version='0.0.0',
    long_description=readme,
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests',
)
