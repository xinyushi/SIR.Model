from setuptools import setup
import sys, os
import setuptools

this_dir = os.path.dirname(os.path.realpath(__file__))

__version__ = '0.0.0'


setup(
    name='sir',
    version=__version__,
    author='Hsiang Wang, Shenghan Mei, Xinyu Shi',
    description='a basic SIR model package with some variations',
    python_requires='>=3.6',
    packages=['sir'],#,'SIR_continuous_reinfected','R_continuous_mask'],
    zip_safe=True,

)

