import pathlib
from setuptools import setup

from popdyn import __version__


HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(
    name='popdyn',
    version=__version__,
    description='Simulation of population dynamics',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/popdynio/popdyn',
    author='Popdynio',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
    packages=['popdyn'],
    include_package_data=True,
    install_requires=['scipy'],
)
