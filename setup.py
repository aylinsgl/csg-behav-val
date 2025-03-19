from setuptools import find_packages, setup

setup(
    name="csg",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torch-geometric>=2.0.4",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "opencv-python>=4.5.1.48",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "tensorboard>=2.4.0"
    ],
)
