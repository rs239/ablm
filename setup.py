#!/usr/bin/env python

from setuptools import setup, find_packages
import abmap

setup(
    name="abmap",
    version=abmap.__version__,
    description="AbMAP: Antibody Mutagenesis Augmented Processing",
    author="Chiho Im, Taylor Sorenson, Abhinav Gupta, Rohit Singh",
    author_email="chihoim@mit.edu",
    # url="http://dscript.csail.mit.edu",
    license="MIT",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "abmap = abmap.__main__:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "torch>=1.11",
        "biopython",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
        "h5py",
        "transformers",
        "positional_encodings",
        "dscript"
    ],
)
