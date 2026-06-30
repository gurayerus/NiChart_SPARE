#!/usr/bin/env python3
"""
Setup script for NiChart_SPARE package
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "NiChart_SPARE - SPARE scores calculation from Brain ROI Volumes"

# Read version from __init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'NiChart_SPARE', '__init__.py')
    if os.path.exists(init_path):
        with open(init_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

setup(
    name="NiChart_SPARE",
    version=get_version(),
    author="Kyunglok Baik, Gareth Harman",
    author_email="software@cbica.upenn.edu",
    description="Complete pipelines for SPARE scores training and analysis from Brain ROI Volumes",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/CBICA/NiChart_SPARE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib",
        "seaborn",
        "pyyaml>=5.1",
        "huggingface_hub>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "NiChart_SPARE=NiChart_SPARE.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="neuroimaging, brain, SPARE, machine learning, medical imaging",
    project_urls={
        "Bug Reports": "https://github.com/CBICA/NiChart_SPARE/issues",
        "Source": "https://github.com/CBICA/NiChart_SPARE",
        "Documentation": "https://github.com/CBICA/NiChart_SPARE/wiki",
    },
)
