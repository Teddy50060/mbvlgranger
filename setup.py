from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # Fallback when requirements.txt isn't available (build isolation)
    requirements = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.3.0",
        "statsmodels>=0.12.0",
        "scikit-learn>=1.0.0",
        "dtaidistance>=2.3.0",
    ]

setup(
    name="mbvlgranger",
    version="0.1.3",
    author="Chakattrai Sookkongwaree",
    author_email="6632033821@student.chula.ac.th",
    description="Multi-Band Variable-Lag Granger Causality: A Unified Framework for Causal Time Series Inference across Frequencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Teddy50060/MBVLgranger",
    project_urls={
        "Bug Tracker": "https://github.com/Teddy50060/MBVLgranger/issues",
        "Documentation": "https://github.com/Teddy50060/MBVLgranger/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "examples": [
            "jupyter>=1.0",
            "seaborn>=0.11",
        ]
    },
    include_package_data=True,
    package_data={
        "mbvlgranger": ["data/*.mat"],
    },
)