# setup.py
"""Setup script for OpenCount CI Enhanced."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
version = {}
with open("src/opencount_ci/__version__.py") as f:
    exec(f.read(), version)

# Read README
readme = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""

setup(
    name="opencount-ci-enhanced",
    version=version["__version__"],
    description="Class-agnostic object counting with confidence intervals, grouping, and classification",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="OpenCount CI Contributors",
    author_email="",
    url="https://github.com/yourusername/opencount-ci",
    license="MIT",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.7",

    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
    ],

    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
    },

    entry_points={
        "console_scripts": [
            "opencount_ci=opencount_ci.cli:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    keywords="computer-vision object-counting bootstrap confidence-interval clustering classification",

    project_urls={
        "Bug Reports": "https://github.com/yourusername/opencount-ci/issues",
        "Source": "https://github.com/yourusername/opencount-ci",
        "Documentation": "https://github.com/yourusername/opencount-ci/blob/main/README.md",
    },
)