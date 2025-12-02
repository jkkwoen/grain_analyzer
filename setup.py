from setuptools import setup, find_packages

setup(
    name="grain-analyzer-standalone",
    version="0.1.0",
    description="Standalone grain analysis tool for XQD files",
    author="jkkwoen",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.20.0",
    ],
    python_requires=">=3.8",
)

