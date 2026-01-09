from setuptools import setup, find_packages

setup(
    name="grain-analyzer",
    version="0.2.0",
    description="Grain analysis tool for AFM/XQD files",
    author="jkkwoen",
    url="https://github.com/jkkwoen/grain_analyzer",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.20.0",
    ],
    extras_require={
        "stardist": [
            "stardist>=0.8.0",
            "tensorflow>=2.10.0",
        ],
        "cellpose": [
            "cellpose>=3.0.0",
        ],
        "all": [
            "stardist>=0.8.0",
            "tensorflow>=2.10.0",
            "cellpose>=3.0.0",
        ],
    },
    python_requires=">=3.8",
)
