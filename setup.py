from setuptools import setup, find_packages

setup(
    name="2x2_analysis",
    version="0.1.0",
    author="Rado Razakamiandra",
    author_email="radofana@gmail.com",
    description="Analysis package for 2x2 detector data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fanrado/2x2_analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.2.0",
        "scipy>=1.4.0",
        "h5flow>=1.0.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.10",
)