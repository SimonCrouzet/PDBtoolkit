from setuptools import setup, find_packages

setup(
    name="pdbtoolkit",
    version="0.1",
    packages=find_packages(where="src"),  # This tells setuptools to look in src directory
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'biopython',
        'abnumber',
        # add other dependencies
    ],
)
