import os
import io
from setuptools import find_packages, setup


src_dir = os.path.abspath(os.path.dirname(__file__))


# Load README
def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name='chemproppred',
    version='v0.1.0',
    author='Gabe Bradford, Jurgis Ruza',
    description='Chemistry-Informed Machine Learning for Polymer Electrolyte Discovery',
    url='https://github.com/learningmatter-mit/Chem-prop-pred',
    license='MIT',
    long_description=read("README.md"),
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'torch',
    ],
    python_requires='>=3.7',
)
