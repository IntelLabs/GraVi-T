from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='gravit',
    version='1.1.0',
    description='Graph learning framework for long-term Video undersTanding',
    long_description=long_description,
    license='Apache License 2.0',
    author='Kyle Min',
    author_email='kyle.min@intel.com',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['pyyaml', 'pandas', 'torch', 'torch-geometric>=2.0.3'],
    scripts=['data/generate_spatial-temporal_graphs.py',
             'data/generate_temporal_graphs.py',
             'tools/train_context_reasoning.py',
             'tools/evaluate.py']
)
