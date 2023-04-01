from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='graphltvu',
    version='1.0.0',
    description='A graph learning framework for long-term video understanding',
    long_description=long_description,
    license='Apache License 2.0',
    author='Kyle Min',
    author_email='kyle.min@intel.com',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['pyyaml', 'pandas', 'torch', 'torch-geometric>=2.0.3'],
    scripts=['data/generate_graph.py',
             'tools/train_context_reasoning.py',
             'tools/evaluate.py']
)
