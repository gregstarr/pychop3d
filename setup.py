from setuptools import setup, find_packages

setup(
    name='pychop3d',
    version='alpha-1',
    description='Python implementation of "Chopper: Partitioning Models into 3D-Printable Parts"',
    url='https://github.com/gregstarr/pychop3d',
    author='gregstarr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['pychop3d'],
    package_dir={'pychop3d': 'pychop3d'},
    python_requires='!=3.6.',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pytest',
        'networkx',
        'pyyaml',
        'requests',
        'trimesh',
        'pyglet'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/gregstarr/pychop3d/issues',
        'Source': 'https://github.com/gregstarr/pychop3d',
    }
)
