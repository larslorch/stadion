from setuptools import setup, find_packages
setup(
    name='stadion',
    version='1.0.0',
    description='Causal Modeling with Stationary Diffusions',
    author='Lars Lorch',
    author_email='lars.lorch@inf.ethz.ch',
    url="https://github.com/larslorch/stadion",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'jax>=0.4.0',
        'jaxlib>=0.4.0',
        'optax>=0.1.3',
        'tensorflow>=2.9.0',
        'tensorflow_datasets>=4.3.0',
        'numpy>=1.19.0',
    ]
)