#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#    history = history_file.read()

requirements = [
    # TODO: put package requirements here
    # "keras",
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='keras_dqn',
    version='0.0.1',
    description="keras_dqn",
    long_description=readme + '\n\n',  # + history,
    author="ntasfi and edersantana",
    url='https://github.com/ntasfi/keras_dqn',
    packages=[
        'keras_dqn',
    ],
    package_dir={'keras_dqn':
                 'keras_dqn'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='x',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Witches',
        'License :: MIT',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
