import setuptools
from pathlib import Path

p = Path(__file__)

setup_requires = [
    'numpy',
    'pytest-runner'
]

install_requires = [
]
test_require = [
    'pytest-cov',
    'pytest-html',
    'pytest'
]

setuptools.setup(
    name="thdbonas",
    version='0.1.0',
    python_requires='>3.5',
    author="Koji Ono",
    author_email="kbu94982@gmail.com",
    description="Pytorch Version: Deep Bayes Optimization for Neural Network Architecture Search (thDBONAS)",
    url='https://github.com/0h-n0/thdbonas',
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=test_require,
    extras_require={
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
