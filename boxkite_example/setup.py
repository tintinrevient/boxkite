from setuptools import setup, find_packages

setup(
    name='boxkite_example',
    version='1.0.0',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['mlflow', 'pandas', 'numpy', 'scikit-learn', 'pipdeptree', 'graphviz==0.18', 'flask', 'boxkite', 'psycopg2', 'build', 'twine', 'boxkite', 'pytest']
)