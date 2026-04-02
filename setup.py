"""Setup script for AML Transaction Monitoring project."""

from setuptools import setup, find_packages

setup(
    name="aml-monitoring",
    version="1.0.0",
    description="Anti-Money Laundering Transaction Monitoring with Anomaly Detection",
    author="Mezemir Neway Guchale",
    author_email="gumezemir@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
    ],
)
