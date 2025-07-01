# navbar/setup.py
from setuptools import setup, find_packages

setup(
    name="navbar",
    version="1.0.0",
    description="Streamlit navbar app launcher",
    py_modules=["navbar"],
    entry_points={
        "console_scripts": [
            "navbar=navbar:run_app",
        ],
    },
    install_requires=[
        "streamlit",
    ],
    python_requires=">=3.7",
)
