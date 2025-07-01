from setuptools import setup, find_packages

setup(
    name="tensorprime",
    version="0.1.0",
    description="A custom tensor library with autograd functionality",
    author="Kayode Oke",
    author_email="<kayodeoke417@gmail.com>",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.1",
        "torch>=2.2.2",
    ],
    extras_require={
        "dev": ["pytest>=6.2.5"],
    },
    python_requires=">=3.8",
)
