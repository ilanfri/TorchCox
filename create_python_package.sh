#!/bin/bash

#The below follows the official Python structure for creating a new package:
# https://packaging.python.org/tutorials/packaging-projects/

read -p "New package name: " pkgname

mkdir $pkgname
cd $pkgname

pkgnamelower=$(echo "$pkgname" | awk '{print tolower($0)}')
mkdir $pkgnamelower
cd $pkgnamelower
touch __init__.py
cd ..

mkdir tests
mkdir data
mkdir notebooks

# FILL IN CORRECT VERSION NUMBER, AUTHOR, EMAIL, AND DESCRIPTION BELOW
# Make sure to use 'data' and 'notebook' directories appropriately for each
#  of those two, the below will automatically ignore those when creating package:
# https://gitlab.com/vinktim/presentation-beyond-notebooks/-/blob/master/reference_demo_ds_project/setup.py

echo 'import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='"\"$pkgnamelower\""', 
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(exclude=["data", "notebooks"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)' > setup.py

touch README.md

#MIT License
echo "Copyright (c) 2021 Ilan Fridman Rojas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE." > LICENSE

#Download a good default .gitignore
wget -O .gitignore https://github.com/github/gitignore/raw/master/Python.gitignore

#After running this bash script, move the .py files containing the functionality of the package into the directory named as your package, which is within the main package directory.

# Then run below commands within directory where
# setup.py file is to:
#- create requirements.txt file
#- create .whl files which make package installable

#pipreqs .
#python3 setup.py sdist bdist_wheel

# Then you can install the package by running this from the directory where setup.py is:
#pip3 install -e .
