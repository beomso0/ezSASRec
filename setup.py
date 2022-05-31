from setuptools import setup, find_packages

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="custom_SASRec", 
  version="0.4.3",
  author="Beomso0",
  author_email="univ3352@gmail.com",
  description="customized SASRec",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/beomso0/custom_SASRec",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
  ],
  python_requires='>=3.7',
  install_requires=requirements,
)