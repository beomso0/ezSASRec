import setuptools

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="ezSASRec", 
    version="0.8.9",
    author="Beomso0",
    license='MIT',
    author_email="univ3352@gmail.com",
    description="easy SASRec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beomso0/ezSASRec",
    packages=setuptools.find_packages(),
    keywords='sasrec, recommendation, sequential',
    classifiers=[
      "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
)