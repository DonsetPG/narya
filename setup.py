from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='narya',
      version='1.0.0',
      description='Deep Reinforcement Learning for Real World football game, and more to come',
      long_description=long_description,
      author='Paul Garnier',
      author_email='paul.garnier.97@gmail.com',
      url='https://github.com/DonsetPG/narya',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'gfootball',
                        'matplotlib',
                        'shapely',
                        'moviepy',
                        'pandas',
                        'pickle'],
      classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
      packages=find_packages())