'''
    setup.py
'''

from setuptools import setup, find_packages

# run the setup
setup(name='eigenstrapping',
      version='0.0.1.8',
      description="For generating surrogate brain maps with spatial autocorrelation using geometric eigenmodes.",
      author='Nikitas C. Koussis, Systems Neuroscience Group Newcastle',
      author_email='nikitas.koussis@gmail.com',
      url='https://github.com/SNG-newy/eigenstrapping',
      packages=find_packages(),
      )
