from distutils.core import setup

setup(name='TagMatcher',
      author='Lukas Kairies',
      install_requires=[
          "numpy",
          "keras"
      ],
      packages=[
          'tagMatcher',
      ])
