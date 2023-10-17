from setuptools import setup

setup(
   name='gestdata',
   version='0.1.0',
   author='Aimar Zabala Vergara',
   author_email='aimar.zabala@ehu.eus',
   packages=['gestdata'],
   url='https://github.com/AimarZ/gestdata/tree/main',
   license='LICENSE.txt',
   description='Este paquete incluye una clase propia y varias funciones para manejar datasets y hacer análisis sobre ellos',
   long_description=open('README.txt').read(),
   tests_require=[],
   install_requires=[
      "seaborn >= 0.13.0",
      "pandas >= 2.1.0",
      "matplotlib >= 3.8.0",
      "numpy >=1.26.0"
   ],
)
