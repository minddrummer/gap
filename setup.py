from distutils.core import setup

setup(
	name='gapkmean',
	version='1.0',
	author='Ke Sang',
	author_email='kesang0156357@gmail.com',
	packages=['gap'],
	url='https://github.com/minddrummer/gap',
	license='LICENSE.txt',
	description='find the best k value of K-mean based on Gap statistics',
	install_requires=[
		'numpy',
		'sklearn',
		'scipy',
	],
)
