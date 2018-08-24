from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='jylipy',
      version='0.1',
      description='JyLi''s Python tools',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url='https://jianyangli2017@bitbucket.org/jianyangli2017/jylipy',
      author='Jian-Yang Li',
      author_email='jianyang2li@gmail.com',
      license='MIT',
      packages=find_packages(), #['jylipy','jylipy.constants','jylipy.HST','jylipy.mesh','jylipy.Photometry'],
      install_requires=['numpy','scipy','astropy','matplotlib','pyds9','pysis','ccdproc','spiceypy'],
      zip_safe=False,
      classifiers = [
          'Intended Audience :: Science/Research',
          #"License :: OSI Approved :: BSD License",
          'Operating System :: OS Independent',
          "Programming Language :: Python :: 3",
          'Topic :: Scientific/Engineering :: Astronomy'
      ]
      )
