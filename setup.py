from setuptools import setup

if __name__ == '__main__':

    setup(name='jylipy',
          version='0.1',
          description='JyLi''s Python tools',
          url='https://jianyangli2017@bitbucket.org/jianyangli2017/jylipy_2to3.git',
          author='Jian-Yang Li',
          author_email='jianyang2li@gmail.com',
          license='MIT',
          packages=['jylipy','jylipy.constants','jylipy.hst','jylipy.mesh','jylipy.photometry'],
          requires=['numpy','scipy','astropy','matplotlib','pyds9','pysis','ccdproc'],
          zip_safe=False,
          classifiers = [
              'Intended Audience :: Science/Research',
              #"License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 3",
              'Topic :: Scientific/Engineering :: Astronomy'
          ]
          )
