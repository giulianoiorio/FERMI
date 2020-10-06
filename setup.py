from distutils.core import setup,Extension




try:
    from Cython.Build import cythonize
    core_src=['fermi/src/intcore.pyx',]
    core_c_ext=Extension('fermi.src.intcore',sources=core_src,extra_link_args=['-fopenmp',],extra_compile_args=['-fopenmp',])
    ext_modules=cythonize(core_c_ext)
except:
    ext_modules=[]

setup(
            name='fermi',
            version='1.1',
            url='https://github.com/iogiul/FERMI',
            author='Giuliano Iorio',
            author_email='giuliano.iorio@unibo.it',
            package_dir={'fermi/src/':''},
            packages=['fermi','fermi/src'],
            ext_modules=ext_modules
    )
