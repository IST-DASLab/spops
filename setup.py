from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension
import subprocess

subprocess.run(["pip install numpy scipy ninja pybind11"], shell=True)
proc = subprocess.Popen(["python3 -m pybind11 --includes"], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
out = out.decode('ascii').strip().split()

setup(
    name='spops',
    packages=find_packages(exclude=['tests', 'tests.*']),
    ext_modules=[Extension(
        'spops_backend_cpu',
        [
            './spops/spops_backend_cpu.cpp',
        ],
        extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++11', '-fPIC', *out, '-march=native', '-fopenmp', '-ffast-math'],
        extra_link_args=['-lgomp']
    )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
