from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess

subprocess.run(["pip install numpy scipy ninja pybind11"], shell=True)
proc = subprocess.Popen(["python3 -m pybind11 --includes"], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
out = out.decode('ascii').strip().split()

setup(
    name='spops',
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires='>=3.10',
    install_requires=['numpy', 'scipy', 'ninja', 'pybind11'],
    ext_modules=[CUDAExtension(
        'spops_backend',
        ['./spops/spops_backend.cpp', './spops/lib/sputnik_spops_kernels.cu', './spops/lib/structure_aware_spops_kernels.cu',
         './spops/lib/shuffler_spops_kernels.cu'],
        dlink=True,
        dlink_libraries=["dlink_lib"],
        # https://github.com/zenny-chen/GPU-architectures-docs-and-demos?tab=readme-ov-file#cuda%E7%9B%B8%E5%85%B3%E6%96%87%E6%A1%A3
        extra_compile_args={'cxx': [],
                            'nvcc': [
                                '-arch=sm_80',
                                # https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py#L1050C13-L1050C17
                                # '-dlto',
                                # '-lcusparse',
                                '-lcublas',
                                '-lcudart',
                                # '--ptxas-options=-v',
                                # '-lineinfo',
                                '-O3',
                            ],
                            'nvcclink': ['-arch=sm_80', '--device-link' ]},
        libraries=['cusparse', 'cublas'],
    ),
    Extension(
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
