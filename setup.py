from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.system('pip install numpy ninja pybind11')

setup(
    name='spops',
    packages=find_packages(exclude=['tests', 'tests.*']),
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
                                '-lcusparse',
                                '-lcublas',
                                '-lcudart',
                                # '--ptxas-options=-v',
                                # '-lineinfo',
                                '-O3',
                            ],
                            'nvcclink': ['-arch=sm_80', '--device-link' ]},
        libraries=['cusparse', 'cublas'],
    )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
