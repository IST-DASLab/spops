import os
import re

import pybind11
import torch
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

pybind_includes = [f"-I{pybind11.get_include()}", f"-I{pybind11.get_include(user=True)}"]


def _cuda_is_available() -> bool:
    """Return True if we can build the CUDA extension.

    Build-time CUDA availability is independent from runtime ``torch.cuda``;
    we need ``nvcc`` (i.e. ``CUDA_HOME``) more than a live GPU. This lets us
    build CPU-only wheels on CI runners without a CUDA toolkit while still
    producing GPU-enabled wheels on developer / build boxes.
    """
    if os.environ.get("SPOPS_FORCE_CPU_ONLY", "0") == "1":
        return False
    return CUDA_HOME is not None and torch.backends.cuda.is_built()


def _cuda_gencode_flags():
    """Build -gencode flags from TORCH_CUDA_ARCH_LIST.

    Default covers A100 (sm_80), L40/L40S/RTX 6000 Ada (sm_89), and H100 (sm_90),
    with PTX embedded for the highest arch for forward compatibility.
    """
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0+PTX")
    flags = []
    for entry in re.split(r"[;\s,]+", arch_list.strip()):
        if not entry:
            continue
        ptx = entry.endswith("+PTX")
        ver = entry[:-4] if ptx else entry
        num = ver.replace(".", "")
        flags.append(f"-gencode=arch=compute_{num},code=sm_{num}")
        if ptx:
            flags.append(f"-gencode=arch=compute_{num},code=compute_{num}")
    return flags


_cpu_extension = Extension(
    "spops_backend_cpu",
    ["./spops/spops_backend_cpu.cpp"],
    extra_compile_args=[
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++11",
        "-fPIC",
        *pybind_includes,
        "-march=native",
        "-fopenmp",
        "-ffast-math",
    ],
    extra_link_args=["-lgomp"],
)

ext_modules = [_cpu_extension]

if _cuda_is_available():
    _gencode = _cuda_gencode_flags()
    ext_modules.insert(
        0,
        CUDAExtension(
            "spops_backend",
            [
                "./spops/spops_backend.cpp",
                "./spops/lib/sputnik_spops_kernels.cu",
                "./spops/lib/structure_aware_spops_kernels.cu",
                "./spops/lib/shuffler_spops_kernels.cu",
            ],
            dlink=True,
            dlink_libraries=["dlink_lib"],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    *_gencode,
                    "-lcublas",
                    "-lcudart",
                    "-O3",
                ],
                "nvcclink": [*_gencode, "--device-link"],
            },
            libraries=["cusparse", "cublas"],
        ),
    )
else:
    print(
        "spops: CUDA toolkit not detected (CUDA_HOME unset or torch built without CUDA). "
        "Building CPU-only wheel; GPU kernels will be unavailable at runtime."
    )

setup(
    name="spops",
    packages=find_packages(exclude=["tests", "tests.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
