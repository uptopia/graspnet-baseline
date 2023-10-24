#!/usr/bin/env python3

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "knn_pytorch.knn_pytorch",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="knn_pytorch",
    version="0.1",
    author="foolyc",
    url="https://github.com/foolyc/torchKNN",
    description="KNN implement in Pytorch 1.0 including both cpu version and gpu version",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
# import glob
# import os
# ROOT = os.path.dirname(os.path.abspath(__file__))

# _ext_src_root = ROOT
# _ext_sources = glob.glob("{}/src/cpu/*.cpp".format(_ext_src_root)) + glob.glob("{}/src/cuda/*.cu".format(_ext_src_root))
# _ext_headers = glob.glob("{}/src/cpu/*.h".format(_ext_src_root)) + glob.glob("{}/src/cuda/*.h".format(_ext_src_root))

# setup(
#     name='knn_pytorch',
#     ext_modules=[
#         CppExtension(
#             name='knn_pytorch.knn_pytorch',
#             sources=_ext_sources,
#             extra_compile_args={
#                 "cxx": ["-O2", "-I{}".format(glob.glob("{}/src/cpu/*.h".format(_ext_src_root)))],
#                 "nvcc": ["-O2", "-I{}".format(glob.glob("{}/src/cuda/*.h".format(_ext_src_root)))],
#             },
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

# import glob
# import os

# import torch
# from setuptools import find_packages
# from setuptools import setup
# from torch.utils.cpp_extension import CUDA_HOME
# from torch.utils.cpp_extension import CppExtension
# from torch.utils.cpp_extension import CUDAExtension

# requirements = ["torch", "torchvision"]


# def get_extensions():
#     # this_dir = os.path.dirname(os.path.abspath(__file__))
#     ROOT = os.path.dirname(os.path.abspath(__file__))
#     # extensions_dir = os.path.join(this_dir, "src")
#     extensions_dir = ROOT#"graspnet-baseline/knn/src/"
#     # main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
#     main_file = glob.glob("{}/src/*.cpp".format(extensions_dir))
#     source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
#     source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

#     sources = main_file + source_cpu

#     print("=================================")
#     print("=================================")
#     print(ROOT)
#     print(main_file)
#     print("=================================")
#     print("=================================")

#     extension = CppExtension

#     extra_compile_args = {
#         "cxx": []
#         }
#     define_macros = []

#     if torch.cuda.is_available() and CUDA_HOME is not None:
#         extension = CUDAExtension
#         sources += source_cuda
#         define_macros += [("WITH_CUDA", None)]
#         extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]

#     sources = [os.path.join(extensions_dir, s) for s in sources]

#     include_dirs = [extensions_dir]

#     print("=================================")
#     print("=================================")
#     print("sources:", sources)
#     print("extensions_dir:",extensions_dir)
#     print("=================================")
#     print("=================================")

#     ext_modules = [
#         extension(
#             name="knn_pytorch.knn_pytorch",
#             sources=sources,
#             include_dirs=include_dirs,
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#     ]

#     return ext_modules


# setup(
#     name="knn_pytorch",
#     version="0.1",
#     author="foolyc",
#     url="https://github.com/foolyc/torchKNN",
#     description="KNN implement in Pytorch 1.0 including both cpu version and gpu version",
#     ext_modules=get_extensions(),
#     cmdclass={
#         "build_ext": torch.utils.cpp_extension.BuildExtension
#     }
# )
