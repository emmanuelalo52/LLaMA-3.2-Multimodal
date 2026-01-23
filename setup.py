from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension

setup(
    name='rmsnorm',
    ext_modules=[
        CUDAExtension('rmsnorm',['Inference/rmsnorm.cu',]),
    ],
    cmdclass={'build_ext':BuildExtension}
)