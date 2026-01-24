from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension

setup(
    name='rmsnorm',
    ext_modules=[
        CUDAExtension(
            name='rmsnorm',
            sources=['Inference/rmsnorm.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)