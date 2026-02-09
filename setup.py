from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '7.5;8.0;8.6;8.9')

setup(
    name='cuda_extensions',
    ext_modules=[
        # RMSNorm extension
        CUDAExtension(
            name='rmsnorm',
            sources=['Tools/rmsnorm/rmsnorm.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
            }
        ),
        
        # SwiGLU extension
        CUDAExtension(
            name='swiglu_fused',
            sources=[
                'Tools/swiglu/swiglu_binding.cpp',
                'Tools/swiglu/swiglu.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-Xptxas=-v',
                ]
            },
            include_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tools/swiglu'),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
    ],
)