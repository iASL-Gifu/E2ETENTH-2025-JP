from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import numpy

# C 実装モジュール（CPUベース）
lidar_graph_c = Extension(
    name='lidar_graph',
    sources=['src/data/graph/lidar_graph.c'],
    include_dirs=[numpy.get_include()],
)

# CUDA 実装モジュール（GPUベース）
lidar_graph_cuda = CUDAExtension(
    name='lidar_graph_cuda',
    sources=['src/data/graph/lidar_graph.cu'],
)

setup(
    name='lidar_graph_package',
    version='1.0',
    description='LiDAR Graph construction in C and CUDA',
    ext_modules=[lidar_graph_c, lidar_graph_cuda],
    cmdclass={'build_ext': BuildExtension},
)
