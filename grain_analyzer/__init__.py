"""
Grain Analyzer Package

AFM 이미지에서 grain(입자) 검출 및 분석을 위한 패키지

주요 기능:
- 세그멘테이션: segment_classical, segment_stardist, segment_cellpose, segment_cellulus
- 통계 계산: calculate_grain_statistics, get_individual_grains
- 고수준 분석: analyze_grains, analyze_single_file_with_grain_data
- GPU 유틸리티: get_torch_device, check_tensorflow_gpu, print_gpu_info

GPU 가속 (환경에 따라 자동 감지):
- NVIDIA GPU: CUDA
- Apple Silicon: MPS (PyTorch) / Metal (TensorFlow)
- 그 외: CPU

사용 예시:
    >>> from grain_analyzer import segment_classical, calculate_grain_statistics
    >>> labels = segment_classical(height, meta)
    >>> stats = calculate_grain_statistics(labels, height, meta)
    
    # GPU 상태 확인
    >>> from grain_analyzer import print_gpu_info
    >>> print_gpu_info()
"""

# Segmentation functions
from .segmentation import (
    segment_classical,
    segment_stardist,
    segment_cellpose,
    segment_cellulus,
)

# Statistics functions
from .statistics import (
    calculate_grain_statistics,
    get_individual_grains,
)

# High-level analysis
from .analyze import (
    analyze_grains,
    analyze_single_file_with_grain_data,
)

# Data wrapper
from .afm_data_wrapper import AFMData

# GPU utilities
from .utils import (
    setup_gpu_environment,
    get_torch_device,
    check_tensorflow_gpu,
    print_gpu_info,
)

# Training utilities
from .train_model import (
    TrainingConfig,
    CellulusTrainer,
    get_hardware_info,
    print_hardware_info,
    setup_environment,
)

__version__ = "0.2.0"

__all__ = [
    # Segmentation
    "segment_classical",
    "segment_stardist",
    "segment_cellpose",
    "segment_cellulus",
    # Statistics
    "calculate_grain_statistics",
    "get_individual_grains",
    # Analysis
    "analyze_grains",
    "analyze_single_file_with_grain_data",
    # Data
    "AFMData",
    # GPU utilities
    "setup_gpu_environment",
    "get_torch_device",
    "check_tensorflow_gpu",
    "print_gpu_info",
    # Training
    "TrainingConfig",
    "CellulusTrainer",
    "get_hardware_info",
    "print_hardware_info",
    "setup_environment",
]
