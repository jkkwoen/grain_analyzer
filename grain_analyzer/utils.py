"""
Utility functions for grain analysis
"""

import numpy as np
from typing import Tuple
from .corrections import AFMCorrections


def nm2_to_px_area(area_nm2: float, pixel_nm: Tuple[float, float]) -> int:
    """Convert area in nm² to pixels
    
    Parameters
    ----------
    area_nm2 : float
        Area in nm²
    pixel_nm : Tuple[float, float]
        Pixel size in nm (x, y)
    
    Returns
    -------
    int
        Area in pixels
    """
    px_area_nm2 = float(pixel_nm[0]) * float(pixel_nm[1])
    if px_area_nm2 <= 0:
        return int(max(1.0, area_nm2))
    return max(1, int(round(area_nm2 / px_area_nm2)))


def apply_grain_excluded_flat_correction(
    height_data: np.ndarray, 
    grain_mask: np.ndarray,
    grain_labels: np.ndarray
) -> np.ndarray:
    """
    Grain 영역을 제외하고 flat 보정을 적용
    
    Parameters
    ----------
    height_data : np.ndarray
        보정할 높이 데이터 (after_slope_correction)
    grain_mask : np.ndarray
        Grain 영역을 나타내는 마스크 (boolean)
    grain_labels : np.ndarray
        Grain 라벨링된 데이터
        
    Returns
    -------
    np.ndarray
        Grain 영역 제외 flat 보정된 데이터
    """
    height_corrected = height_data.copy()
    
    # Grain 영역이 아닌 배경 영역만 사용
    background_mask = ~grain_mask
    
    if np.sum(background_mask) < height_data.size * 0.1:  # 배경이 너무 작으면
        print("⚠️  Warning: Background area too small, using standard flat correction")
        corrector = AFMCorrections()
        corrector.set_flat_method("line_by_line")
        return corrector.correct_flat(height_corrected)
    
    # 외곽선 경계에서 기준값 계산 (grain 제외)
    boundary_mask = np.zeros_like(grain_mask)
    boundary_width = max(2, min(height_data.shape) // 50)
    
    # 상하좌우 외곽선 생성
    boundary_mask[:boundary_width, :] = True  # 상단
    boundary_mask[-boundary_width:, :] = True  # 하단
    boundary_mask[:, :boundary_width] = True  # 좌측
    boundary_mask[:, -boundary_width:] = True  # 우측
    
    # Grain 제외된 외곽선만 사용
    reference_mask = background_mask & boundary_mask
    
    if np.sum(reference_mask) > 0:
        reference_level = np.mean(height_data[reference_mask])
        height_corrected = height_data - reference_level
    else:
        # 외곽선에서 참조를 구할 수 없으면 전체 배경 사용
        reference_level = np.mean(height_data[background_mask])
        height_corrected = height_data - reference_level
    
    # 라인별 최종 평면 보정 (배경 영역만 적용)
    for i in range(height_data.shape[0]):
        row_mask = background_mask[i, :]
        if np.sum(row_mask) > 0:
            row_avg = np.mean(height_corrected[i, row_mask])
            if not np.isnan(row_avg):
                height_corrected[i, :] -= row_avg
    
    for j in range(height_data.shape[1]):
        col_mask = background_mask[:, j]
        if np.sum(col_mask) > 0:
            col_avg = np.mean(height_corrected[:, col_mask])
            if not np.isnan(col_avg):
                height_corrected[:, j] -= col_avg
    
    return height_corrected

