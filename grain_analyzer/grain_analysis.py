"""
Grain Analysis Module
Grain detection and analysis functions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage import measure


class AFMGrainAnalyzer:
    """AFM grain analysis class"""
    
    def __init__(self):
        self.grain_labels = None
        self.grain_props = None
    
    def segment_by_marker_growth(
        self,
        height: np.ndarray,
        marker_coords_px: np.ndarray,
        *,
        meta: Optional[Dict] = None,
        mask: Optional[np.ndarray] = None,
        max_radius_nm: Optional[float] = None,
        anisotropic_nm_metric: bool = True,
    ) -> np.ndarray:
        """마커를 중심으로 원/타원을 키워가며 분할.

        - 픽셀 비등방성을 nm 샘플링으로 보정하여 타원 거리 기반 최근접 마커에 할당
        - mask가 주어지면 그 영역 내에서만 분할
        - max_radius_nm로 마커별 최대 성장 반경 제한 (nm)
        Returns: labels (int32)
        """
        h, w = height.shape
        labels = np.zeros((h, w), dtype=np.int32)
        if marker_coords_px is None or np.size(marker_coords_px) == 0:
            return labels

        # 픽셀→nm 스케일
        xp_nm, yp_nm = 1.0, 1.0
        if meta and 'pixel_nm' in meta:
            xp_nm, yp_nm = float(meta['pixel_nm'][0]), float(meta['pixel_nm'][1])
        sampling = (yp_nm, xp_nm) if anisotropic_nm_metric else None

        # 시드 마스크
        seeds_mask = np.zeros((h, w), dtype=bool)
        valid_coords: List[Tuple[int, int]] = []
        for r, c in np.asarray(marker_coords_px, dtype=int):
            if 0 <= r < h and 0 <= c < w:
                seeds_mask[r, c] = True
                valid_coords.append((r, c))
        if not np.any(seeds_mask):
            return labels

        # 최근접 시드 인덱스 (nm metric)
        dist_nm, (iy, ix) = ndimage.distance_transform_edt(~seeds_mask, sampling=sampling, return_indices=True)

        # 시드 라벨 테이블
        seed_labels = np.zeros_like(labels)
        for i, (r, c) in enumerate(valid_coords, start=1):
            seed_labels[r, c] = i
        labels = seed_labels[iy, ix]

        # 마스크 제한
        if mask is not None:
            labels = np.where(mask.astype(bool), labels, 0)

        # 반경 제한
        if max_radius_nm is not None and max_radius_nm > 0:
            labels = np.where(dist_nm <= float(max_radius_nm), labels, 0)

        return labels

    def calculate_grain_statistics(self, grain_labels: np.ndarray, grain_props: Dict, 
                                 meta: Optional[Dict] = None) -> Dict:
        """입자 통계 계산
        
        Parameters
        ----------
        grain_labels : np.ndarray
            입자 라벨링
        grain_props : dict
            입자 속성
        meta : dict, optional
            메타데이터 (픽셀 크기 정보)
        
        Returns
        -------
        dict
            입자 통계
        """
        if not grain_props or 'area' not in grain_props or len(grain_props['area']) == 0:
            return {}

        areas = grain_props['area']
        perimeters = grain_props['perimeter']
        eccentricities = grain_props['eccentricity']
        solidities = grain_props['solidity']

        if 'centroid' in grain_props:
            centroids = grain_props['centroid']
        elif 'centroid-0' in grain_props and 'centroid-1' in grain_props:
            centroids = [(grain_props['centroid-0'][i], grain_props['centroid-1'][i])
                         for i in range(len(areas))]
        else:
            centroids = [(0, 0)] * len(areas)

        major_axis = grain_props['major_axis_length']
        minor_axis = grain_props['minor_axis_length']

        equivalent_diameters = np.sqrt(4 * areas / np.pi)
        
        # Handle division by zero for aspect ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            aspect_ratios = major_axis / minor_axis
            # Replace infinity or NaN with 0 (or another suitable default like 1.0)
            # Typically for lines (minor_axis=0), aspect ratio is undefined or infinite
            aspect_ratios = np.nan_to_num(aspect_ratios, nan=0.0, posinf=0.0, neginf=0.0)

        total_area_px = grain_labels.shape[0] * grain_labels.shape[1]
        grain_area_px = np.sum(areas)
        grain_density = len(areas) / total_area_px
        area_fraction = grain_area_px / total_area_px

        # nm 단위로 변환
        if meta and 'pixel_nm' in meta:
            pixel_x_nm, pixel_y_nm = meta['pixel_nm']
            pixel_area_nm2 = pixel_x_nm * pixel_y_nm
            
            areas_nm2 = areas * pixel_area_nm2
            perimeters_nm = perimeters * np.sqrt(pixel_area_nm2)
            equivalent_diameters_nm = equivalent_diameters * np.sqrt(pixel_area_nm2)
            major_axis_nm = major_axis * np.sqrt(pixel_area_nm2)
            minor_axis_nm = minor_axis * np.sqrt(pixel_area_nm2)
        else:
            areas_nm2 = areas
            perimeters_nm = perimeters
            equivalent_diameters_nm = equivalent_diameters
            major_axis_nm = major_axis
            minor_axis_nm = minor_axis

        individual_grains: List[Dict] = []
        for i in range(len(areas)):
            grain_data = {
                'grain_id': i + 1,
                'area_px': float(areas[i]),
                'area_nm2': float(areas_nm2[i]),
                'perimeter_px': float(perimeters[i]),
                'perimeter_nm': float(perimeters_nm[i]),
                'equivalent_diameter_px': float(equivalent_diameters[i]),
                'equivalent_diameter_nm': float(equivalent_diameters_nm[i]),
                'aspect_ratio': float(aspect_ratios[i]),
                'eccentricity': float(eccentricities[i]),
                'solidity': float(solidities[i]),
                'orientation': float(grain_props['orientation'][i]),
                'centroid': centroids[i],
                'major_axis_length_px': float(major_axis[i]),
                'major_axis_length_nm': float(major_axis_nm[i]),
                'minor_axis_length_px': float(minor_axis[i]),
                'minor_axis_length_nm': float(minor_axis_nm[i]),
            }
            individual_grains.append(grain_data)

        return {
            'num_grains': int(len(areas)),
            'mean_area_px': float(np.mean(areas)),
            'mean_area_nm2': float(np.mean(areas_nm2)),
            'std_area_px': float(np.std(areas)),
            'std_area_nm2': float(np.std(areas_nm2)),
            'min_area_px': float(np.min(areas)),
            'min_area_nm2': float(np.min(areas_nm2)),
            'max_area_px': float(np.max(areas)),
            'max_area_nm2': float(np.max(areas_nm2)),
            'mean_diameter_px': float(np.mean(equivalent_diameters)),
            'mean_diameter_nm': float(np.mean(equivalent_diameters_nm)),
            'std_diameter_px': float(np.std(equivalent_diameters)),
            'std_diameter_nm': float(np.std(equivalent_diameters_nm)),
            'mean_perimeter_px': float(np.mean(perimeters)),
            'mean_perimeter_nm': float(np.mean(perimeters_nm)),
            'mean_eccentricity': float(np.mean(eccentricities)),
            'mean_aspect_ratio': float(np.mean(aspect_ratios)),
            'mean_solidity': float(np.mean(solidities)),
            'grain_density': float(grain_density),
            'area_fraction': float(area_fraction),
            'equivalent_diameters_px': equivalent_diameters,
            'equivalent_diameters_nm': equivalent_diameters_nm,
            'areas_px': areas,
            'areas_nm2': areas_nm2,
            'perimeters_px': perimeters,
            'perimeters_nm': perimeters_nm,
            'eccentricities': eccentricities,
            'aspect_ratios': aspect_ratios,
            'solidities': solidities,
            'individual_grains': individual_grains,
        }


# Convenience functions
def segment_by_marker_growth(height: np.ndarray,
                             marker_coords_px: np.ndarray,
                             *,
                             meta: Optional[Dict] = None,
                             mask: Optional[np.ndarray] = None,
                             max_radius_nm: Optional[float] = None,
                             anisotropic_nm_metric: bool = True) -> np.ndarray:
    """편의 함수: 마커 중심 원/타원 성장 기반 분할"""
    analyzer = AFMGrainAnalyzer()
    return analyzer.segment_by_marker_growth(
        height,
        marker_coords_px,
        meta=meta,
        mask=mask,
        max_radius_nm=max_radius_nm,
        anisotropic_nm_metric=anisotropic_nm_metric,
    )


def calculate_grain_statistics(grain_labels: np.ndarray, grain_props: Dict, 
                             meta: Optional[Dict] = None) -> Dict:
    """편의 함수: 입자 통계 계산"""
    analyzer = AFMGrainAnalyzer()
    return analyzer.calculate_grain_statistics(grain_labels, grain_props, meta)

