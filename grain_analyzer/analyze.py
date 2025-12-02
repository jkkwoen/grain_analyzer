"""
Main analysis function for grain analysis
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import math

from .afm_data_wrapper import AFMData
from .utils import nm2_to_px_area, apply_grain_excluded_flat_correction
from .grain_analysis import segment_by_marker_growth, calculate_grain_statistics
from skimage import measure


def analyze_single_file_with_grain_data(
    xqd_file: Path, 
    output_dir: Path,
    gaussian_sigma: float = 1.0,
    min_area_nm2: float = 78.5,
    min_peak_separation_nm: float = 10.0
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Analyze a single XQD file and return grain data and statistics.
    
    Parameters
    ----------
    xqd_file : Path
        Path to the XQD file
    output_dir : Path
        Output directory for PDF file
    gaussian_sigma : float
        Gaussian smoothing sigma
    min_area_nm2 : float
        Minimum grain area in nm²
    min_peak_separation_nm : float
        Minimum peak separation in nm
    
    Returns
    -------
    Tuple[bool, Optional[Dict], Optional[Dict], Optional[str]]
        (success, individual_grain_data, grain_stats, pdf_path)
        - success: Whether analysis was successful
        - individual_grain_data: List of dictionaries with individual grain properties
        - grain_stats: Dictionary with overall grain statistics
        - pdf_path: Path to the generated PDF file
    """
    
    print(f"📊 Processing: {xqd_file.name}")
    
    try:
        # Load data using AFMData
        data = AFMData(str(xqd_file))
        print(f"   ✓ Data loaded: {data.get_data().shape}")
        
        # Create output directory for this file
        file_output_dir = output_dir / xqd_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Grain Analysis
        print("   🔬 Performing grain analysis...")
        
        # Apply corrections using AFMData
        data.first_correction().second_correction().third_correction()
        data.flat_correction("line_by_line").baseline_correction("min_to_zero")
        
        # Get corrected data and metadata
        height_corrected = data.get_data()
        meta = data.get_meta()
        height_raw = data.get_raw_data()
        
        pixel_nm = meta.get("pixel_nm", (1.0, 1.0))
        xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])
        x_size_nm, y_size_nm = meta.get("scan_size_nm", (height_raw.shape[1], height_raw.shape[0]))
        extent = [0, x_size_nm, 0, y_size_nm]
        px_area_nm2 = xp_nm * yp_nm
        
        # Grain detection
        min_area_px = nm2_to_px_area(min_area_nm2, pixel_nm)
        from skimage.filters import gaussian, threshold_otsu
        from skimage.morphology import remove_small_objects
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from sklearn.cluster import DBSCAN
        from skimage.segmentation import find_boundaries
        
        h_smooth = gaussian(height_corrected, sigma=gaussian_sigma, preserve_range=True)
        thr = threshold_otsu(h_smooth)
        binary = h_smooth > thr
        binary = remove_small_objects(binary, min_size=min_area_px)
        
        # Distance transform and peak detection
        distance = ndi.distance_transform_edt(binary)
        avg_px_nm = float(np.mean(pixel_nm)) if np.all(np.isfinite(pixel_nm)) else 1.0
        approx_min_radius_px = max(1, int(round((10.0 / avg_px_nm) / 2.0)))
        
        coords = peak_local_max(
            distance,
            labels=binary,
            min_distance=approx_min_radius_px,
            exclude_border=False,
            footprint=None,
            threshold_abs=0.0,
            threshold_rel=0.0,
        )
        
        rep_coords = np.zeros((0, 2), dtype=int)
        
        if coords.size > 0:
            coords_nm = np.column_stack([coords[:, 0] * yp_nm, coords[:, 1] * xp_nm])
            try:
                clustering = DBSCAN(eps=float(min_peak_separation_nm), min_samples=1, metric="euclidean").fit(coords_nm)
                labels_db = clustering.labels_
                rep_indices = []
                for lab in np.unique(labels_db):
                    idxs = np.where(labels_db == lab)[0]
                    best_i = idxs[np.argmax(distance[coords[idxs, 0], coords[idxs, 1]])]
                    rep_indices.append(best_i)
                rep_coords = coords[np.array(rep_indices)]
            except Exception:
                rep_coords = coords
        
        # Voronoi segmentation
        voronoi_labels = segment_by_marker_growth(
            height_corrected,
            rep_coords,
            meta=meta,
            mask=binary,
            max_radius_nm=None,
            anisotropic_nm_metric=True,
        )
        boundaries = find_boundaries(voronoi_labels, mode="outer") if voronoi_labels.max() > 0 else np.zeros_like(voronoi_labels, dtype=bool)
        
        # Generate grain mask
        grain_mask = voronoi_labels > 0
        unique_labels = np.unique(voronoi_labels)
        unique_labels = unique_labels[unique_labels > 0]
        
        print(f"   ✓ Grain analysis completed: {len(unique_labels)} grains detected")
        
        # 2. Calculate grain statistics
        print("   📊 Calculating grain statistics...")
        
        # Get region properties for grain statistics
        if int(voronoi_labels.max()) > 0:
            grain_props = measure.regionprops_table(
                voronoi_labels,
                intensity_image=height_corrected,
                properties=['area', 'centroid', 'eccentricity',
                           'major_axis_length', 'minor_axis_length',
                           'orientation', 'perimeter', 'solidity']
            )
        else:
            grain_props = {}
        
        # Calculate overall grain statistics
        grain_stats = calculate_grain_statistics(voronoi_labels, grain_props, meta)
        
        # 3. Extract individual grain data
        print("   📋 Extracting individual grain data...")
        individual_grain_data = _extract_individual_grain_data(
            voronoi_labels, height_corrected, meta, rep_coords, pixel_nm
        )
        
        # 4. Create PDF with original and grain_mask plots
        print("   📄 Creating PDF plot...")
        pdf_path = _create_grain_analysis_pdf(
            height_raw, height_corrected, grain_mask, voronoi_labels, boundaries,
            xqd_file.stem, file_output_dir, extent, len(unique_labels)
        )
        
        print(f"   ✅ All analyses completed for {xqd_file.name}")
        return True, individual_grain_data, grain_stats, str(pdf_path)
        
    except Exception as e:
        print(f"   ❌ Error processing {xqd_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def _extract_individual_grain_data(
    labels: np.ndarray,
    height_nm: np.ndarray,
    meta: Dict[str, Any],
    rep_coords: np.ndarray,
    pixel_nm: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """Extract individual grain data as list of dictionaries."""
    
    if int(labels.max()) == 0:
        return []
    
    xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])
    px_area_nm2 = xp_nm * yp_nm
    
    # Get region properties
    stats = measure.regionprops_table(
        labels,
        intensity_image=height_nm,
        properties=[
            'label', 'area', 'eccentricity', 'centroid', 'perimeter',
            'major_axis_length', 'minor_axis_length', 'orientation',
            'convex_area', 'solidity'
        ]
    )
    
    # Map rep_coords to labels
    label_to_peak_rc: Dict[int, Tuple[int, int]] = {}
    if rep_coords is not None and rep_coords.size > 0:
        for r, c in np.asarray(rep_coords, dtype=int):
            if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]:
                lab = int(labels[r, c])
                if lab > 0 and lab not in label_to_peak_rc:
                    label_to_peak_rc[lab] = (r, c)
    
    individual_grains = []
    labels_list = stats['label']
    
    for idx in range(len(labels_list)):
        lab_id = int(labels_list[idx])
        if lab_id <= 0:
            continue
        
        mask = (labels == lab_id)
        if not np.any(mask):
            continue
        
        # Basic properties
        area_px = float(stats['area'][idx])
        area_nm2 = area_px * px_area_nm2
        diameter_nm = 2.0 * math.sqrt(area_nm2 / math.pi) if area_nm2 > 0 else 0.0
        eq_radius_nm = diameter_nm / 2.0
        
        # Centroid
        cent_y = stats['centroid-0'][idx]
        cent_x = stats['centroid-1'][idx]
        cx_nm = float(cent_x) * xp_nm
        cy_nm = float(cent_y) * yp_nm
        centroid_h_nm = float(np.mean(height_nm[mask]))
        
        # Peak
        if lab_id in label_to_peak_rc:
            pr, pc = label_to_peak_rc[lab_id]
        else:
            ys, xs = np.nonzero(mask)
            if ys.size > 0:
                vals = height_nm[ys, xs]
                kmax = int(np.argmax(vals))
                pr, pc = int(ys[kmax]), int(xs[kmax])
            else:
                pr, pc = int(round(cent_y)), int(round(cent_x))
        
        peak_x_nm = float(pc) * xp_nm
        peak_y_nm = float(pr) * yp_nm
        peak_h_nm = float(height_nm[pr, pc])
        peak_to_centroid_dist_nm = float(math.hypot(peak_x_nm - cx_nm, peak_y_nm - cy_nm))
        
        # Volume
        vol_nm3 = float(np.sum(height_nm[mask]) * px_area_nm2)
        
        # Shape properties
        major_axis_px = float(stats['major_axis_length'][idx])
        minor_axis_px = float(stats['minor_axis_length'][idx])
        major_axis_nm = major_axis_px * math.sqrt(px_area_nm2)
        minor_axis_nm = minor_axis_px * math.sqrt(px_area_nm2)
        aspect_ratio = major_axis_px / minor_axis_px if minor_axis_px > 0 else 0.0
        
        # Height statistics
        vals_mask = height_nm[mask]
        height_mean = float(np.mean(vals_mask))
        height_std = float(np.std(vals_mask))
        height_min = float(np.min(vals_mask))
        height_max = float(np.max(vals_mask))
        
        grain_data = {
            'grain_id': int(lab_id),
            'area_px': area_px,
            'area_nm2': area_nm2,
            'diameter_nm': diameter_nm,
            'equivalent_radius_nm': eq_radius_nm,
            'volume_nm3': vol_nm3,
            'centroid_x_nm': cx_nm,
            'centroid_y_nm': cy_nm,
            'centroid_height_nm': centroid_h_nm,
            'peak_x_nm': peak_x_nm,
            'peak_y_nm': peak_y_nm,
            'peak_height_nm': peak_h_nm,
            'peak_to_centroid_dist_nm': peak_to_centroid_dist_nm,
            'major_axis_nm': major_axis_nm,
            'minor_axis_nm': minor_axis_nm,
            'aspect_ratio': aspect_ratio,
            'orientation_deg': float(stats['orientation'][idx]) * 180.0 / math.pi,
            'perimeter_nm': float(stats['perimeter'][idx]) * math.sqrt(px_area_nm2),
            'eccentricity': float(stats['eccentricity'][idx]),
            'solidity': float(stats['solidity'][idx]),
            'convex_area_nm2': float(stats['convex_area'][idx]) * px_area_nm2,
            'height_mean_nm': height_mean,
            'height_std_nm': height_std,
            'height_min_nm': height_min,
            'height_max_nm': height_max,
        }
        
        individual_grains.append(grain_data)
    
    return individual_grains


def _create_grain_analysis_pdf(
    height_raw: np.ndarray,
    height_corrected: np.ndarray,
    grain_mask: np.ndarray,
    grain_labels: np.ndarray,
    boundaries: np.ndarray,
    stem: str,
    output_dir: Path,
    extent: List[float],
    num_grains: int
) -> Path:
    """Create PDF with original and grain_mask plots."""
    
    pdf_path = output_dir / f"{stem}_grain_analysis.pdf"
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Original height data
    im1 = axes[0].imshow(height_raw, cmap='gray', origin='lower', extent=extent, 
                         vmin=np.percentile(height_raw, 2), 
                         vmax=np.percentile(height_raw, 98))
    axes[0].set_xlabel('X [nm]')
    axes[0].set_ylabel('Y [nm]')
    axes[0].set_title('Original Height Data')
    plt.colorbar(im1, ax=axes[0], label='Height [nm]')
    
    # Right: Grain mask overlay
    im2 = axes[1].imshow(height_corrected, cmap='gray', origin='lower', extent=extent,
                         vmin=np.percentile(height_corrected, 2),
                         vmax=np.percentile(height_corrected, 98))
    
    # Overlay grain labels with colormap
    if grain_labels.max() > 0:
        im3 = axes[1].imshow(grain_labels, cmap='tab20', origin='lower', extent=extent, 
                            alpha=0.5, vmin=0, vmax=grain_labels.max())
        plt.colorbar(im3, ax=axes[1], label='Grain ID')
    
    # Draw boundaries
    if np.any(boundaries):
        y_coords, x_coords = np.where(boundaries)
        if extent:
            x_nm = x_coords * (extent[1] / height_corrected.shape[1])
            y_nm = y_coords * (extent[3] / height_corrected.shape[0])
            axes[1].scatter(x_nm, y_nm, c='red', s=0.1, alpha=0.6)
        else:
            axes[1].scatter(x_coords, y_coords, c='red', s=0.1, alpha=0.6)
    
    axes[1].set_xlabel('X [nm]')
    axes[1].set_ylabel('Y [nm]')
    axes[1].set_title(f'Grain Mask Overlay (N={num_grains} grains)')
    
    fig.suptitle(f'Grain Analysis - {stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ PDF saved: {pdf_path}")
    
    return pdf_path

