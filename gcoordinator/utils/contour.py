import numpy as np
from typing import List, Tuple


# Marching squares lookup table
# Maps case index to list of edge pairs to connect
# Edge indices: 0=bottom, 1=right, 2=top, 3=left
_EDGE_TABLE = {
    0: [], 1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)],
    4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)],
    8: [(2, 3)], 9: [(2, 0)], 10: [(0, 1), (2, 3)], 11: [(2, 1)],
    12: [(1, 3)], 13: [(1, 0)], 14: [(0, 3)], 15: []
}


def find_contours(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                  level: float = 0) -> List[np.ndarray]:
    """
    Find contour lines at a given level using the Marching Squares algorithm.
    Optimized vectorized implementation.
    
    Args:
        x: 1D array of x coordinates (length M)
        y: 1D array of y coordinates (length N)
        z: 2D array of z values (shape NxM)
        level: The contour level to find
        
    Returns:
        List of paths, where each path is an Nx2 numpy array of (x, y) coordinates
    """
    ny, nx = z.shape
    
    if len(x) != nx or len(y) != ny:
        raise ValueError(f"Shape mismatch: x has {len(x)} elements, y has {len(y)} elements, "
                        f"but z has shape {z.shape}")
    
    if ny < 2 or nx < 2:
        return []
    
    # Get corner values for all cells at once
    # z00 = bottom-left, z01 = bottom-right, z11 = top-right, z10 = top-left
    z00 = z[:-1, :-1]  # (ny-1, nx-1)
    z01 = z[:-1, 1:]
    z11 = z[1:, 1:]
    z10 = z[1:, :-1]
    
    # Check for NaN - skip cells with any NaN
    valid = ~(np.isnan(z00) | np.isnan(z01) | np.isnan(z11) | np.isnan(z10))
    
    # Calculate case index for all cells
    above0 = (z00 >= level).astype(np.int32)
    above1 = (z01 >= level).astype(np.int32)
    above2 = (z11 >= level).astype(np.int32)
    above3 = (z10 >= level).astype(np.int32)
    case_index = above0 + 2 * above1 + 4 * above2 + 8 * above3
    
    # Cell coordinates
    cell_i, cell_j = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
    
    # Collect segments
    all_segments = []
    
    # Process each non-trivial case
    for case in range(1, 15):
        mask = (case_index == case) & valid
        if not np.any(mask):
            continue
        
        edge_pairs = _EDGE_TABLE[case]
        if not edge_pairs:
            continue
        
        # Get cell indices where this case occurs
        ci = cell_i[mask]
        cj = cell_j[mask]
        
        # Get corner values for these cells
        v0 = z00[mask]
        v1 = z01[mask]
        v2 = z11[mask]
        v3 = z10[mask]
        
        # Precompute edge intersection points
        # Edge 0: bottom (corner 0 to 1)
        # Edge 1: right (corner 1 to 2)  
        # Edge 2: top (corner 2 to 3)
        # Edge 3: left (corner 3 to 0)
        
        def interp_edge(edge_idx):
            if edge_idx == 0:  # bottom
                va, vb = v0, v1
                xa, xb = x[cj], x[cj + 1]
                ya = yb = y[ci]
            elif edge_idx == 1:  # right
                va, vb = v1, v2
                xa = xb = x[cj + 1]
                ya, yb = y[ci], y[ci + 1]
            elif edge_idx == 2:  # top
                va, vb = v2, v3
                xa, xb = x[cj + 1], x[cj]
                ya = yb = y[ci + 1]
            else:  # left
                va, vb = v3, v0
                xa = xb = x[cj]
                ya, yb = y[ci + 1], y[ci]
            
            dv = vb - va
            t = np.where(np.abs(dv) < 1e-10, 0.5, (level - va) / dv)
            px = xa + t * (xb - xa)
            py = ya + t * (yb - ya)
            return px, py
        
        # Generate segments for each edge pair
        for e1, e2 in edge_pairs:
            px1, py1 = interp_edge(e1)
            px2, py2 = interp_edge(e2)
            
            for k in range(len(ci)):
                all_segments.append(((px1[k], py1[k]), (px2[k], py2[k])))
    
    # Connect segments into paths using union-find approach
    return _connect_segments_fast(all_segments)


def _connect_segments_fast(segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                           tolerance: float = 1e-8) -> List[np.ndarray]:
    """
    Connect line segments into continuous paths using a hash-based approach.
    """
    if not segments:
        return []
    
    # Quantize points for hashing
    def quantize(p):
        return (round(p[0] / tolerance), round(p[1] / tolerance))
    
    # Build adjacency list
    from collections import defaultdict
    adjacency = defaultdict(list)
    
    for seg_idx, (p0, p1) in enumerate(segments):
        q0 = quantize(p0)
        q1 = quantize(p1)
        adjacency[q0].append((seg_idx, 0))  # 0 means p0 is the connection point
        adjacency[q1].append((seg_idx, 1))  # 1 means p1 is the connection point
    
    used = [False] * len(segments)
    paths = []
    
    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue
        
        used[start_idx] = True
        seg = segments[start_idx]
        path = [seg[0], seg[1]]
        
        # Extend in both directions
        for direction in [1, 0]:  # 1 = extend from end, 0 = extend from start
            while True:
                if direction == 1:
                    end_point = path[-1]
                else:
                    end_point = path[0]
                
                q = quantize(end_point)
                found = False
                
                for seg_idx, endpoint_idx in adjacency[q]:
                    if used[seg_idx]:
                        continue
                    
                    seg = segments[seg_idx]
                    used[seg_idx] = True
                    found = True
                    
                    # Add the other endpoint to path
                    if endpoint_idx == 0:
                        new_point = seg[1]
                    else:
                        new_point = seg[0]
                    
                    if direction == 1:
                        path.append(new_point)
                    else:
                        path.insert(0, new_point)
                    break
                
                if not found:
                    break
        
        paths.append(np.array(path))
    
    return paths
