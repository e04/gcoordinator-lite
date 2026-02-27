"""
This module provides functions for generating infill paths for 3D printing.

Functions:
- gyroid_infill: Generates a gyroid infill pattern for a given path or path list.
- line_infill: Generates a line infill pattern for a given path or path list.
"""

import numpy as np
from gcoordinator.path_generator import Path, PathList
from gcoordinator.utils.contour import find_contours
from gcoordinator.utils.polygon import points_in_polygon


def simplify_path(points, epsilon):
    """
    Ramer-Douglas-Peucker algorithm for 2D points to reduce file size.
    """
    if len(points) < 3:
        return points
    
    stack =[(0, len(points) - 1)]
    keep = np.ones(len(points), dtype=bool)
    
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
            
        line_vec = points[end] - points[start]
        line_len_sq = np.sum(line_vec**2)
        
        if line_len_sq == 0.0:
            diff = points[start+1:end] - points[start]
            dists = np.sum(diff**2, axis=1)
        else:
            diff = points[start+1:end] - points[start]
            cross = diff[:, 0] * line_vec[1] - diff[:, 1] * line_vec[0]
            dists = cross**2 / line_len_sq
            
        max_dist_idx = np.argmax(dists)
        max_dist = dists[max_dist_idx]
        
        if max_dist > epsilon**2:
            idx = start + 1 + max_dist_idx
            stack.append((start, idx))
            stack.append((idx, end))
        else:
            keep[start+1:end] = False
            
    return points[keep]

def gyroid_infill(path, infill_distance=1, value=0):
    """
    Generates a gyroid infill pattern for a given path.

    Args:
        path (Path or PathList): The path to generate the infill pattern for.
        infill_distance (float): The distance between the gyroid surfaces.
        value (float): The value to subtract from the gyroid equation.

    Returns:
        PathList: A PathList object containing the generated infill pattern.

    Raises:
        TypeError: If path is not a Path or PathList object.

    """
    if isinstance(path, Path):
        path_list = PathList([path])
    elif isinstance(path, PathList):
        path_list = path
    else:
        raise TypeError("path must be a Path or PathList object")

    # Set initial values
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    # Examine the coordinate sequence of each path object and
    #  update the minimum and maximum values
    for path in path_list.paths:
        x_coords = path.x
        y_coords = path.y
        if len(x_coords)>0:
            min_x = min(min_x, min(x_coords))
            max_x = max(max_x, max(x_coords))
            resolution_x = int((max_x-min_x)/0.4)
        if len(y_coords)>0:
            min_y = min(min_y, min(y_coords))
            max_y = max(max_y, max(y_coords))
            resolution_y = int((max_y - min_y)/0.4)
    z_height = path_list.paths[0].center[2]

    # Grid parameters
    # Resolution of the grid
    x = np.linspace(min_x, max_x, resolution_x)
    y = np.linspace(min_y, max_y, resolution_y)
    X, Y = np.meshgrid(x, y)

    # Equation for the Gyroid surface
    theta = np.pi/4
    p = np.pi*np.cos(theta)*np.sqrt(2)/infill_distance # Period of the gyroid surface
    equation = np.sin((X *np.cos(theta) + Y *np.sin(theta))*p) * np.cos((-X *np.sin(theta) + Y *np.cos(theta))*p) \
                + np.sin((-X *np.sin(theta) + Y *np.cos(theta))*p) * np.cos(z_height*p ) \
                + np.sin(z_height*p ) * np.cos((X *np.cos(theta) + Y *np.sin(theta))*p)\
                -value

    insides = []
    for path in path_list.paths:
        x_list = path.x
        y_list = path.y

        # Determine the inside region
        polygon = np.column_stack([x_list, y_list])
        points = np.column_stack((X.flatten(), Y.flatten()))
        inside = points_in_polygon(points, polygon)
        inside = inside.reshape(X.shape).astype(float)
        inside[inside == 1] = -1 # change inside to -1
        inside[inside == 0] = 1  # Change outside  to 1
        insides.append(inside)

    result = insides[0]  # Set the first ndarray as the initial value

    for i in range(1, len(insides)):
        result = np.multiply(result, insides[i])  # Calculate the Adamar product

    # Replace -1 with np.nan
    result[result == 1] = np.nan

    # Calculate contours
    slice_plane = equation * result
    contour_paths = find_contours(x, y, slice_plane, level=0)

    infill_path_list = []
    for contour_path in contour_paths:
        x_coords = contour_path[:, 0]
        y_coords = contour_path[:, 1]
        z_coords = np.full_like(x_coords, z_height)
        wall = Path(x_coords, y_coords, z_coords)
        infill_path_list.append(wall)

    return PathList(infill_path_list)

def line_infill(path, infill_distance=1, angle=np.pi/4):
    """
    Generates a line infill pattern for a given path.

    Args:
        path (Path or PathList): The path to generate the infill pattern for.
        infill_distance (float, optional): The distance between the lines in the infill pattern. Defaults to 1.
        angle (float, optional): The angle of the infill pattern in radians. Defaults to np.pi/4.

    Returns:
        PathList: A PathList object containing the infill pattern.

    Raises:
        TypeError: If the path argument is not a Path or PathList object.

    """
    if isinstance(path, Path):
        path_list = PathList([path])
    elif isinstance(path, PathList):
        path_list = path
    else:
        raise TypeError("path must be a Path or PathList object")

    if len(path_list.paths) == 0:
        return PathList([])

    if infill_distance <= 0:
        raise ValueError("infill_distance must be positive")

    z_height = path_list.paths[0].center[2]
    
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    
    # Collect all edges from polygons and transform into u, v line-aligned coordinates
    edges =[]
    v_min = float('inf')
    v_max = float('-inf')
    
    for path_obj in path_list.paths:
        x_coords = path_obj.x
        y_coords = path_obj.y
        if len(x_coords) < 2:
            continue
            
        u_coords = x_coords * cos_a + y_coords * sin_a
        v_coords = x_coords * sin_a - y_coords * cos_a
        
        v_min = min(v_min, np.min(v_coords))
        v_max = max(v_max, np.max(v_coords))
        
        for i in range(len(x_coords) - 1):
            edges.append(( (u_coords[i], v_coords[i]), (u_coords[i+1], v_coords[i+1]) ))
            
        if len(x_coords) > 2:
            edges.append(( (u_coords[-1], v_coords[-1]), (u_coords[0], v_coords[0]) ))

    if not edges:
        return PathList([])

    edges_arr = np.array(edges) # shape (N, 2, 2)
    u1 = edges_arr[:, 0, 0]
    v1 = edges_arr[:, 0, 1]
    u2 = edges_arr[:, 1, 0]
    v2 = edges_arr[:, 1, 1]
    
    # Determine the integer scaling steps mapping out the infinite lines
    k_min = int(np.ceil(v_min / infill_distance))
    k_max = int(np.floor(v_max / infill_distance))
    
    if k_max < k_min:
        return PathList([])
        
    k_vals = np.arange(k_min, k_max + 1)
    infill_path_list =[]
    
    # Fast vectorized geometric analytical intersections 
    for k in k_vals:
        V = k * infill_distance
        
        # Identifies when an edge straddles over the raycast coordinate
        mask1 = (v1 <= V) & (V < v2)
        mask2 = (v2 <= V) & (V < v1)
        mask = mask1 | mask2
        
        if not np.any(mask):
            continue
            
        u1_m = u1[mask]
        v1_m = v1[mask]
        u2_m = u2[mask]
        v2_m = v2[mask]
        
        # Formulate explicit intersection tracking line distance
        t = (V - v1_m) / (v2_m - v1_m)
        u_inter = u1_m + t * (u2_m - u1_m)
        
        u_inter = np.sort(u_inter)
        
        # Stitch up pairs of internal intersections representing exactly where it is 'infill'ing (Parity rule)
        for i in range(0, len(u_inter) - 1, 2):
            u_start = u_inter[i]
            u_end = u_inter[i+1]
            
            # Avoid duplicate segments generated by collinear vertices
            if u_end - u_start < 1e-5:
                continue
                
            # Convert coordinate basis back to natural (X, Y) layout
            x_start = u_start * cos_a + V * sin_a
            y_start = u_start * sin_a - V * cos_a
            
            x_end = u_end * cos_a + V * sin_a
            y_end = u_end * sin_a - V * cos_a
            
            wall = Path(np.array([x_start, x_end]), 
                        np.array([y_start, y_end]), 
                        np.array([z_height, z_height]))
            infill_path_list.append(wall)
            
    return PathList(infill_path_list)
