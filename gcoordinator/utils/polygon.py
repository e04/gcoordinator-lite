import numpy as np

def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Determine if points are inside a polygon using the ray casting algorithm.
    Fully vectorized implementation for performance.
    
    Args:
        points: Nx2 array of (x, y) coordinates to test
        polygon: Mx2 array of (x, y) coordinates defining the polygon vertices
        
    Returns:
        Boolean array of length N, True if point is inside the polygon
        
    Example:
        >>> polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        >>> points = np.array([[0.5, 0.5], [2, 2]])
        >>> points_in_polygon(points, polygon)
        array([ True, False])
    """
    n_points = len(points)
    n_vertices = len(polygon)
    
    if n_vertices < 3:
        return np.zeros(n_points, dtype=bool)
    
    # Ensure polygon is closed
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])
    
    # Get edge vertices (vectorized)
    x1 = polygon[:-1, 0]  # shape (M,)
    y1 = polygon[:-1, 1]
    x2 = polygon[1:, 0]
    y2 = polygon[1:, 1]
    
    # Points
    px = points[:, 0]  # shape (N,)
    py = points[:, 1]
    
    # Compute for all edges at once using broadcasting
    # px, py: (N,) -> (N, 1) for broadcasting with (M,) edge arrays
    px = px[:, np.newaxis]  # (N, 1)
    py = py[:, np.newaxis]  # (N, 1)
    
    # Skip horizontal edges
    non_horizontal = y1 != y2  # (M,)
    
    # Condition 1: point's y is between edge's y range
    cond1 = (y1 > py) != (y2 > py)  # (N, M)
    
    # Calculate x intersection for all points and all edges
    # Avoid division by zero for horizontal edges (will be masked out)
    dy = y2 - y1
    dy = np.where(dy == 0, 1, dy)  # Avoid div by zero
    x_intersect = x1 + (py - y1) * (x2 - x1) / dy  # (N, M)
    
    # Condition 2: point is to the left of intersection
    cond2 = px < x_intersect  # (N, M)
    
    # Combine conditions, masking out horizontal edges
    crossings = cond1 & cond2 & non_horizontal  # (N, M)
    
    # Count crossings for each point (odd = inside)
    crossing_count = np.sum(crossings, axis=1)
    
    return (crossing_count % 2) == 1


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """
    Determine if a single point is inside a polygon.
    
    Args:
        x: x coordinate of the point
        y: y coordinate of the point
        polygon: Mx2 array of (x, y) coordinates defining the polygon vertices
        
    Returns:
        True if point is inside the polygon, False otherwise
    """
    points = np.array([[x, y]])
    return points_in_polygon(points, polygon)[0]
