"""
DensityFilter.jl

Implementation of density filtering for mesh-independent topology optimization.
Enhanced with adaptive filtering for unstructured meshes.
"""

using Ferrite
using LinearAlgebra

export apply_density_filter, apply_adaptive_density_filter, should_use_adaptive_filter

"""
    apply_density_filter(grid, densities, sensitivities, filter_radius)

Original density filter for uniform meshes (Sigmund filter).
"""
function apply_density_filter(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius::Float64
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)
    
    # Get cell centers
    cell_centers = calculate_cell_centers(grid)
    
    # Apply filter
    for i = 1:n_cells
        numerator = 0.0
        denominator = 0.0
        
        for j = 1:n_cells
            distance = norm(cell_centers[i] - cell_centers[j])
            weight = max(0.0, filter_radius - distance)
            
            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight
            end
        end
        
        if denominator > 0.0 && densities[i] > 1e-6
            filtered_sensitivities[i] = numerator / (densities[i] * denominator)
        else
            filtered_sensitivities[i] = sensitivities[i]
        end
    end
    
    return filtered_sensitivities
end

"""
    apply_adaptive_density_filter(grid, densities, sensitivities, base_filter_radius)

Adaptive density filter for non-uniform meshes.
Each element uses radius scaled by its local size relative to median.
"""
function apply_adaptive_density_filter(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    base_filter_radius::Float64
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)
    
    # Calculate element sizes and median
    element_sizes = calculate_element_sizes(grid)
    median_size = estimate_median_size(element_sizes)
    
    # Get cell centers
    cell_centers = calculate_cell_centers(grid)
    
    # Apply adaptive filter
    for i = 1:n_cells
        # Scale radius by element size
        local_radius = base_filter_radius * (element_sizes[i] / median_size)
        
        numerator = 0.0
        denominator = 0.0
        
        for j = 1:n_cells
            distance = norm(cell_centers[i] - cell_centers[j])
            weight = max(0.0, local_radius - distance)
            
            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight
            end
        end
        
        if denominator > 0.0 && densities[i] > 1e-6
            filtered_sensitivities[i] = numerator / (densities[i] * denominator)
        else
            filtered_sensitivities[i] = sensitivities[i]
        end
    end
    
    return filtered_sensitivities
end

"""
    should_use_adaptive_filter(grid)

Simple criterion to decide which filter to use.
Returns true if element size variation > 3.0
"""
function should_use_adaptive_filter(grid::Grid)
    element_sizes = calculate_element_sizes(grid)
    min_size = minimum(element_sizes)
    max_size = maximum(element_sizes)
    size_ratio = max_size / min_size
    
    return size_ratio > 3.0
end

"""
    calculate_element_sizes(grid)

Calculate characteristic size for each element.
"""
function calculate_element_sizes(grid::Grid)
    n_cells = getncells(grid)
    element_sizes = zeros(n_cells)
    
    for cell_idx in 1:n_cells
        coords = getcoordinates(grid, cell_idx)
        
        if length(coords) == 4  # Tetrahedron
            element_sizes[cell_idx] = calculate_tet_size(coords)
        elseif length(coords) == 8  # Hexahedron  
            element_sizes[cell_idx] = calculate_hex_size(coords)
        else
            element_sizes[cell_idx] = calculate_avg_edge_length(coords)
        end
    end
    
    return element_sizes
end

"""
    calculate_tet_size(coords)

Tetrahedral element size as cube root of volume.
"""
function calculate_tet_size(coords::Vector{Vec{3, Float64}})
    p0, p1, p2, p3 = coords[1], coords[2], coords[3], coords[4]
    B = hcat(p1 - p0, p2 - p0, p3 - p0)
    volume = abs(det(B)) / 6.0
    return volume^(1/3)
end

"""
    calculate_hex_size(coords)

Hexahedral element size as geometric mean of edges.
"""
function calculate_hex_size(coords::Vector{Vec{3, Float64}})
    edge1 = norm(coords[2] - coords[1])
    edge2 = norm(coords[4] - coords[1]) 
    edge3 = norm(coords[5] - coords[1])
    return (edge1 * edge2 * edge3)^(1/3)
end

"""
    calculate_avg_edge_length(coords)

Average edge length for any element type.
"""
function calculate_avg_edge_length(coords::Vector{Vec{3, Float64}})
    n_nodes = length(coords)
    total_length = 0.0
    n_edges = 0
    
    for i in 1:n_nodes
        for j in (i+1):n_nodes
            total_length += norm(coords[j] - coords[i])
            n_edges += 1
        end
    end
    
    return total_length / n_edges
end

"""
    estimate_median_size(element_sizes)

Calculate median element size.
"""
function estimate_median_size(element_sizes::Vector{Float64})
    sorted_sizes = sort(element_sizes)
    n = length(sorted_sizes)
    
    if n % 2 == 1
        return sorted_sizes[div(n+1, 2)]
    else
        return (sorted_sizes[div(n, 2)] + sorted_sizes[div(n, 2) + 1]) / 2
    end
end

"""
    calculate_cell_centers(grid)

Calculate center coordinates of all cells.
"""
function calculate_cell_centers(grid::Grid)
    n_cells = getncells(grid)
    cell_centers = Vector{Vec{3, Float64}}(undef, n_cells)
    
    for (cell_id, cell) in enumerate(getcells(grid))
        node_coords = [grid.nodes[node_id].x for node_id in cell.nodes]
        center = sum(node_coords) / length(node_coords)
        cell_centers[cell_id] = center
    end
    
    return cell_centers
end
