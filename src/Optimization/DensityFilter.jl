"""
DensityFilter.jl

Improved density filtering implementation that properly accounts for element sizes.
This addresses the issue where filter radius needs to be scaled by actual element dimensions.
"""

using Ferrite
using LinearAlgebra

export apply_density_filter, apply_adaptive_density_filter, should_use_adaptive_filter,
       estimate_characteristic_element_size, apply_density_filter_scaled

"""
    estimate_characteristic_element_size(grid)

Estimate characteristic element size for uniform or nearly-uniform meshes.
This is used to scale the filter radius appropriately.
"""
function estimate_characteristic_element_size(grid::Grid)
    # Sample a few elements to estimate characteristic size
    n_sample = min(100, getncells(grid))
    sample_indices = 1:div(getncells(grid), n_sample):getncells(grid)
    
    total_size = 0.0
    count = 0
    
    for cell_idx in sample_indices
        coords = getcoordinates(grid, cell_idx)
        
        if length(coords) == 4  # Tetrahedron
            size = calculate_tet_size(coords)
        elseif length(coords) == 8  # Hexahedron  
            size = calculate_hex_size(coords)
        else
            size = calculate_avg_edge_length(coords)
        end
        
        total_size += size
        count += 1
    end
    
    return count > 0 ? total_size / count : 1.0
end

"""
    apply_density_filter_scaled(grid, densities, sensitivities, filter_radius_ratio)

Apply density filter with proper scaling for element sizes.
filter_radius_ratio is given as multiple of characteristic element size (like in Sigmund's paper).

Parameters:
- `grid`: Computational mesh
- `densities`: Current density distribution  
- `sensitivities`: Sensitivity values
- `filter_radius_ratio`: Filter radius as multiple of element size (e.g., 1.5)

Returns:
- Filtered sensitivities
"""
function apply_density_filter_scaled(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius_ratio::Float64
)
    # Estimate characteristic element size
    char_element_size = estimate_characteristic_element_size(grid)
    
    # Calculate actual filter radius in mesh coordinates
    filter_radius = filter_radius_ratio * char_element_size
    
    println("Characteristic element size: $char_element_size")
    println("Filter radius ratio: $filter_radius_ratio")
    println("Actual filter radius: $filter_radius")
    
    # Check if mesh is uniform enough for basic filter
    if should_use_adaptive_filter(grid)
        println("Using adaptive filter due to non-uniform mesh")
        return apply_adaptive_density_filter(grid, densities, sensitivities, filter_radius)
    else
        println("Using basic filter with scaled radius")
        return apply_density_filter_basic(grid, densities, sensitivities, filter_radius)
    end
end

"""
    apply_density_filter_basic(grid, densities, sensitivities, filter_radius)

Basic density filter implementation (improved version of original).
filter_radius should be in same units as mesh coordinates.
"""
function apply_density_filter_basic(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius::Float64
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)
    
    # Get cell centers
    cell_centers = calculate_cell_centers(grid)
    
    # Apply filter with proper Sigmund formulation
    for i = 1:n_cells
        numerator = 0.0
        denominator = 0.0
        
        for j = 1:n_cells
            distance = norm(cell_centers[i] - cell_centers[j])
            
            # Sigmund's linear weight function: w = max(0, rmin - distance)
            weight = max(0.0, filter_radius - distance)
            
            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight * densities[j]  # Note: weight by density!
            end
        end
        
        if denominator > 1e-12
            filtered_sensitivities[i] = numerator / denominator
        else
            filtered_sensitivities[i] = sensitivities[i]
        end
    end
    
    return filtered_sensitivities
end

"""
    apply_density_filter(grid, densities, sensitivities, filter_radius)

Main density filter function - automatically selects appropriate method.
For backward compatibility, assumes filter_radius is in mesh coordinates.
"""
function apply_density_filter(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius::Float64
)
    # Check if mesh requires adaptive filtering
    if should_use_adaptive_filter(grid)
        return apply_adaptive_density_filter(grid, densities, sensitivities, filter_radius)
    else
        return apply_density_filter_basic(grid, densities, sensitivities, filter_radius)
    end
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
    
    println("Applying adaptive density filter")
    println("Element size range: [$(minimum(element_sizes)), $(maximum(element_sizes))]")
    println("Median element size: $median_size")
    
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
                denominator += weight * densities[j]
            end
        end
        
        if denominator > 1e-12
            filtered_sensitivities[i] = numerator / denominator
        else
            filtered_sensitivities[i] = sensitivities[i]
        end
    end
    
    return filtered_sensitivities
end

"""
    should_use_adaptive_filter(grid)

Determine if adaptive filtering is needed based on mesh uniformity.
Returns true if element size variation > 2.0
"""
function should_use_adaptive_filter(grid::Grid)
    element_sizes = calculate_element_sizes(grid)
    min_size = minimum(element_sizes)
    max_size = maximum(element_sizes)
    size_ratio = max_size / min_size
    
    # Use adaptive filter if size variation is significant
    use_adaptive = size_ratio > 2.0
    
    if use_adaptive
        println("Mesh size ratio: $size_ratio > 2.0, using adaptive filter")
    else
        println("Mesh size ratio: $size_ratio ≤ 2.0, using basic filter")
    end
    
    return use_adaptive
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

"""
    validate_filter_parameters(grid, filter_radius_ratio)

Validate and provide recommendations for filter parameters.
"""
function validate_filter_parameters(grid::Grid, filter_radius_ratio::Float64)
    char_size = estimate_characteristic_element_size(grid)
    actual_radius = filter_radius_ratio * char_size
    
    println("Filter parameter validation:")
    println("  Characteristic element size: $char_size")
    println("  Filter radius ratio: $filter_radius_ratio")
    println("  Actual filter radius: $actual_radius")
    
    if filter_radius_ratio < 1.0
        @warn "Filter radius ratio < 1.0 may not provide sufficient mesh independence"
    elseif filter_radius_ratio > 3.0
        @warn "Filter radius ratio > 3.0 may over-smooth the design"
    end
    
    # Estimate how many elements are in filter radius
    element_sizes = calculate_element_sizes(grid)
    avg_size = sum(element_sizes) / length(element_sizes)
    elements_in_radius = (actual_radius / avg_size)^3  # Rough estimate for 3D
    
    println("  Estimated elements in filter radius: $(round(Int, elements_in_radius))")
    
    if elements_in_radius < 8
        @warn "Filter radius may be too small - fewer than 8 elements in filter region"
    elseif elements_in_radius > 100
        @warn "Filter radius may be too large - more than 100 elements in filter region"
    end
end
