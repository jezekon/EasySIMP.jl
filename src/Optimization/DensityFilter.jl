"""
DensityFilter.jl - Simplified density filtering for SIMP topology optimization

Three approaches:
1. Automatic: automatically choose between uniform/adaptive based on mesh
2. Uniform mesh: constant filter radius scaled by uniform element size
3. Adaptive mesh: locally varying filter radius based on local element size
"""

using Ferrite
using LinearAlgebra

export apply_density_filter,
    apply_density_filter_uniform,
    apply_density_filter_adaptive,
    estimate_element_size,
    calculate_cell_centers

"""
    apply_density_filter(grid, densities, sensitivities, filter_radius_ratio)

Automatic density filter that chooses uniform or adaptive based on mesh uniformity.
Recommended for most users.

# Arguments
- `grid`: Computational mesh
- `densities`: Current density distribution  
- `sensitivities`: Sensitivity values
- `filter_radius_ratio`: Filter radius as multiple of element size (like Sigmund's rmin)

# Returns
- Filtered sensitivities

# Example
```julia
# Automatic choice: filter_radius = 1.5 × element_size
filtered_sens = apply_density_filter(grid, densities, sensitivities, 1.5)
```
"""
function apply_density_filter(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius_ratio::Float64,
)
    # Check mesh uniformity
    element_sizes = calculate_element_sizes(grid)
    size_variation = maximum(element_sizes) / minimum(element_sizes)

    if size_variation > 1.5
        # Non-uniform mesh: use adaptive filter
        return apply_density_filter_adaptive(
            grid,
            densities,
            sensitivities,
            filter_radius_ratio,
        )
    else
        # Uniform mesh: use uniform filter
        return apply_density_filter_uniform(
            grid,
            densities,
            sensitivities,
            filter_radius_ratio,
        )
    end
end

"""
    apply_density_filter_uniform(grid, densities, sensitivities, filter_radius_ratio)

Density filter for uniform meshes where all elements have similar size.
Filter radius = filter_radius_ratio × characteristic_element_size

# Arguments
- `grid`: Computational mesh
- `densities`: Current density distribution  
- `sensitivities`: Sensitivity values
- `filter_radius_ratio`: Filter radius as multiple of characteristic element size

# Returns
- Filtered sensitivities

# Example
```julia
# For uniform mesh: filter_radius = 1.5 × characteristic_element_size
filtered_sens = apply_density_filter_uniform(grid, densities, sensitivities, 1.5)
```
"""
function apply_density_filter_uniform(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius_ratio::Float64,
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)

    # Calculate characteristic element size and actual filter radius
    char_element_size = estimate_element_size(grid)
    filter_radius = filter_radius_ratio * char_element_size

    # Get cell centers once
    cell_centers = calculate_cell_centers(grid)

    # Apply Sigmund's filter formula
    for i = 1:n_cells
        numerator = 0.0
        denominator = 0.0

        for j = 1:n_cells
            distance = norm(cell_centers[i] - cell_centers[j])

            # Sigmund's linear weight: w = max(0, rmin - distance)
            weight = max(0.0, filter_radius - distance)

            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight * densities[j]
            end
        end

        filtered_sensitivities[i] =
            denominator > 1e-12 ? numerator / denominator : sensitivities[i]
    end

    return filtered_sensitivities
end

"""
    apply_density_filter_adaptive(grid, densities, sensitivities, filter_radius_ratio)

Adaptive density filter for non-uniform meshes.
Each element uses filter radius = filter_radius_ratio × local_element_size

# Arguments
- `grid`: Computational mesh
- `densities`: Current density distribution  
- `sensitivities`: Sensitivity values
- `filter_radius_ratio`: Filter radius as multiple of local element size

# Returns
- Filtered sensitivities

# Example
```julia
# For non-uniform mesh: filter_radius = 1.5 × local_element_size for each element
filtered_sens = apply_density_filter_adaptive(grid, densitivities, sensitivities, 1.5)
```
"""
function apply_density_filter_adaptive(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius_ratio::Float64,
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)

    # Calculate element sizes and cell centers
    element_sizes = calculate_element_sizes(grid)
    cell_centers = calculate_cell_centers(grid)

    # Apply adaptive filter
    for i = 1:n_cells
        # Local filter radius = filter_radius_ratio × element_size[i]
        local_radius = filter_radius_ratio * element_sizes[i]

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

        filtered_sensitivities[i] =
            denominator > 1e-12 ? numerator / denominator : sensitivities[i]
    end

    return filtered_sensitivities
end

"""
    estimate_element_size(grid)

Estimate characteristic element size for the entire mesh.
Useful for determining appropriate filter radius.
"""
function estimate_element_size(grid::Grid)
    # Sample first few elements
    n_sample = min(10, getncells(grid))
    total_size = 0.0

    for cell_idx = 1:n_sample
        coords = getcoordinates(grid, cell_idx)
        total_size += calculate_single_element_size(coords)
    end

    return total_size / n_sample
end

"""
    calculate_element_sizes(grid)

Calculate characteristic size for each element in the grid.
Used by adaptive filter and mesh uniformity check.
"""
function calculate_element_sizes(grid::Grid)
    n_cells = getncells(grid)
    element_sizes = zeros(n_cells)

    for cell_idx = 1:n_cells
        coords = getcoordinates(grid, cell_idx)
        element_sizes[cell_idx] = calculate_single_element_size(coords)
    end

    return element_sizes
end

"""
    calculate_single_element_size(coords)

Calculate characteristic size of a single element from its coordinates.
"""
function calculate_single_element_size(coords::Vector{Vec{3,Float64}})
    n_nodes = length(coords)

    if n_nodes == 4  # Tetrahedron
        return calculate_tet_size(coords)
    elseif n_nodes == 8  # Hexahedron  
        return calculate_hex_size(coords)
    else
        # Generic: average edge length
        total_length = 0.0
        n_edges = 0

        for i = 1:n_nodes
            for j = (i+1):n_nodes
                total_length += norm(coords[j] - coords[i])
                n_edges += 1
            end
        end

        return n_edges > 0 ? total_length / n_edges : 1.0
    end
end

"""
    calculate_tet_size(coords)

Tetrahedral element size as cube root of volume.
"""
function calculate_tet_size(coords::Vector{Vec{3,Float64}})
    p0, p1, p2, p3 = coords[1], coords[2], coords[3], coords[4]
    # Volume = |det(B)| / 6 where B = [p1-p0, p2-p0, p3-p0]
    B = hcat(p1 - p0, p2 - p0, p3 - p0)
    volume = abs(det(B)) / 6.0
    return volume^(1/3)  # Characteristic length
end

"""
    calculate_hex_size(coords)

Hexahedral element size as geometric mean of three edge lengths.
"""
function calculate_hex_size(coords::Vector{Vec{3,Float64}})
    # Take three orthogonal edges from first node
    edge1 = norm(coords[2] - coords[1])
    edge2 = norm(coords[4] - coords[1])
    edge3 = norm(coords[5] - coords[1])
    return (edge1 * edge2 * edge3)^(1/3)
end

"""
    calculate_cell_centers(grid)

Calculate center coordinates of all cells.
"""
function calculate_cell_centers(grid::Grid)
    n_cells = getncells(grid)
    cell_centers = Vector{Vec{3,Float64}}(undef, n_cells)

    for (cell_id, cell) in enumerate(getcells(grid))
        node_coords = [grid.nodes[node_id].x for node_id in cell.nodes]
        center = sum(node_coords) / length(node_coords)
        cell_centers[cell_id] = center
    end

    return cell_centers
end

"""
    print_filter_info(grid, filter_radius_ratio, filter_type="auto")

Print information about filter settings for debugging.
"""
function print_filter_info(
    grid::Grid,
    filter_radius_ratio::Float64,
    filter_type::String = "auto",
)
    char_size = estimate_element_size(grid)
    element_sizes = calculate_element_sizes(grid)
    size_variation = maximum(element_sizes) / minimum(element_sizes)

    println("Density filter information:")
    println("  Characteristic element size: $(round(char_size, digits=3))")
    println("  Element size variation: $(round(size_variation, digits=2))")
    println("  Filter radius ratio: $filter_radius_ratio")

    if filter_type == "auto"
        actual_type = size_variation > 2.0 ? "adaptive" : "uniform"
        println("  Filter type: $actual_type (automatically chosen)")
    else
        println("  Filter type: $filter_type (manually specified)")
    end

    if filter_type == "uniform" || (filter_type == "auto" && size_variation <= 2.0)
        actual_radius = filter_radius_ratio * char_size
        println("  Actual filter radius: $(round(actual_radius, digits=3))")
    else
        min_radius = filter_radius_ratio * minimum(element_sizes)
        max_radius = filter_radius_ratio * maximum(element_sizes)
        println(
            "  Filter radius range: $(round(min_radius, digits=3)) - $(round(max_radius, digits=3))",
        )
    end
end
