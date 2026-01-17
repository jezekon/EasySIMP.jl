"""
DensityFilter.jl - Density filtering for SIMP topology optimization

Uses KD-tree spatial indexing for O(n log n) neighbor search instead of O(n²).
Filter radius = filter_radius_ratio × characteristic_element_size

Recommended filter_radius_ratio (from literature):
- Tetrahedral meshes: 1.3 - 1.5
- Hexahedral meshes: 2.3 - 2.5
"""

using Ferrite
using LinearAlgebra
using NearestNeighbors
using StaticArrays

export apply_density_filter,
    apply_density_filter_uniform,
    apply_density_filter_adaptive,
    estimate_element_size,
    calculate_cell_centers,
    FilterCache,
    create_filter_cache,
    apply_density_filter_cached!

# =============================================================================
# FILTER CACHE STRUCTURE
# =============================================================================

"""
    FilterCache

Pre-computed data structure for efficient density filtering.
Stores KD-tree and neighbor lists to avoid repeated O(n²) searches.

# Fields
- `neighbor_lists`: Pre-computed neighbor indices for each cell
- `cell_centers`: Center coordinates of all cells
- `filter_radius`: Actual filter radius used
"""
struct FilterCache
    neighbor_lists::Vector{Vector{Int}}
    cell_centers::Vector{Vec{3,Float64}}
    filter_radius::Float64
end

"""
    create_filter_cache(grid, filter_radius_ratio)

Create a FilterCache with pre-computed neighbors for all cells.
Call ONCE before the optimization loop.

# Arguments
- `grid`: Ferrite Grid object
- `filter_radius_ratio`: Filter radius as multiple of element size

# Returns
- `FilterCache` with pre-computed neighbor lists
"""
function create_filter_cache(grid::Grid, filter_radius_ratio::Float64)
    n_cells = getncells(grid)
    
    # Calculate cell centers
    cell_centers = calculate_cell_centers(grid)
    
    # Determine filter radius
    char_size = estimate_element_size(grid)
    filter_radius = filter_radius_ratio * char_size
    
    # Build KD-tree from cell centers
    centers_matrix = zeros(3, n_cells)
    for i in 1:n_cells
        centers_matrix[1, i] = cell_centers[i][1]
        centers_matrix[2, i] = cell_centers[i][2]
        centers_matrix[3, i] = cell_centers[i][3]
    end
    kdtree = KDTree(centers_matrix)
    
    # Pre-compute neighbor lists for all cells
    neighbor_lists = Vector{Vector{Int}}(undef, n_cells)
    for i in 1:n_cells
        point = SVector{3,Float64}(cell_centers[i][1], cell_centers[i][2], cell_centers[i][3])
        neighbor_lists[i] = inrange(kdtree, point, filter_radius)
    end
    
    avg_neighbors = sum(length.(neighbor_lists)) / n_cells
    println("FilterCache created: $(n_cells) cells, r=$(round(filter_radius, digits=4)), avg_neighbors=$(round(avg_neighbors, digits=1))")
    
    return FilterCache(neighbor_lists, cell_centers, filter_radius)
end

# =============================================================================
# CACHED FILTER APPLICATION (USE IN OPTIMIZATION LOOP)
# =============================================================================

"""
    apply_density_filter_cached!(filtered_sens, cache, densities, sensitivities)

Apply Sigmund's sensitivity filter using pre-computed neighbors.
Zero allocations - use in optimization loops.

# Arguments
- `filtered_sens`: Output vector (modified in-place)
- `cache`: Pre-computed FilterCache
- `densities`: Current density distribution
- `sensitivities`: Raw sensitivities to filter

# Returns
- `filtered_sens` (modified in-place)
"""
function apply_density_filter_cached!(
    filtered_sens::Vector{Float64},
    cache::FilterCache,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
)
    n_cells = length(densities)
    cell_centers = cache.cell_centers
    filter_radius = cache.filter_radius
    
    @inbounds for i in 1:n_cells
        numerator = 0.0
        denominator = 0.0
        center_i = cell_centers[i]
        
        for j in cache.neighbor_lists[i]
            distance = norm(cell_centers[j] - center_i)
            weight = max(0.0, filter_radius - distance)
            
            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight * densities[j]
            end
        end
        
        filtered_sens[i] = denominator > 1e-12 ? numerator / denominator : sensitivities[i]
    end
    
    return filtered_sens
end

# =============================================================================
# STANDARD INTERFACE (for backward compatibility)
# =============================================================================

"""
    apply_density_filter(grid, densities, sensitivities, filter_radius_ratio)

Automatic density filter - chooses uniform or adaptive based on mesh uniformity.
For optimization loops, prefer create_filter_cache() + apply_density_filter_cached!()
"""
function apply_density_filter(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius_ratio::Float64,
)
    element_sizes = calculate_element_sizes(grid)
    size_variation = maximum(element_sizes) / minimum(element_sizes)

    if size_variation > 1.5
        return apply_density_filter_adaptive(grid, densities, sensitivities, filter_radius_ratio)
    else
        return apply_density_filter_uniform(grid, densities, sensitivities, filter_radius_ratio)
    end
end

"""
    apply_density_filter_uniform(grid, densities, sensitivities, filter_radius_ratio)

Sigmund's sensitivity filter for uniform meshes using KD-tree acceleration.
"""
function apply_density_filter_uniform(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius_ratio::Float64,
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)

    char_element_size = estimate_element_size(grid)
    filter_radius = filter_radius_ratio * char_element_size
    cell_centers = calculate_cell_centers(grid)
    
    # Build KD-tree
    centers_matrix = zeros(3, n_cells)
    for i in 1:n_cells
        centers_matrix[1, i] = cell_centers[i][1]
        centers_matrix[2, i] = cell_centers[i][2]
        centers_matrix[3, i] = cell_centers[i][3]
    end
    kdtree = KDTree(centers_matrix)

    for i in 1:n_cells
        numerator = 0.0
        denominator = 0.0
        
        # KD-tree range query: O(log n) instead of O(n)
        point = SVector{3,Float64}(cell_centers[i][1], cell_centers[i][2], cell_centers[i][3])
        neighbors = inrange(kdtree, point, filter_radius)

        for j in neighbors
            distance = norm(cell_centers[i] - cell_centers[j])
            weight = max(0.0, filter_radius - distance)

            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight * densities[j]
            end
        end

        filtered_sensitivities[i] = denominator > 1e-12 ? numerator / denominator : sensitivities[i]
    end

    return filtered_sensitivities
end

"""
    apply_density_filter_adaptive(grid, densities, sensitivities, filter_radius_ratio)

Adaptive filter for non-uniform meshes - local filter radius for each element.
"""
function apply_density_filter_adaptive(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius_ratio::Float64,
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)

    element_sizes = calculate_element_sizes(grid)
    cell_centers = calculate_cell_centers(grid)
    max_radius = filter_radius_ratio * maximum(element_sizes)
    
    # Build KD-tree
    centers_matrix = zeros(3, n_cells)
    for i in 1:n_cells
        centers_matrix[1, i] = cell_centers[i][1]
        centers_matrix[2, i] = cell_centers[i][2]
        centers_matrix[3, i] = cell_centers[i][3]
    end
    kdtree = KDTree(centers_matrix)

    for i in 1:n_cells
        local_radius = filter_radius_ratio * element_sizes[i]
        numerator = 0.0
        denominator = 0.0
        
        point = SVector{3,Float64}(cell_centers[i][1], cell_centers[i][2], cell_centers[i][3])
        neighbors = inrange(kdtree, point, local_radius)

        for j in neighbors
            distance = norm(cell_centers[i] - cell_centers[j])
            weight = max(0.0, local_radius - distance)

            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight * densities[j]
            end
        end

        filtered_sensitivities[i] = denominator > 1e-12 ? numerator / denominator : sensitivities[i]
    end

    return filtered_sensitivities
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    estimate_element_size(grid)

Estimate characteristic element size (average edge length).
"""
function estimate_element_size(grid::Grid)
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

Calculate characteristic size for each element.
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

Calculate characteristic size of a single element.
"""
function calculate_single_element_size(coords::Vector{Vec{3,Float64}})
    n_nodes = length(coords)

    if n_nodes == 4  # Tetrahedron
        return calculate_tet_size(coords)
    elseif n_nodes == 8  # Hexahedron
        return calculate_hex_size(coords)
    else
        total_length = 0.0
        n_edges = 0
        for i = 1:n_nodes, j = (i+1):n_nodes
            total_length += norm(coords[j] - coords[i])
            n_edges += 1
        end
        return n_edges > 0 ? total_length / n_edges : 1.0
    end
end

"""
    calculate_tet_size(coords)

Tetrahedral element size as average edge length.
"""
function calculate_tet_size(coords::Vector{Vec{3,Float64}})
    edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    total_length = sum(norm(coords[j] - coords[i]) for (i, j) in edges)
    return total_length / 6.0
end

"""
    calculate_hex_size(coords)

Hexahedral element size as geometric mean of three orthogonal edge lengths.
"""
function calculate_hex_size(coords::Vector{Vec{3,Float64}})
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
        cell_centers[cell_id] = sum(node_coords) / length(node_coords)
    end

    return cell_centers
end

"""
    print_filter_info(grid, filter_radius_ratio, filter_type="auto")

Print filter settings information.
"""
function print_filter_info(grid::Grid, filter_radius_ratio::Float64, filter_type::String = "auto")
    char_size = estimate_element_size(grid)
    element_sizes = calculate_element_sizes(grid)
    size_variation = maximum(element_sizes) / minimum(element_sizes)

    cell = getcells(grid, 1)
    cell_type = cell isa Ferrite.Tetrahedron ? "Tetrahedron" : "Hexahedron"

    println("Density filter information:")
    println("  Element type: $cell_type")
    println("  Characteristic element size: $(round(char_size, digits=4))")
    println("  Element size variation: $(round(size_variation, digits=2))")
    println("  Filter radius ratio: $filter_radius_ratio")
    println("  Actual filter radius: $(round(filter_radius_ratio * char_size, digits=4))")

    actual_type = filter_type == "auto" ? (size_variation > 1.5 ? "adaptive" : "uniform") : filter_type
    println("  Filter type: $actual_type")
end
