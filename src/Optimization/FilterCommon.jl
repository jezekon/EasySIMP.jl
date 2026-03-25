"""
FilterCommon.jl - Shared filter infrastructure for SIMP topology optimization

Provides FilterCache, KD-tree construction, and geometry helpers used by both
sensitivity filtering and density filtering.

Filter radius = filter_radius_ratio × characteristic_element_size

Recommended filter_radius_ratio (from literature):
- Tetrahedral meshes: 1.3 - 1.5
- Hexahedral meshes: 2.3 - 2.5
"""

using Ferrite
using LinearAlgebra
using NearestNeighbors
using StaticArrays

export estimate_element_size,
    calculate_cell_centers,
    FilterCache,
    create_filter_cache

# =============================================================================
# FILTER CACHE STRUCTURE
# =============================================================================

"""
    FilterCache

Pre-computed data structure for efficient filtering.
Stores KD-tree and neighbor lists to avoid repeated O(n²) searches.

# Fields
- `neighbor_lists`: Pre-computed neighbor indices for each cell
- `cell_centers`: Center coordinates of all cells
- `filter_radius`: Actual filter radius used
- `element_volumes`: Volume of each element
"""
struct FilterCache
    neighbor_lists::Vector{Vector{Int}}
    cell_centers::Vector{Vec{3,Float64}}
    filter_radius::Float64
    element_volumes::Vector{Float64}
end

"""
    create_filter_cache(grid, filter_radius_ratio, element_volumes)

Create a FilterCache with pre-computed neighbors for all cells.
Call ONCE before the optimization loop.

# Arguments
- `grid`: Ferrite Grid object
- `filter_radius_ratio`: Filter radius as multiple of element size
- `element_volumes`: Volume of each element

# Returns
- `FilterCache` with pre-computed neighbor lists
"""
function create_filter_cache(
    grid::Grid,
    filter_radius_ratio::Float64,
    element_volumes::Vector{Float64},
)
    n_cells = getncells(grid)

    # Calculate cell centers
    cell_centers = calculate_cell_centers(grid)

    # Determine filter radius
    char_size = estimate_element_size(grid)
    filter_radius = filter_radius_ratio * char_size

    # Build KD-tree from cell centers
    centers_matrix = zeros(3, n_cells)
    for i = 1:n_cells
        centers_matrix[1, i] = cell_centers[i][1]
        centers_matrix[2, i] = cell_centers[i][2]
        centers_matrix[3, i] = cell_centers[i][3]
    end
    kdtree = KDTree(centers_matrix)

    # Pre-compute neighbor lists for all cells
    neighbor_lists = Vector{Vector{Int}}(undef, n_cells)
    for i = 1:n_cells
        point =
            SVector{3,Float64}(cell_centers[i][1], cell_centers[i][2], cell_centers[i][3])
        neighbor_lists[i] = inrange(kdtree, point, filter_radius)
    end

    avg_neighbors = sum(length.(neighbor_lists)) / n_cells
    println(
        "FilterCache created: $(n_cells) cells, r=$(round(filter_radius, digits=4)), avg_neighbors=$(round(avg_neighbors, digits=1))",
    )

    return FilterCache(neighbor_lists, cell_centers, filter_radius, element_volumes)
end

# =============================================================================
# GEOMETRY HELPERS
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
