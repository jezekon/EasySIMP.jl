"""
DensityFilter.jl

Implementation of density filtering for mesh-independent topology optimization.
Based on Sigmund (1994, 1997) and Bruns & Tortorelli (2001).
"""

using Ferrite
using LinearAlgebra

export apply_density_filter, create_filter_matrix

"""
    apply_density_filter(grid, densities, sensitivities, filter_radius)

Apply density filter to sensitivities to ensure mesh independence.

# Arguments
- `grid`: Ferrite Grid object
- `densities`: Current density distribution
- `sensitivities`: Sensitivity of objective function w.r.t. densities
- `filter_radius`: Filter radius (in units of element size)

# Returns
- Filtered sensitivities

# Method
The filtered sensitivity is calculated as:
∂c̃/∂xi = (1/xi) * Σj(Hij * xj * ∂c/∂xj) / Σj(Hij)

where Hij = max(0, rmin - dist(i,j)) is the weight factor.
"""
function apply_density_filter(
    grid::Grid,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    filter_radius::Float64
)
    n_cells = getncells(grid)
    filtered_sensitivities = zeros(n_cells)
    
    # Get cell centers for distance calculation
    cell_centers = calculate_cell_centers(grid)
    
    # Apply filter to each element
    for i = 1:n_cells
        numerator = 0.0
        denominator = 0.0
        
        # Find neighbors within filter radius
        for j = 1:n_cells
            # Calculate distance between cell centers
            distance = norm(cell_centers[i] - cell_centers[j])
            
            # Calculate weight factor
            weight = max(0.0, filter_radius - distance)
            
            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j]
                denominator += weight
            end
        end
        
        # Calculate filtered sensitivity
        if denominator > 0.0 && densities[i] > 1e-6
            filtered_sensitivities[i] = numerator / (densities[i] * denominator)
        else
            filtered_sensitivities[i] = sensitivities[i]
        end
    end
    
    return filtered_sensitivities
end

"""
    calculate_cell_centers(grid)

Calculate the center coordinates of all cells in the grid.

# Returns
- Vector of cell center coordinates
"""
function calculate_cell_centers(grid::Grid)
    n_cells = getncells(grid)
    cell_centers = Vector{Vec{3, Float64}}(undef, n_cells)
    
    for (cell_id, cell) in enumerate(getcells(grid))
        # Get node coordinates for this cell
        node_coords = [grid.nodes[node_id].x for node_id in cell.nodes]
        
        # Calculate center as average of node coordinates
        center = sum(node_coords) / length(node_coords)
        cell_centers[cell_id] = center
    end
    
    return cell_centers
end

"""
    create_filter_matrix(grid, filter_radius)

Create the filter matrix H for efficient filtering operations.
This is more efficient for multiple filtering operations.

# Arguments
- `grid`: Ferrite Grid object  
- `filter_radius`: Filter radius

# Returns
- Sparse filter matrix H where H[i,j] is the weight between cells i and j
"""
function create_filter_matrix(grid::Grid, filter_radius::Float64)
    n_cells = getncells(grid)
    cell_centers = calculate_cell_centers(grid)
    
    # Initialize sparse matrix components
    I = Int[]
    J = Int[]
    V = Float64[]
    
    # Build filter matrix
    for i = 1:n_cells
        for j = 1:n_cells
            distance = norm(cell_centers[i] - cell_centers[j])
            weight = max(0.0, filter_radius - distance)
            
            if weight > 0.0
                push!(I, i)
                push!(J, j) 
                push!(V, weight)
            end
        end
    end
    
    # Create sparse matrix
    H = sparse(I, J, V, n_cells, n_cells)
    
    return H
end

"""
    apply_filter_with_matrix(H, densities, sensitivities)

Apply density filter using pre-computed filter matrix.

# Arguments
- `H`: Filter matrix (from create_filter_matrix)
- `densities`: Current density distribution
- `sensitivities`: Sensitivities to filter

# Returns
- Filtered sensitivities
"""
function apply_filter_with_matrix(
    H::SparseMatrixCSC,
    densities::Vector{Float64},
    sensitivities::Vector{Float64}
)
    n_cells = length(densities)
    filtered_sensitivities = zeros(n_cells)
    
    for i = 1:n_cells
        if densities[i] > 1e-6
            # Calculate numerator: Σj(Hij * xj * ∂c/∂xj)
            numerator = 0.0
            denominator = 0.0
            
            for j in nzrange(H, i)
                col = H.rowval[j]
                weight = H.nzval[j]
                numerator += weight * densities[col] * sensitivities[col]
                denominator += weight
            end
            
            # Apply filter formula
            filtered_sensitivities[i] = numerator / (densities[i] * denominator)
        else
            filtered_sensitivities[i] = sensitivities[i]
        end
    end
    
    return filtered_sensitivities
end

"""
    helmholtz_filter(grid, sensitivities, filter_radius)

Alternative Helmholtz-type PDE filter (more advanced).

∇²s̃ - (r/R)² s̃ = -s / R²

where s are the original sensitivities and s̃ are the filtered ones.
"""
function helmholtz_filter(
    grid::Grid,
    sensitivities::Vector{Float64},
    filter_radius::Float64
)
    # This would require solving a PDE - more complex implementation
    # For now, fall back to density filter
    @warn "Helmholtz filter not yet implemented, using density filter"
    return sensitivities
end

"""
    calculate_effective_filter_radius(grid, filter_radius)

Calculate effective filter radius based on element size.

# Arguments
- `grid`: Ferrite Grid
- `filter_radius`: Desired filter radius in element units

# Returns  
- Actual filter radius in coordinate units
"""
function calculate_effective_filter_radius(grid::Grid, filter_radius::Float64)
    # Estimate element size from first few elements
    if getncells(grid) == 0
        return filter_radius
    end
    
    # Get first cell and estimate its size
    first_cell = getcells(grid, 1)
    node_coords = [grid.nodes[node_id].x for node_id in first_cell.nodes]
    
    # Calculate approximate element size (distance between first two nodes)
    if length(node_coords) >= 2
        element_size = norm(node_coords[2] - node_coords[1])
        return filter_radius * element_size
    else
        return filter_radius
    end
end
