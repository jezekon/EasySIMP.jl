module Utils

export calculate_volume

include("TerminalStyle.jl")
export print_error, print_warning, print_info, print_success, print_data

using Ferrite
using LinearAlgebra: dot

"""
    calculate_volume(element_volumes::Vector{Float64}, densities::Vector{Float64})

Calculate weighted volume as dot(element_volumes, densities).
Fast path for use inside the optimization loop where element_volumes are pre-computed.
"""
function calculate_volume(element_volumes::Vector{Float64}, densities::Vector{Float64})
    return dot(element_volumes, densities)
end

"""
    calculate_volume(element_volumes::Vector{Float64})

Calculate total volume as sum of pre-computed element volumes.
"""
function calculate_volume(element_volumes::Vector{Float64})
    return sum(element_volumes)
end

"""
    calculate_volume(grid::Ferrite.Grid, density_data::Union{Vector{Float64}, Nothing}=nothing)

Calculates the total weighted volume of a mesh with density distribution from SIMP results.
Works with both tetrahedral and hexahedral (8-node) elements.

Parameters:
- `grid`: Ferrite Grid object representing the mesh
- `density_data`: Optional vector containing density values for each cell. 
                 If not provided, a uniform density of 1.0 is assumed for all elements.

Returns:
- The total weighted volume of the mesh (∑ element_volume * density) in cubic units
"""
function calculate_volume(
    grid::Ferrite.Grid,
    density_data::Union{Vector{Float64},Nothing} = nothing,
)
    # Initialize total volume
    total_volume = 0.0

    # Create a default density array of 1.0 if no density data provided
    num_cells = getncells(grid)
    actual_density_data = if density_data === nothing
        # println("No density data provided, assuming uniform density of 1.0")
        ones(Float64, num_cells)
    else
        # Check if provided density data length matches the number of cells
        if length(density_data) != num_cells
            error(
                "Density data length ($(length(density_data))) does not match number of cells ($num_cells)",
            )
        end
        density_data
    end

    # Determine the cell type from the grid
    cell_type = typeof(getcells(grid, 1))

    # Choose appropriate reference shape based on cell type
    if cell_type <: Ferrite.Hexahedron
        # For 8-node hexahedral elements
        ip = Lagrange{RefHexahedron,1}()
        qr = QuadratureRule{RefHexahedron}(2)  # Higher-order quadrature for accuracy
    else
        # Default to tetrahedron
        ip = Lagrange{RefTetrahedron,1}()
        qr = QuadratureRule{RefTetrahedron}(2)
    end

    # Create cell values for integration
    cellvalues = CellValues(qr, ip)

    # Create DofHandler to iterate over cells
    dh = DofHandler(grid)
    add!(dh, :u, ip)  # Add a scalar field for volume calculation
    close!(dh)

    # Iterate over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)

        # Get cell ID to find its density
        cell_id = cellid(cell)
        density = actual_density_data[cell_id]

        # For each quadrature point, add the integration weight times Jacobian determinant
        cell_volume = 0.0
        for q_point = 1:getnquadpoints(cellvalues)
            # getdetJdV provides the volume element at the quadrature point
            dΩ = getdetJdV(cellvalues, q_point)
            cell_volume += dΩ
        end

        # Multiply cell volume by its density and add to total
        total_volume += cell_volume * density
    end

    # @info "Total weighted mesh volume: $total_volume cubic units"
    return total_volume
end


end
