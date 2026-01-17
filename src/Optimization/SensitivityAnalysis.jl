"""
SensitivityAnalysis.jl

Implementation of sensitivity analysis for SIMP topology optimization.
Calculates derivatives of objective function with respect to design variables.
"""

using Ferrite
using LinearAlgebra
using ..FiniteElementAnalysis: compute_lame_parameters

export calculate_sensitivities, calculate_sensitivities!, calculate_compliance_sensitivity

"""
    calculate_sensitivities!(sensitivities, grid, dh, cellvalues, densities, u, E0, Emin, ν, p)

Calculate sensitivities in-place without allocations (use in optimization loop).

# Arguments
- `sensitivities`: Output vector (modified in-place)
- `grid`: Ferrite Grid object
- `dh`: DofHandler  
- `cellvalues`: CellValues for integration
- `densities`: Current density distribution
- `u`: Displacement vector
- `E0`: Base Young's modulus
- `Emin`: Minimum Young's modulus
- `ν`: Poisson's ratio
- `p`: Penalization power
"""
function calculate_sensitivities!(
    sensitivities::Vector{Float64},
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    densities::Vector{Float64},
    u::Vector{Float64},
    E0::Float64,
    Emin::Float64,
    ν::Float64,
    p::Float64,
)
    # Lamé parameters for unit Young's modulus
    λ0 = ν / ((1 + ν) * (1 - 2ν))
    μ0 = 1.0 / (2 * (1 + ν))
    
    n_basefuncs = getnbasefunctions(cellvalues)
    ke_unit = zeros(n_basefuncs, n_basefuncs)

    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        density = densities[cell_id]
        cell_dofs = celldofs(cell)
        
        reinit!(cellvalues, cell)
        fill!(ke_unit, 0.0)
        
        # Compute unit stiffness matrix
        for q_point = 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i = 1:n_basefuncs
                ∇Ni = shape_gradient(cellvalues, q_point, i)
                εi = symmetric(∇Ni)
                for j = 1:n_basefuncs
                    ∇Nj = shape_gradient(cellvalues, q_point, j)
                    εj = symmetric(∇Nj)
                    σ = λ0 * tr(εj) * one(εj) + 2μ0 * εj
                    ke_unit[i, j] += (εi ⊡ σ) * dΩ
                end
            end
        end

        # Derivative of Young's modulus w.r.t. density
        dE_drho = p * density^(p-1) * (E0 - Emin)

        # Compliance sensitivity: ∂c/∂ρ = -∂E/∂ρ * u^T * k_unit * u
        u_elem = @view u[cell_dofs]
        sensitivities[cell_id] = -dE_drho * dot(u_elem, ke_unit * u_elem)
    end
end

"""
    calculate_sensitivities(grid, dh, cellvalues, densities, u, E0, Emin, ν, p)

Calculate sensitivities (allocating version for backward compatibility).

# Arguments
- `grid`: Ferrite Grid object
- `dh`: DofHandler  
- `cellvalues`: CellValues for integration
- `densities`: Current density distribution
- `u`: Displacement vector
- `E0`: Base Young's modulus
- `Emin`: Minimum Young's modulus
- `ν`: Poisson's ratio
- `p`: Penalization power

# Returns
- Vector of sensitivities ∂c/∂ρe for each element

# Theory
For compliance minimization:
∂c/∂ρe = -p * ρe^(p-1) * (E0 - Emin) * ue^T * k0 * ue

where:
- p is the penalization power
- ρe is the element density  
- ue is the element displacement vector
- k0 is the element stiffness matrix for unit Young's modulus
"""
function calculate_sensitivities(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    densities::Vector{Float64},
    u::Vector{Float64},
    E0::Float64,
    Emin::Float64,
    ν::Float64,
    p::Float64,
)
    n_cells = getncells(grid)
    sensitivities = zeros(n_cells)
    calculate_sensitivities!(sensitivities, grid, dh, cellvalues, densities, u, E0, Emin, ν, p)
    return sensitivities
end

"""
    calculate_compliance_sensitivity(cell, cellvalues, density, u_element, E0, Emin, ν, p)

Calculate compliance sensitivity for a single element using SIMP parameters.

# Arguments
- `cell`: Current cell iterator
- `cellvalues`: CellValues for integration
- `density`: Element density
- `u_element`: Element displacement vector
- `E0`: Base Young's modulus
- `Emin`: Minimum Young's modulus  
- `ν`: Poisson's ratio
- `p`: Penalization power

# Returns
- Compliance sensitivity ∂c/∂ρe for this element
"""
function calculate_compliance_sensitivity(
    cell,
    cellvalues,
    density::Float64,
    u_element::Vector{Float64},
    E0::Float64,
    Emin::Float64,
    ν::Float64,
    p::Float64,
)
    reinit!(cellvalues, cell)

    # Calculate Lamé parameters for unit Young's modulus
    λ0, μ0 = compute_lame_parameters(1.0, ν)

    # Calculate element stiffness matrix for unit Young's modulus
    n_basefuncs = getnbasefunctions(cellvalues)
    ke_unit = zeros(n_basefuncs, n_basefuncs)

    for q_point = 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)

        for i = 1:n_basefuncs
            ∇Ni = shape_gradient(cellvalues, q_point, i)
            εi = symmetric(∇Ni)

            for j = 1:n_basefuncs
                ∇Nj = shape_gradient(cellvalues, q_point, j)
                εj = symmetric(∇Nj)
                σ = λ0 * tr(εj) * one(εj) + 2μ0 * εj
                ke_unit[i, j] += (εi ⊡ σ) * dΩ
            end
        end
    end

    # Derivative of Young's modulus w.r.t. density
    dE_drho = p * density^(p-1) * (E0 - Emin)

    # Compliance sensitivity: ∂c/∂ρ = -∂E/∂ρ * u^T * k_unit * u
    sensitivity = -dE_drho * dot(u_element, ke_unit * u_element)

    return sensitivity
end
