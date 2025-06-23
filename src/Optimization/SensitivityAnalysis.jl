"""
SensitivityAnalysis.jl

Implementation of sensitivity analysis for SIMP topology optimization.
Calculates derivatives of objective function with respect to design variables.
"""

using Ferrite
using LinearAlgebra
using ..FiniteElementAnalysis

export calculate_sensitivities, calculate_compliance_sensitivity

"""
    calculate_sensitivities(grid, dh, cellvalues, material_model, densities, u)

Calculate sensitivities of the objective function (compliance) with respect to element densities.

# Arguments
- `grid`: Ferrite Grid object
- `dh`: DofHandler  
- `cellvalues`: CellValues for integration
- `material_model`: SIMP material model function
- `densities`: Current density distribution
- `u`: Displacement vector

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
    material_model,
    densities::Vector{Float64},
    u::Vector{Float64}
)
    n_cells = getncells(grid)
    sensitivities = zeros(n_cells)
    
    # Iterate over all cells
    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        density = densities[cell_id]
        
        # Get element displacement vector
        cell_dofs = celldofs(cell)
        u_element = u[cell_dofs]
        
        # Calculate compliance sensitivity for this element
        sensitivities[cell_id] = calculate_compliance_sensitivity(
            cell, cellvalues, material_model, density, u_element
        )
    end
    
    return sensitivities
end

"""
    calculate_compliance_sensitivity(cell, cellvalues, material_model, density, u_element)

Calculate compliance sensitivity for a single element.
"""
function calculate_compliance_sensitivity(
    cell,
    cellvalues,
    material_model,
    density::Float64,
    u_element::Vector{Float64}
)
    # Get material parameters for current density
    λ, μ = material_model(density)
    
    # Get base material parameters (for derivative calculation)
    λ0, μ0 = material_model(1.0)  # Material parameters at full density
    λmin, μmin = material_model(1e-9)  # Material parameters at void
    
    # Reinitialize cell values
    reinit!(cellvalues, cell)
    
    # Calculate element stiffness matrix for unit Young's modulus
    n_basefuncs = getnbasefunctions(cellvalues)
    ke_unit = zeros(n_basefuncs, n_basefuncs)
    
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        
        for i in 1:n_basefuncs
            ∇Ni = shape_gradient(cellvalues, q_point, i)
            εi = symmetric(∇Ni)
            
            for j in 1:n_basefuncs
                ∇Nj = shape_gradient(cellvalues, q_point, j)
                εj = symmetric(∇Nj)
                
                # Use unit material properties
                σ = λ0 * tr(εj) * one(εj) + 2μ0 * εj
                ke_unit[i, j] += (εi ⊡ σ) * dΩ
            end
        end
    end
    
    # Calculate sensitivity using adjoint method
    # For SIMP: E(ρ) = Emin + ρ^p * (E0 - Emin)
    # ∂E/∂ρ = p * ρ^(p-1) * (E0 - Emin)
    
    # Assuming p = 3 (can be made parameter)
    p = 3.0
    E0 = 1.0  # Base Young's modulus
    Emin = 1e-9  # Void Young's modulus
    
    # Derivative of Young's modulus w.r.t. density
    dE_drho = p * density^(p-1) * (E0 - Emin)
    
    # Compliance sensitivity: ∂c/∂ρ = -∂E/∂ρ * u^T * k_unit * u
    sensitivity = -dE_drho * dot(u_element, ke_unit * u_element)
    
    return sensitivity
end

