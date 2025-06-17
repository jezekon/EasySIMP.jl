"""
SensitivityAnalysis.jl

Implementation of sensitivity analysis for SIMP topology optimization.
Calculates derivatives of objective function with respect to design variables.
"""

using Ferrite
using LinearAlgebra
using ..FiniteElementAnalysis

export calculate_sensitivities, calculate_compliance_sensitivity, 
       calculate_volume_sensitivity

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

"""
    calculate_volume_sensitivity(grid)

Calculate sensitivities of volume constraint with respect to element densities.

For unit element volumes: ∂V/∂ρe = 1.0 for all elements
"""
function calculate_volume_sensitivity(grid::Grid)
    n_cells = getncells(grid)
    # For unit element volumes, volume sensitivity is 1.0 for all elements
    return ones(n_cells)
end

"""
    calculate_stress_sensitivity(grid, dh, cellvalues, material_model, densities, u)

Calculate sensitivities of stress constraints (for stress-constrained optimization).
This is more advanced and not needed for basic compliance minimization.
"""
function calculate_stress_sensitivity(
    grid::Grid,
    dh::DofHandler, 
    cellvalues,
    material_model,
    densities::Vector{Float64},
    u::Vector{Float64}
)
    # Placeholder for stress-constrained optimization
    # This would calculate ∂σ_vm/∂ρe for von Mises stress constraints
    @warn "Stress sensitivity calculation not yet implemented"
    return zeros(getncells(grid))
end

"""
    verify_sensitivities(grid, dh, cellvalues, material_model, densities, u; 
                        perturbation=1e-6)

Verify analytical sensitivities using finite differences (for debugging).

# Arguments
- All parameters as in calculate_sensitivities
- `perturbation`: Size of finite difference perturbation

# Returns
- Comparison between analytical and finite difference sensitivities
"""
function verify_sensitivities(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    material_model,
    densities::Vector{Float64},
    u::Vector{Float64};
    perturbation::Float64 = 1e-6
)
    n_cells = getncells(grid)
    
    # Calculate analytical sensitivities
    analytical = calculate_sensitivities(grid, dh, cellvalues, material_model, densities, u)
    
    # Calculate finite difference sensitivities
    finite_diff = zeros(n_cells)
    
    # Calculate baseline compliance
    c0 = calculate_compliance(grid, dh, cellvalues, material_model, densities, u)
    
    for i = 1:min(10, n_cells)  # Only check first 10 elements for efficiency
        # Perturb density
        densities_pert = copy(densities)
        densities_pert[i] += perturbation
        
        # Recalculate compliance
        K_pert = allocate_matrix(dh)
        f_pert = zeros(ndofs(dh))
        assemble_stiffness_matrix_simp!(K_pert, f_pert, dh, cellvalues, material_model, densities_pert)
        u_pert = K_pert \ f_pert
        c_pert = calculate_compliance(grid, dh, cellvalues, material_model, densities_pert, u_pert)
        
        # Finite difference approximation
        finite_diff[i] = (c_pert - c0) / perturbation
    end
    
    # Compare results
    println("Sensitivity verification (first 10 elements):")
    println("Element | Analytical | Finite Diff | Relative Error")
    for i = 1:min(10, n_cells)
        rel_error = abs(analytical[i] - finite_diff[i]) / (abs(analytical[i]) + 1e-12)
        @printf("%7d | %10.4e | %11.4e | %13.4e\n", i, analytical[i], finite_diff[i], rel_error)
    end
    
    return analytical, finite_diff
end

"""
    calculate_compliance(grid, dh, cellvalues, material_model, densities, u)

Helper function to calculate total compliance.
"""
function calculate_compliance(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    material_model, 
    densities::Vector{Float64},
    u::Vector{Float64}
)
    # Simple calculation: c = 0.5 * u^T * K * u
    # But we need to reassemble K for given densities
    
    K = allocate_matrix(dh) 
    f = zeros(ndofs(dh))
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities)
    
    return 0.5 * dot(u, K * u)
end
