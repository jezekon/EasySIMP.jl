module Optimization

using Ferrite
using SparseArrays
using LinearAlgebra
using Printf
using ..FiniteElementAnalysis
using ..Utils

# Export main functions
export simp_optimize, OptimizationParameters, OptimizationResult

# Include submodules
include("OptimalityCriteria.jl")
include("DensityFilter.jl") 
include("SensitivityAnalysis.jl")

"""
    OptimizationParameters

Parameters for SIMP topology optimization.
"""
mutable struct OptimizationParameters
    # Material properties
    E0::Float64                    # Young's modulus of solid material
    Emin::Float64                  # Young's modulus of void material
    ν::Float64                     # Poisson's ratio
    p::Float64                     # Penalization power (default: 3.0)
    
    # Optimization settings
    volume_fraction::Float64       # Target volume fraction
    max_iterations::Int           # Maximum iterations (default: 200)
    tolerance::Float64            # Convergence tolerance (default: 0.01)
    
    # Filter settings
    filter_radius::Float64        # Density filter radius
    
    # OC parameters
    move_limit::Float64           # Move limit for OC (default: 0.2)
    damping::Float64              # Damping coefficient (default: 0.5)
    
    # Default constructor
    function OptimizationParameters(;
        E0 = 1.0,
        Emin = 1e-9,
        ν = 0.3,
        p = 3.0,
        volume_fraction = 0.5,
        max_iterations = 200,
        tolerance = 0.01,
        filter_radius = 1.5,
        move_limit = 0.2,
        damping = 0.5
    )
        new(E0, Emin, ν, p, volume_fraction, max_iterations, tolerance, 
            filter_radius, move_limit, damping)
    end
end

"""
    OptimizationResult

Results from SIMP topology optimization.
"""
struct OptimizationResult
    densities::Vector{Float64}      # Final density distribution
    displacements::Vector{Float64}   # Final displacement field
    stresses::Dict                  # Stress field
    compliance::Float64             # Final compliance
    volume::Float64                 # Final volume
    iterations::Int                 # Number of iterations
    converged::Bool                 # Convergence flag
    compliance_history::Vector{Float64}  # Compliance history
    volume_history::Vector{Float64}      # Volume history
end

"""
    simp_optimize(grid, dh, cellvalues, material_params, forces, boundary_conditions, params)

Main SIMP topology optimization function.

# Arguments
- `grid`: Ferrite Grid object
- `dh`: DofHandler
- `cellvalues`: CellValues for integration
- `material_params`: Material parameters (λ, μ)
- `forces`: Applied forces
- `boundary_conditions`: Boundary conditions
- `params`: OptimizationParameters

# Returns
- `OptimizationResult`: Complete optimization results
"""
function simp_optimize(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    material_params,
    forces,
    boundary_conditions,
    params::OptimizationParameters
)
    print_info("Starting SIMP topology optimization")
    
    # Initialize
    n_cells = getncells(grid)
    densities = fill(params.volume_fraction, n_cells)
    
    # Create material model
    material_model = create_simp_material_model(params.E0, params.ν, params.Emin, params.p)
    
    # History tracking
    compliance_history = Float64[]
    volume_history = Float64[]
    
    # Main optimization loop
    converged = false
    iteration = 0
    
    for iteration = 1:params.max_iterations
        print_info("Iteration $iteration")
        
        # Store old densities for convergence check
        old_densities = copy(densities)
        
        # Assemble system matrices
        K = allocate_matrix(dh)
        f = zeros(ndofs(dh))
        
        assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities)
        
        # Apply forces and boundary conditions
        apply_forces_and_bcs!(K, f, forces, boundary_conditions)
        
        # Solve FE system
        u = K \ f
        
        # Calculate compliance
        compliance = 0.5 * dot(u, K * u)
        current_volume = calculate_volume(grid, densities)
        
        # Store history
        push!(compliance_history, compliance)
        push!(volume_history, current_volume)
        
        print_data("Compliance: $compliance")
        print_data("Volume fraction: $(current_volume / calculate_volume(grid))")
        
        # Sensitivity analysis
        sensitivities = calculate_sensitivities(
            grid, dh, cellvalues, material_model, densities, u
        )
        
        # Density filtering
        filtered_sensitivities = apply_density_filter(
            grid, densities, sensitivities, params.filter_radius
        )
        
        # Update densities using OC
        densities = optimality_criteria_update(
            densities, 
            filtered_sensitivities,
            params.volume_fraction,
            calculate_volume(grid),
            params.move_limit,
            params.damping
        )
        
        # Check convergence
        change = maximum(abs.(densities - old_densities))
        print_data("Change: $change")
        
        if change < params.tolerance
            print_success("Converged after $iteration iterations")
            converged = true
            break
        end
    end
    
    # Final analysis
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities)
    apply_forces_and_bcs!(K, f, forces, boundary_conditions)
    u = K \ f
    
    final_compliance = 0.5 * dot(u, K * u)
    final_volume = calculate_volume(grid, densities)
    
    # Calculate stresses
    stress_field, max_von_mises, max_stress_cell = calculate_stresses_simp(
        u, dh, cellvalues, material_model, densities
    )
    
    print_success("Optimization completed")
    print_data("Final compliance: $final_compliance")
    print_data("Final volume fraction: $(final_volume / calculate_volume(grid))")
    
    return OptimizationResult(
        densities,
        u,
        stress_field,
        final_compliance,
        final_volume,
        iteration,
        converged,
        compliance_history,
        volume_history
    )
end

"""
Helper function to apply forces and boundary conditions
"""
function apply_forces_and_bcs!(K, f, forces, boundary_conditions)
    # Apply forces
    for force in forces
        apply_force!(f, force...)
    end
    
    # Apply boundary conditions
    for bc in boundary_conditions
        apply!(K, f, bc)
    end
end

end # module
