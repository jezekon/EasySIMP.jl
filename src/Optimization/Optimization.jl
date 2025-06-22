module Optimization

using Ferrite
using SparseArrays
using LinearAlgebra
using Printf
using IterativeSolvers  # Add this dependency
using ..FiniteElementAnalysis
using ..Utils

# Export main functions
export simp_optimize, OptimizationParameters, OptimizationResult
export SolverOptions, WarmStartSolver, create_optimized_solver_options  # New exports

# Include submodules
include("ProgressTable.jl")
include("OptimalityCriteria.jl")
include("DensityFilter.jl") 
include("SensitivityAnalysis.jl")
include("WarmStartSolver.jl")  # New optimized solver

"""
    OptimizationParameters

Parameters for SIMP topology optimization with performance options.
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
    
    # NEW: Solver optimization options
    solver_options::SolverOptions  # Advanced solver configuration
    
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
        damping = 0.5,
        solver_options = SolverOptions()  # Default optimized solver options
    )
        new(E0, Emin, ν, p, volume_fraction, max_iterations, tolerance, 
            filter_radius, move_limit, damping, solver_options)
    end
end

"""
    OptimizationResult

Results from SIMP topology optimization with performance metrics.
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
    
    # NEW: Performance metrics
    total_cg_iterations::Int        # Total CG iterations used
    average_cg_per_iteration::Float64  # Average CG iterations per solve
    solver_strategy_history::Vector{Symbol}  # Solver strategies used
end

"""
    simp_optimize(grid, dh, cellvalues, material_params, forces, boundary_conditions, params, acceleration_data=nothing)

Main SIMP topology optimization function with performance optimizations.

# Arguments
- `grid`: Ferrite Grid object
- `dh`: DofHandler
- `cellvalues`: CellValues for integration
- `forces`: Applied forces
- `boundary_conditions`: Boundary conditions
- `params`: OptimizationParameters (now with solver_options)
- `acceleration_data`: Optional tuple (acceleration_vector, base_density) for variable density acceleration

# Returns
- `OptimizationResult`: Complete optimization results with performance metrics
"""
function simp_optimize(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    forces,
    boundary_conditions,
    params::OptimizationParameters,
    acceleration_data=nothing
)
    print_info("Starting SIMP topology optimization with performance optimizations")
    
    if acceleration_data !== nothing
        acceleration_vector, base_density = acceleration_data
        print_info("Variable density acceleration enabled: $(acceleration_vector) with base density $(base_density) kg/m³")
    end
    
    # Initialize
    n_cells = getncells(grid)
    n_dofs = ndofs(dh)
    densities = fill(params.volume_fraction, n_cells)
    
    # NEW: Initialize optimized solver
    solver = WarmStartSolver(n_dofs, n_cells, params.solver_options)
    print_info("Initialized warm-start solver with adaptive tolerance")
    
    # Create special cellvalues for volume calculation with higher order quadrature
    volume_cellvalues = create_volume_quadrature(grid)
    
    # Calculate element volumes with higher order quadrature
    element_volumes = calculate_element_volumes(grid, volume_cellvalues)
    total_volume = sum(element_volumes)
    
    print_data("Total mesh volume: $total_volume")
    print_data("Element volume range: [$(minimum(element_volumes)), $(maximum(element_volumes))]")
    
    # Create material model
    material_model = create_simp_material_model(params.E0, params.ν, params.Emin, params.p)
    
    # History tracking
    compliance_history = Float64[]
    volume_history = Float64[]
    solver_strategy_history = Symbol[]
    
    # NEW: Pre-allocate matrices with proper sparsity pattern (KEY OPTIMIZATION)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    print_info("Pre-allocated sparse matrix structure ($(nnz(K)) non-zeros)")
    
    # Main optimization loop
    converged = false
    iteration = 0
    old_densities = copy(densities)
    
    print_header()  # Print progress table header
    
    for iteration = 1:params.max_iterations
        # Store old densities for convergence check
        old_densities = copy(densities)
        
        # NEW: Optimized assembly and solve with warm-start
        u, density_change, strategy = solve_system_optimized!(
            solver, K, f, dh, cellvalues, material_model,
            old_densities, densities, boundary_conditions...
        )
        
        # Store strategy used
        push!(solver_strategy_history, strategy)
        
        # Apply forces (needs to be done after assembly)
        # Reset force vector (it was modified in solve_system_optimized!)
        fill!(f, 0.0)
        
        # Apply variable density acceleration if provided
        if acceleration_data !== nothing
            acceleration_vector, base_density = acceleration_data
            variable_densities = densities .* base_density
            apply_variable_density_volume_force!(f, dh, cellvalues, acceleration_vector, variable_densities)
        end
        
        # Apply other forces
        for force in forces
            apply_force!(f, force...)
        end
        
        # Re-solve with forces (this could be optimized further by combining with previous solve)
        for ch in boundary_conditions
            apply!(K, f, ch)
        end
        u = K \ f  # Quick direct solve since K is already factorized (if using direct solver)
        
        # Calculate compliance
        compliance = 0.5 * dot(u, K * u)
        current_volume = calculate_volume(grid, densities)
        
        # Store history
        push!(compliance_history, compliance)
        push!(volume_history, current_volume)
        
        # Sensitivity analysis
        sensitivities = calculate_sensitivities(
            grid, dh, cellvalues, material_model, densities, u
        )
        
        # Density filtering with proper element size scaling
        filtered_sensitivities = apply_density_filter(
            grid, densities, sensitivities, params.filter_radius
        )
        
        # Update densities using OC
        densities = optimality_criteria_update(
              densities, 
              filtered_sensitivities,
              params.volume_fraction,
              total_volume,
              element_volumes,
              params.move_limit,
              params.damping
          )

        # Volume constraint diagnostics
        current_volume = calculate_volume(grid, densities)
        current_volume_fraction = current_volume / calculate_volume(grid)
        
        # Print progress (with solver information)
        progress = OptimizationProgress(
            iteration,
            current_volume_fraction,
            compliance,
            norm(f),
            check_sensitivity_health_quiet(sensitivities),
            maximum(abs.(densities - old_densities))
        )
        print_iteration(progress)
        
        # Check convergence
        change = maximum(abs.(densities - old_densities))
        
        if change < params.tolerance
            print_success("Converged after $iteration iterations")
            converged = true
            break
        end
        
        # Print periodic solver statistics
        if iteration % 10 == 0
            print_solver_statistics(solver)
        end
    end
    
    # Final analysis with optimized solver
    print_info("Performing final analysis...")
    u, _, _ = solve_system_optimized!(
        solver, K, f, dh, cellvalues, material_model,
        old_densities, densities, boundary_conditions...
    )
    
    # Apply final forces
    fill!(f, 0.0)
    if acceleration_data !== nothing
        acceleration_vector, base_density = acceleration_data
        variable_densities = densities .* base_density
        apply_variable_density_volume_force!(f, dh, cellvalues, acceleration_vector, variable_densities)
    end
    
    for force in forces
        apply_force!(f, force...)
    end
    
    for ch in boundary_conditions
        apply!(K, f, ch)
    end
    u = K \ f
    
    final_compliance = 0.5 * dot(u, K * u)
    final_volume = calculate_volume(grid, densities)
    
    # Calculate stresses
    stress_field, max_von_mises, max_stress_cell = calculate_stresses_simp(
        u, dh, cellvalues, material_model, densities
    )
    
    # Print final performance statistics
    print_solver_statistics(solver)
    
    print_success("Optimization completed")
    print_data("Final compliance: $final_compliance")
    print_data("Final volume fraction: $(final_volume / calculate_volume(grid))")
    
    # Calculate performance metrics
    avg_cg_per_iteration = solver.total_cg_iterations / max(1, iteration)
    
    return OptimizationResult(
        densities,
        u,
        stress_field,
        final_compliance,
        final_volume,
        iteration,
        converged,
        compliance_history,
        volume_history,
        solver.total_cg_iterations,
        avg_cg_per_iteration,
        solver_strategy_history
    )
end

"""
Helper function to apply forces and boundary conditions
"""
function apply_forces_and_bcs!(K, f, forces, boundary_conditions)
    # Apply point forces
    for force in forces
        apply_force!(f, force...)
    end
    
    # Apply boundary conditions
    for bc in boundary_conditions
        apply!(K, f, bc)
    end
end

"""
    calculate_element_volumes(grid, cellvalues)
    
Calculate volume of each element in the grid using Gaussian quadrature.
"""
function calculate_element_volumes(grid::Grid, cellvalues)
    n_cells = getncells(grid)
    element_volumes = zeros(n_cells)
    
    # Iterate over cell indices
    for cell_idx in 1:n_cells
        # Get coordinates of the cell using cell index
        coords = getcoordinates(grid, cell_idx)
        
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, coords)
        
        # Integrate 1 over element to get volume
        volume = 0.0
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            volume += dΩ
        end
        
        element_volumes[cell_idx] = volume
    end
    
    return element_volumes
end

"""
    create_volume_quadrature(grid)
    
Create cellvalues specifically for volume calculation with 3rd order quadrature.
"""
function create_volume_quadrature(grid::Grid{dim}) where dim
    # Get cell type from first cell
    cell = getcells(grid, 1)
    
    if cell isa Hexahedron
        # For hexahedral elements - 3rd order quadrature
        ip = Lagrange{RefHexahedron, 1}()
        qr = QuadratureRule{RefHexahedron}(3)  # 3rd order
    elseif cell isa Tetrahedron  
        # For tetrahedral elements - 3rd order quadrature
        ip = Lagrange{RefTetrahedron, 1}()
        qr = QuadratureRule{RefTetrahedron}(3)  # 3rd order
    elseif cell isa Quadrilateral && dim == 2
        # For 2D quad elements
        ip = Lagrange{RefQuadrilateral, 1}()
        qr = QuadratureRule{RefQuadrilateral}(3)
    elseif cell isa Triangle && dim == 2
        # For 2D triangle elements  
        ip = Lagrange{RefTriangle, 1}()
        qr = QuadratureRule{RefTriangle}(3)
    else
        error("Unsupported cell type: $(typeof(cell))")
    end
    
    return CellValues(qr, ip)
end

"""
    create_optimized_solver_options(;conservative=false)

Create optimized solver options with recommended settings.

# Arguments
- `conservative`: If true, use more conservative settings for stability

# Returns
- `SolverOptions`: Configured solver options
"""
function create_optimized_solver_options(;conservative::Bool=false)
    if conservative
        return SolverOptions(
            use_warm_start = false,
            preconditioner_update_freq = 3,  # More frequent updates
            iterative_tolerance_adaptive = true,
            incremental_assembly_threshold = 0.01,  # Higher threshold
            relaxed_tolerance = 1e-4,  # Stricter tolerance
            strict_tolerance = 1e-6,
            max_cg_iterations = 500
        )
    else
        return SolverOptions(
            use_warm_start = true,
            preconditioner_update_freq = 3,
            iterative_tolerance_adaptive = true,
            incremental_assembly_threshold = 0.01,
            relaxed_tolerance = 1e-4,  # More aggressive tolerance
            strict_tolerance = 1e-6,
            max_cg_iterations = 500 
        )
    end
end

end # module
