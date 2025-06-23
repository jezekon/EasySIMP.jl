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
    
    # Performance
    use_cache::Bool               # Enable element matrix caching (default: true)
    
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
        use_cache = true
    )
        new(E0, Emin, ν, p, volume_fraction, max_iterations, tolerance, 
            filter_radius, move_limit, damping, use_cache)
    end
end

"""
    simp_optimize(grid, dh, cellvalues, forces, boundary_conditions, params, acceleration_data=nothing)

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
    simp_optimize(grid, dh, cellvalues, material_params, forces, boundary_conditions, params, acceleration_data=nothing)

Main SIMP topology optimization function.

# Arguments
- `grid`: Ferrite Grid object
- `dh`: DofHandler
- `cellvalues`: CellValues for integration
- `forces`: Applied forces
- `boundary_conditions`: Boundary conditions
- `params`: OptimizationParameters
- `acceleration_data`: Optional tuple (acceleration_vector, base_density) for variable density acceleration

# Returns
- `OptimizationResult`: Complete optimization results
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
    print_info("Starting SIMP topology optimization")
    
    if acceleration_data !== nothing
        acceleration_vector, base_density = acceleration_data
        print_info("Variable density acceleration enabled: $(acceleration_vector) with base density $(base_density) kg/m³")
    end
    
    # Initialize
    n_cells = getncells(grid)
    densities = fill(params.volume_fraction, n_cells)
    
    # Create cache if enabled
    cache = params.use_cache ? Dict{Symbol, Any}() : nothing
    if cache !== nothing
        print_info("Performance caching enabled")
    end
    
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
    
    # Main optimization loop
    converged = false
    iteration = 0
    
    for iteration = 1:params.max_iterations
        print_info("Iteration $iteration")
        
        # Store old densities for convergence check
        old_densities = copy(densities)
        
        # Assemble system matrices (with optional caching)
        K = allocate_matrix(dh)
        f = zeros(ndofs(dh))
        
        assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities, cache)
        
        # Apply variable density acceleration if provided
        if acceleration_data !== nothing
            acceleration_vector, base_density = acceleration_data
            variable_densities = densities .* base_density
            apply_variable_density_volume_force!(f, dh, cellvalues, acceleration_vector, variable_densities)
        end
        
        # Apply forces and boundary conditions
        apply_forces_and_bcs!(K, f, forces, boundary_conditions)
        
        # Debug: Check if forces are non-zero
        force_magnitude = norm(f)
        println("Total force vector magnitude: $(force_magnitude)")
        
        # Solve FE system
        u = K \ f
        
        # Calculate compliance
        compliance = 0.5 * dot(u, K * u)
        current_volume = calculate_volume(grid, densities)
        
        # Store history
        push!(compliance_history, compliance)
        push!(volume_history, current_volume)
        
        print_data("Compliance: $compliance")
        
        # Sensitivity analysis
        sensitivities = calculate_sensitivities(
            grid, dh, cellvalues, material_model, densities, u
        )
        
        # Density filtering with proper element size scaling
        filtered_sensitivities = apply_density_filter(
            grid, densities, sensitivities, params.filter_radius
        )
        
        # Validate filter parameters on first iteration
        if iteration == 1
            print_filter_info(grid, params.filter_radius, "auto")
        end
        
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
        
        print_data("Volume fraction after OC: $(current_volume_fraction)")
        
        # Check for extreme values
        if current_volume_fraction < 0.01 || current_volume_fraction > 0.99
            print_warning("Extreme volume fraction detected: $current_volume_fraction")
            print_warning("This may indicate OC algorithm instability")
        end
        
        # Check convergence
        change = maximum(abs.(densities - old_densities))
        
        if change < params.tolerance
            print_success("Converged after $iteration iterations")
            converged = true
            break
        end
    end
    
    # Final analysis
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities, cache)
    
    # Apply final acceleration if provided
    if acceleration_data !== nothing
        acceleration_vector, base_density = acceleration_data
        variable_densities = densities .* base_density
        apply_variable_density_volume_force!(f, dh, cellvalues, acceleration_vector, variable_densities)
    end
    
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

end # module
