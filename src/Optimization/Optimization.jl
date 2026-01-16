module Optimization

using Ferrite
using SparseArrays
using LinearAlgebra
using Printf
using Dates
using ..FiniteElementAnalysis
using ..Utils
using ..PostProcessing

# Export main functions
export simp_optimize, OptimizationParameters, OptimizationResult

# Export load condition types
export AbstractLoadCondition, PointLoad, SurfaceTractionLoad

# Include submodules
include("LoadConditions.jl")
include("ProgressTable.jl")
include("OptimalityCriteria.jl")
include("DensityFilter.jl")
include("SensitivityAnalysis.jl")
include("OptimizationLogger.jl")

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

"""
    OptimizationParameters

Parameters for SIMP topology optimization.

# Fields
- `E0`: Base Young's modulus
- `Emin`: Minimum Young's modulus (void regions)
- `ν`: Poisson's ratio
- `p`: SIMP penalization power
- `volume_fraction`: Target volume fraction
- `max_iterations`: Maximum optimization iterations
- `tolerance`: Convergence tolerance
- `filter_radius`: Density filter radius (× element size)
- `move_limit`: OC move limit
- `damping`: OC damping coefficient
- `use_cache`: Enable element matrix caching
- `export_interval`: Export results every N iterations (0 = disabled)
- `export_path`: Directory for intermediate results
- `task_name`: Name for logging
"""
mutable struct OptimizationParameters
    # Material properties
    E0::Float64
    Emin::Float64
    ν::Float64
    p::Float64

    # Optimization settings
    volume_fraction::Float64
    max_iterations::Int
    tolerance::Float64

    # Filter settings
    filter_radius::Float64

    # OC parameters
    move_limit::Float64
    damping::Float64

    # Performance
    use_cache::Bool

    # Intermediate export settings
    export_interval::Int
    export_path::String

    # Task name for logging
    task_name::String

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
        use_cache = true,
        export_interval = 0,
        export_path = "",
        task_name = "SIMP_Optimization",
    )
        new(
            E0,
            Emin,
            ν,
            p,
            volume_fraction,
            max_iterations,
            tolerance,
            filter_radius,
            move_limit,
            damping,
            use_cache,
            export_interval,
            export_path,
            task_name,
        )
    end
end

# =============================================================================
# OPTIMIZATION RESULT
# =============================================================================

"""
    OptimizationResult

Results from SIMP topology optimization.

# Fields
- `densities`: Final density distribution
- `displacements`: Final displacement vector
- `stresses`: Stress field (Dict of stress tensors)
- `compliance`: Final compliance value
- `volume`: Final volume
- `iterations`: Number of iterations performed
- `converged`: Whether optimization converged
- `compliance_history`: Compliance at each iteration
- `volume_history`: Volume at each iteration
"""
struct OptimizationResult
    densities::Vector{Float64}
    displacements::Vector{Float64}
    stresses::Dict
    compliance::Float64
    volume::Float64
    iterations::Int
    converged::Bool
    compliance_history::Vector{Float64}
    volume_history::Vector{Float64}
end

# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

"""
    simp_optimize(grid, dh, cellvalues, loads, boundary_conditions, params, acceleration_data=nothing)

Run SIMP topology optimization.

# Arguments
- `grid`: Ferrite Grid object
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation
- `loads`: Vector of load conditions (PointLoad, SurfaceTractionLoad, or legacy tuple)
- `boundary_conditions`: Vector of ConstraintHandlers
- `params`: OptimizationParameters
- `acceleration_data`: Optional tuple (acceleration_vector, density) for body forces

# Returns
- `OptimizationResult` containing final design and history

# Example
```julia
results = simp_optimize(grid, dh, cellvalues, [load], [ch_fixed], params)
```
"""
function simp_optimize(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    loads,
    boundary_conditions,
    params::OptimizationParameters,
    acceleration_data = nothing,
)
    print_info("Starting SIMP topology optimization")

    # Initialize logger
    logger = nothing
    if !isempty(params.export_path)
        logger = OptimizationLogger(params.export_path, params.task_name)
    end

    if acceleration_data !== nothing
        acceleration_vector, base_density = acceleration_data
        print_info("Variable density acceleration enabled: $(acceleration_vector)")
    end

    # Initialize densities
    n_cells = getncells(grid)
    densities = fill(params.volume_fraction, n_cells)

    # Create cache if enabled
    cache = params.use_cache ? Dict{Symbol,Any}() : nothing

    # Create volume quadrature and calculate element volumes
    volume_cellvalues = create_volume_quadrature(grid)
    element_volumes = calculate_element_volumes(grid, volume_cellvalues)
    total_volume = sum(element_volumes)
    total_mesh_volume = calculate_volume(grid)

    print_data("Total mesh volume: $total_volume")

    # Create material model
    material_model = create_simp_material_model(params.E0, params.ν, params.Emin, params.p)

    # History tracking
    compliance_history = Float64[]
    volume_history = Float64[]

    # Main optimization loop
    converged = false
    iteration = 0

    for iteration = 1:params.max_iterations
        old_densities = copy(densities)

        # Assemble system
        K = allocate_matrix(dh)
        f = zeros(ndofs(dh))
        assemble_stiffness_matrix_simp!(
            K,
            f,
            dh,
            cellvalues,
            material_model,
            densities,
            cache,
        )

        # Apply acceleration if provided
        if acceleration_data !== nothing
            acceleration_vector, base_density = acceleration_data
            variable_densities = densities .* base_density
            apply_variable_density_volume_force!(
                f,
                dh,
                cellvalues,
                acceleration_vector,
                variable_densities,
            )
        end

        # Apply loads and boundary conditions
        apply_loads_and_bcs!(K, f, loads, boundary_conditions)

        # Solve
        u = K \ f

        # Calculate compliance and volume
        compliance = 0.5 * dot(u, K * u)
        current_volume = calculate_volume(grid, densities)
        current_volume_fraction = current_volume / total_mesh_volume

        push!(compliance_history, compliance)
        push!(volume_history, current_volume)

        # Sensitivity analysis
        sensitivities = calculate_sensitivities(
            grid,
            dh,
            cellvalues,
            densities,
            u,
            params.E0,
            params.Emin,
            params.ν,
            params.p,
        )

        # Density filtering
        filtered_sensitivities =
            apply_density_filter(grid, densities, sensitivities, params.filter_radius)

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
            params.damping,
        )

        # Check convergence
        change = maximum(abs.(densities - old_densities))

        # Log iteration
        if logger !== nothing
            log_iteration!(logger, iteration, compliance, current_volume_fraction, change)
        end

        # Print progress
        @printf(
            "Iter %4d | Compliance: %.4e | Vol.Frac: %.4f | Change: %.4e\n",
            iteration,
            compliance,
            current_volume_fraction,
            change
        )

        # Export intermediate results
        if params.export_interval > 0 && iteration % params.export_interval == 0
            if !isempty(params.export_path)
                export_intermediate_result(
                    grid,
                    dh,
                    cellvalues,
                    material_model,
                    densities,
                    loads,
                    boundary_conditions,
                    acceleration_data,
                    compliance,
                    current_volume,
                    iteration,
                    cache,
                    compliance_history,
                    volume_history,
                    params.export_path,
                )
            end
        end

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

    if acceleration_data !== nothing
        acceleration_vector, base_density = acceleration_data
        variable_densities = densities .* base_density
        apply_variable_density_volume_force!(
            f,
            dh,
            cellvalues,
            acceleration_vector,
            variable_densities,
        )
    end

    apply_loads_and_bcs!(K, f, loads, boundary_conditions)
    u = K \ f

    final_compliance = 0.5 * dot(u, K * u)
    final_volume = calculate_volume(grid, densities)

    # Calculate stresses
    stress_field, max_von_mises, max_stress_cell =
        calculate_stresses_simp(u, dh, cellvalues, material_model, densities)

    # Write summary and close logger
    if logger !== nothing
        write_summary(logger, final_compliance, final_volume, converged)
        close_logger(logger)
    end

    print_success("Optimization completed")
    print_data("Final compliance: $final_compliance")
    print_data("Final volume fraction: $(final_volume / total_mesh_volume)")

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
    )
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    apply_loads_and_bcs!(K, f, loads, boundary_conditions)

Apply all load conditions and boundary conditions to the system.
Supports AbstractLoadCondition types and legacy tuple format.
"""
function apply_loads_and_bcs!(K, f, loads, boundary_conditions)
    # Apply each load condition
    for load in loads
        apply_load_condition!(f, load)
    end

    # Apply boundary conditions
    for bc in boundary_conditions
        apply!(K, f, bc)
    end
end

"""
    export_intermediate_result(...)

Export intermediate optimization results to VTU file.
"""
function export_intermediate_result(
    grid,
    dh,
    cellvalues,
    material_model,
    densities,
    loads,
    boundary_conditions,
    acceleration_data,
    compliance,
    current_volume,
    iteration,
    cache,
    compliance_history,
    volume_history,
    export_path,
)
    K_temp = allocate_matrix(dh)
    f_temp = zeros(ndofs(dh))
    assemble_stiffness_matrix_simp!(
        K_temp,
        f_temp,
        dh,
        cellvalues,
        material_model,
        densities,
        cache,
    )

    if acceleration_data !== nothing
        acceleration_vector, base_density = acceleration_data
        variable_densities = densities .* base_density
        apply_variable_density_volume_force!(
            f_temp,
            dh,
            cellvalues,
            acceleration_vector,
            variable_densities,
        )
    end

    apply_loads_and_bcs!(K_temp, f_temp, loads, boundary_conditions)
    u_temp = K_temp \ f_temp

    stress_field_temp, _, _ =
        calculate_stresses_simp(u_temp, dh, cellvalues, material_model, densities)

    intermediate_result = OptimizationResult(
        copy(densities),
        u_temp,
        stress_field_temp,
        compliance,
        current_volume,
        iteration,
        false,
        copy(compliance_history),
        copy(volume_history),
    )

    results_data = create_results_data(grid, dh, intermediate_result)
    export_results_vtu(
        results_data,
        joinpath(export_path, "iter_$(lpad(iteration, 4, '0'))"),
        include_history = false,
    )
end

"""
    calculate_element_volumes(grid, cellvalues)

Calculate volume of each element in the grid.
"""
function calculate_element_volumes(grid::Grid, cellvalues)
    n_cells = getncells(grid)
    element_volumes = zeros(n_cells)

    for cell_idx = 1:n_cells
        coords = getcoordinates(grid, cell_idx)
        reinit!(cellvalues, coords)

        volume = 0.0
        for q_point = 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            volume += dΩ
        end
        element_volumes[cell_idx] = volume
    end

    return element_volumes
end

"""
    create_volume_quadrature(grid)

Create CellValues for volume calculation with 3rd order quadrature.
"""
function create_volume_quadrature(grid::Grid{dim}) where {dim}
    cell = getcells(grid, 1)

    if cell isa Hexahedron
        ip = Lagrange{RefHexahedron,1}()
        qr = QuadratureRule{RefHexahedron}(3)
    elseif cell isa Tetrahedron
        ip = Lagrange{RefTetrahedron,1}()
        qr = QuadratureRule{RefTetrahedron}(3)
    elseif cell isa Quadrilateral && dim == 2
        ip = Lagrange{RefQuadrilateral,1}()
        qr = QuadratureRule{RefQuadrilateral}(3)
    elseif cell isa Triangle && dim == 2
        ip = Lagrange{RefTriangle,1}()
        qr = QuadratureRule{RefTriangle}(3)
    else
        error("Unsupported cell type: $(typeof(cell))")
    end

    return CellValues(qr, ip)
end

end # module
