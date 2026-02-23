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

# Export filter cache
export FilterCache, create_filter_cache

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
- `tolerance_checkpoints`: List of tolerance values for intermediate export
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

    # Tolerance checkpoints for multi-tolerance export
    tolerance_checkpoints::Vector{Float64}

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
        tolerance_checkpoints = Float64[],
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
            tolerance_checkpoints,
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
- `energy`: Final energy value
- `volume`: Final volume
- `iterations`: Number of iterations performed
- `converged`: Whether optimization converged
- `energy_history`: Energy at each iteration
- `volume_history`: Volume at each iteration
"""
struct OptimizationResult
    densities::Vector{Float64}
    displacements::Vector{Float64}
    stresses::Dict
    energy::Float64
    volume::Float64
    iterations::Int
    converged::Bool
    energy_history::Vector{Float64}
    volume_history::Vector{Float64}
end

# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

"""
    simp_optimize(grid, dh, cellvalues, loads, boundary_conditions, params, acceleration_data=nothing)

Run SIMP topology optimization with memory-optimized implementation.

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

    # =========================================================================
    # PRE-ALLOCATIONS (ONCE BEFORE LOOP)
    # =========================================================================
    n_cells = getncells(grid)
    n_dofs = ndofs(dh)
    n_basefuncs = getnbasefunctions(cellvalues)

    # Global matrices - allocated ONCE, reused every iteration
    K = allocate_matrix(dh)
    f = zeros(n_dofs)
    u = zeros(n_dofs)

    # Sensitivity vectors - allocated ONCE
    sensitivities = zeros(n_cells)
    filtered_sensitivities = zeros(n_cells)

    # Assembly buffer for in-place operations
    ke_buffer = zeros(n_basefuncs, n_basefuncs)
    fe_buffer = zeros(n_basefuncs)

    # Initialize densities
    densities = fill(params.volume_fraction, n_cells)
    old_densities = zeros(n_cells)

    # Create material model
    material_model = create_simp_material_model(params.E0, params.ν, params.Emin, params.p)

    # Create volume quadrature and calculate element volumes
    volume_cellvalues = create_volume_quadrature(grid)
    element_volumes = calculate_element_volumes(grid, volume_cellvalues)
    total_volume = sum(element_volumes)
    total_mesh_volume = calculate_volume(grid)

    print_data("Total mesh volume: $total_volume")

    # Create FilterCache - KD-tree built ONCE
    filter_cache = create_filter_cache(grid, params.filter_radius)
    print_filter_info(grid, params.filter_radius, "auto")

    # Create element stiffness cache if enabled
    cache =
        params.use_cache ?
        initialize_element_cache(dh, cellvalues, material_model, n_cells) : nothing

    # History tracking
    energy_history = Float64[]
    volume_history = Float64[]

    # Tolerance checkpoint tracking
    checkpoint_triggered = falses(length(params.tolerance_checkpoints))
    if !isempty(params.tolerance_checkpoints)
        print_info("Tolerance checkpoints enabled: $(params.tolerance_checkpoints)")
    end

    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    converged = false
    iteration = 0

    for iter = 1:params.max_iterations
        iteration = iter
        copyto!(old_densities, densities)

        # Reset matrices (NO reallocation - just zero out values)
        fill!(K.nzval, 0.0)
        fill!(f, 0.0)

        # Assemble system with in-place operations
        assemble_stiffness_matrix_simp_inplace!(
            K,
            f,
            dh,
            cellvalues,
            material_model,
            densities,
            cache,
            ke_buffer,
            fe_buffer,
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

        # Solve using Symmetric wrapper for CHOLMOD compatibility
        u .= cholesky(Symmetric(K, :L)) \ f

        # Calculate energy and volume
        energy = 0.5 * dot(u, K * u)
        current_volume = calculate_volume(grid, densities)
        current_volume_fraction = current_volume / total_mesh_volume

        push!(energy_history, energy)
        push!(volume_history, current_volume)

        # Sensitivity analysis (reuse sensitivities vector)
        calculate_sensitivities!(
            sensitivities,
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

        # Density filtering with cached neighbors (zero allocations)
        apply_density_filter_cached!(
            filtered_sensitivities,
            filter_cache,
            densities,
            sensitivities,
        )

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
            log_iteration!(logger, iteration, energy, current_volume_fraction, change)
        end

        # Print progress
        @printf(
            "Iter %4d | Energy: %.4e | Vol.Frac: %.4f | Change: %.4e\n",
            iteration,
            energy,
            current_volume_fraction,
            change
        )

        # -----------------------------------------------------------------
        # CHECK TOLERANCE CHECKPOINTS
        # -----------------------------------------------------------------
        if !isempty(params.tolerance_checkpoints) && !isempty(params.export_path)
            for (idx, cp) in enumerate(params.tolerance_checkpoints)
                if !checkpoint_triggered[idx] && change < cp
                    checkpoint_triggered[idx] = true
                    tol_pct = round(Int, cp * 100)
                    tol_str = @sprintf("%02d", tol_pct)

                    print_info(
                        "Tolerance checkpoint $(cp) reached at iteration $(iteration)",
                    )

                    # Calculate stresses for checkpoint export
                    stress_cp, _, _ = calculate_stresses_simp(
                        u,
                        dh,
                        cellvalues,
                        material_model,
                        densities,
                    )
                    cp_result = OptimizationResult(
                        copy(densities),
                        copy(u),
                        stress_cp,
                        energy,
                        current_volume,
                        iteration,
                        false,
                        copy(energy_history),
                        copy(volume_history),
                    )
                    cp_results_data = create_results_data(grid, dh, cp_result)
                    PostProcessing.export_main_results(
                        cp_results_data,
                        joinpath(params.export_path, "final_results_$(tol_str)tol"),
                    )
                    print_success("Checkpoint exported: final_results_$(tol_str)tol.vtu")
                end
            end
        end

        # Export intermediate results (periodic interval export)
        if params.export_interval > 0 && iteration % params.export_interval == 0
            if !isempty(params.export_path)
                # Reuse existing K, f, u - no extra allocation
                stress_field_temp, _, _ =
                    calculate_stresses_simp(u, dh, cellvalues, material_model, densities)
                intermediate_result = OptimizationResult(
                    copy(densities),
                    copy(u),
                    stress_field_temp,
                    energy,
                    current_volume,
                    iteration,
                    false,
                    copy(energy_history),
                    copy(volume_history),
                )
                results_data = create_results_data(grid, dh, intermediate_result)
                export_results_vtu(
                    results_data,
                    joinpath(params.export_path, "iter_$(lpad(iteration, 4, '0'))"),
                    include_history = false,
                )
            end
        end

        # Periodic garbage collection to prevent fragmentation
        if iteration % 20 == 0
            GC.gc(false)  # Minor collection
        end

        if change < params.tolerance
            print_success("Converged after $iteration iterations")
            converged = true
            break
        end
    end

    # =========================================================================
    # FINAL ANALYSIS
    # =========================================================================
    fill!(K.nzval, 0.0)
    fill!(f, 0.0)
    assemble_stiffness_matrix_simp_inplace!(
        K,
        f,
        dh,
        cellvalues,
        material_model,
        densities,
        cache,
        ke_buffer,
        fe_buffer,
    )

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

    # Solve using Symmetric wrapper for CHOLMOD compatibility
    u .= cholesky(Symmetric(K, :L)) \ f

    final_energy = 0.5 * dot(u, K * u)
    final_volume = calculate_volume(grid, densities)

    # Calculate stresses
    stress_field, max_von_mises, max_stress_cell =
        calculate_stresses_simp(u, dh, cellvalues, material_model, densities)

    # Write summary and close logger
    if logger !== nothing
        write_summary(logger, final_energy, final_volume, converged)
        close_logger(logger)
    end

    # Final GC
    GC.gc(true)

    print_success("Optimization completed")
    print_data("Final energy: $final_energy")
    print_data("Final volume fraction: $(final_volume / total_mesh_volume)")

    return OptimizationResult(
        densities,
        u,
        stress_field,
        final_energy,
        final_volume,
        iteration,
        converged,
        energy_history,
        volume_history,
    )
end

# =============================================================================
# IN-PLACE ASSEMBLY FUNCTIONS
# =============================================================================

"""
    initialize_element_cache(dh, cellvalues, material_model, n_cells)

Initialize element stiffness matrix cache with unit matrices.
"""
function initialize_element_cache(dh, cellvalues, material_model, n_cells)
    n_basefuncs = getnbasefunctions(cellvalues)
    unit_matrices = Vector{Matrix{Float64}}(undef, n_cells)
    λ_unit, μ_unit = material_model(1.0)

    print_info("Computing element stiffness matrix cache...")

    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        reinit!(cellvalues, cell)
        ke_unit = assemble_element_stiffness_matrix(cellvalues, λ_unit, μ_unit)
        unit_matrices[cell_id] = ke_unit
    end

    # Store unit material parameters for scaling
    cache = Dict{Symbol,Any}()
    cache[:unit_matrices] = unit_matrices
    cache[:λ_unit] = λ_unit
    cache[:μ_unit] = μ_unit

    print_success("Element cache initialized: $(n_cells) unit matrices")
    return cache
end

"""
    assemble_stiffness_matrix_simp_inplace!(K, f, dh, cellvalues, material_model, densities, cache, ke_buffer, fe_buffer)

Assemble global stiffness matrix with in-place operations to avoid allocations.
"""
function assemble_stiffness_matrix_simp_inplace!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    densities,
    cache,
    ke_buffer,
    fe_buffer,
)
    if cache !== nothing
        assemble_with_cache_inplace!(
            K,
            f,
            dh,
            material_model,
            densities,
            cache,
            ke_buffer,
            fe_buffer,
        )
    else
        assemble_variable_material_inplace!(
            K,
            f,
            dh,
            cellvalues,
            material_model,
            densities,
            ke_buffer,
            fe_buffer,
        )
    end
end

"""
    assemble_with_cache_inplace!(K, f, dh, material_model, densities, cache, ke_buffer, fe_buffer)

Assembly using cached unit matrices with in-place scaling.
"""
function assemble_with_cache_inplace!(
    K,
    f,
    dh,
    material_model,
    densities,
    cache,
    ke_buffer,
    fe_buffer,
)
    unit_matrices = cache[:unit_matrices]
    λ_unit = cache[:λ_unit]

    fill!(fe_buffer, 0.0)
    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        density = densities[cell_id]

        # Get current material parameters
        λ_current, _ = material_model(density)
        scaling_factor = λ_current / λ_unit

        # In-place scaling: ke_buffer = scaling_factor * unit_matrix
        unit_ke = unit_matrices[cell_id]
        @inbounds for j = 1:size(ke_buffer, 2)
            for i = 1:size(ke_buffer, 1)
                ke_buffer[i, j] = scaling_factor * unit_ke[i, j]
            end
        end

        assemble!(assembler, celldofs(cell), ke_buffer, fe_buffer)
    end
end

"""
    assemble_variable_material_inplace!(K, f, dh, cellvalues, material_model, densities, ke_buffer, fe_buffer)

Assembly for variable material properties without caching.
"""
function assemble_variable_material_inplace!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    densities,
    ke_buffer,
    fe_buffer,
)
    fill!(fe_buffer, 0.0)
    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)

        cell_id = cellid(cell)
        density = densities[cell_id]
        λ, μ = material_model(density)

        # Compute element stiffness into buffer
        assemble_element_stiffness_matrix!(ke_buffer, cellvalues, λ, μ)
        assemble!(assembler, celldofs(cell), ke_buffer, fe_buffer)
    end
end

"""
    assemble_element_stiffness_matrix!(ke, cellvalues, λ, μ)

Compute element stiffness matrix in-place.
"""
function assemble_element_stiffness_matrix!(ke::Matrix{Float64}, cellvalues, λ, μ)
    n_basefuncs = getnbasefunctions(cellvalues)
    fill!(ke, 0.0)

    for q_point = 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)

        for i = 1:n_basefuncs
            ∇Ni = shape_gradient(cellvalues, q_point, i)
            εi = symmetric(∇Ni)

            for j = 1:n_basefuncs
                ∇Nj = shape_gradient(cellvalues, q_point, j)
                εj = symmetric(∇Nj)
                σ = λ * tr(εj) * one(εj) + 2μ * εj
                ke[i, j] += (εi ⊡ σ) * dΩ
            end
        end
    end
end

"""
    assemble_element_stiffness_matrix(cellvalues, λ, μ)

Compute element stiffness matrix (allocating version for cache initialization).
"""
function assemble_element_stiffness_matrix(cellvalues, λ, μ)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    assemble_element_stiffness_matrix!(ke, cellvalues, λ, μ)
    return ke
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    apply_loads_and_bcs!(K, f, loads, boundary_conditions)

Apply all load conditions and boundary conditions to the system.
"""
function apply_loads_and_bcs!(K, f, loads, boundary_conditions)
    for load in loads
        apply_load_condition!(f, load)
    end

    for bc in boundary_conditions
        apply!(K, f, bc)
    end
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
