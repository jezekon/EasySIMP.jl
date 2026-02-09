# =============================================================================
# 3D BEAM WITH 4-CORNER FIXATION - SIMP Topology Optimization
# BATCH RUN: Multiple tolerance values
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# TOLERANCE VALUES TO TEST
# -----------------------------------------------------------------------------
# tolerance_values = [0.01, 0.02, 0.04, 0.08, 0.16]
tolerance_values = [0.16, 0.08, 0.04, 0.02, 0.01]

# -----------------------------------------------------------------------------
# 1. MESH GENERATION (shared for all runs)
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(Hexahedron, (40, 20, 20), Vec((0.0, 0.0, 0.0)), Vec((2.0, 1.0, 1.0)))
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 1.0
ν = 0.3
material_model = create_simp_material_model(E0, ν, 1e-9, 3.0)

println("\nMaterial properties:")
println("  E₀ = $E0")
println("  ν = $ν")

# -----------------------------------------------------------------------------
# 3. FEM SETUP
# -----------------------------------------------------------------------------
println("\nSetting up FEM problem...")
dh, cellvalues, K, f = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 4. BOUNDARY CONDITIONS - 4 CORNER FIXATIONS
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

xmax, ymax, zmax = 2.0, 1.0, 1.0
fix_size = 0.3

fixed_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]

    if abs(x) < 1e-6
        in_corner1 = (y <= fix_size + 1e-6) && (z <= fix_size + 1e-6)
        in_corner2 = (y >= ymax - fix_size - 1e-6) && (z <= fix_size + 1e-6)
        in_corner3 = (y <= fix_size + 1e-6) && (z >= zmax - fix_size - 1e-6)
        in_corner4 = (y >= ymax - fix_size - 1e-6) && (z >= zmax - fix_size - 1e-6)

        if in_corner1 || in_corner2 || in_corner3 || in_corner4
            push!(fixed_nodes, node_id)
        end
    end
end
println("  ✓ Fixed corner nodes: $(length(fixed_nodes))")

# -----------------------------------------------------------------------------
# 5. FORCE APPLICATION - CIRCULAR REGION
# -----------------------------------------------------------------------------
load_radius = 0.1
force_center = [xmax, ymax/2, zmax/2]

force_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]

    if abs(x - xmax) < 1e-6
        dist_sq = (y - force_center[2])^2 + (z - force_center[3])^2
        if dist_sq <= load_radius^2 + 1e-6
            push!(force_nodes, node_id)
        end
    end
end

if isempty(force_nodes)
    println("  ⚠ No force nodes found, using closest to center")
    min_dist = Inf
    closest = 1
    for node_id = 1:getnnodes(grid)
        dist = norm(grid.nodes[node_id].x - force_center)
        if dist < min_dist
            min_dist = dist
            closest = node_id
        end
    end
    force_nodes = Set([closest])
end
println("  ✓ Force nodes (circular): $(length(force_nodes))")

# -----------------------------------------------------------------------------
# 6. BATCH OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING BATCH OPTIMIZATION")
println("Tolerance values: $tolerance_values")
println("="^80)

all_results = Dict{Float64,Any}()

for tol in tolerance_values
    # Format tolerance for folder name (e.g., 0.01 -> "01", 0.08 -> "08", 0.16 -> "16")
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    results_dir = "./results/05_3D_2x1x1_4Legs_$(tol_str)tol_r2.0"
    mkpath(results_dir)

    println("\n" * "="^80)
    println("RUNNING: tolerance = $tol (folder: $(tol_str)tol)")
    println("="^80)

    # Fresh K and f for each run
    K_run = allocate_matrix(dh)
    f_run = zeros(ndofs(dh))

    # Assemble and apply BC
    assemble_stiffness_matrix_simp!(
        K_run,
        f_run,
        dh,
        cellvalues,
        material_model,
        fill(0.4, getncells(grid)),
    )

    ch_fixed = apply_fixed_boundary!(K_run, f_run, dh, fixed_nodes)
    apply_force!(f_run, dh, collect(force_nodes), [0.0, 0.0, -1.0])

    # Export boundary conditions
    export_boundary_conditions(
        grid,
        dh,
        fixed_nodes,
        force_nodes,
        joinpath(results_dir, "boundary_conditions"),
    )

    # Optimization parameters
    opt_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-9,
        ν = ν,
        p = 3.0,
        volume_fraction = 0.4,
        max_iterations = 2000,
        tolerance = tol,
        filter_radius = 2.0,
        move_limit = 0.2,
        damping = 0.5,
        use_cache = true,
        export_interval = 2000,
        export_path = results_dir,
    )

    println("  Target volume fraction: $(opt_params.volume_fraction)")
    println("  Tolerance: $(opt_params.tolerance)")
    println("  Filter radius: $(opt_params.filter_radius)")

    # Run optimization
    results = simp_optimize(
        grid,
        dh,
        cellvalues,
        [PointLoad(dh, collect(force_nodes), [0.0, 0.0, -1.0])],
        [ch_fixed],
        opt_params,
    )

    # Export final results
    results_data = create_results_data(grid, dh, results)
    export_results_vtu(results_data, joinpath(results_dir, "final"))

    # Store results
    all_results[tol] = (
        compliance = results.compliance,
        volume_fraction = results.volume / calculate_volume(grid),
        iterations = results.iterations,
        converged = results.converged,
        results_dir = results_dir,
    )

    println(
        "\n  ✓ Completed: tol=$tol, iterations=$(results.iterations), compliance=$(round(results.compliance, digits=6))",
    )

    # Clean up memory between runs
    GC.gc(true)
end

# -----------------------------------------------------------------------------
# 7. FINAL SUMMARY
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("BATCH OPTIMIZATION COMPLETED")
println("="^80)
println("\nSummary of all runs:")
println("-"^80)
@printf(
    "%-10s | %-12s | %-12s | %-10s | %-10s\n",
    "Tolerance",
    "Compliance",
    "Vol.Frac",
    "Iterations",
    "Converged"
)
println("-"^80)

for tol in sort(collect(keys(all_results)))
    r = all_results[tol]
    @printf(
        "%-10.4f | %-12.6f | %-12.4f | %-10d | %-10s\n",
        tol,
        r.compliance,
        r.volume_fraction,
        r.iterations,
        r.converged ? "Yes" : "No"
    )
end
println("-"^80)

println("\nResults saved to:")
for tol in sort(collect(keys(all_results)))
    println("  $(all_results[tol].results_dir)/")
end
println("="^80)
