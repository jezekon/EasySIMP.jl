# =============================================================================
# 3D MBB BEAM WITH SYMMETRY - SIMP Topology Optimization
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
material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

# -----------------------------------------------------------------------------
# 3. FEM SETUP
# -----------------------------------------------------------------------------
println("Setting up FEM problem...")
dh, cellvalues, K, f = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 4. BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("Applying boundary conditions...")

# Symmetry: Left face (x=0) - fixed in X direction only
symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], eps())
println("  ✓ Symmetry nodes (X-fixed): $(length(symmetry_nodes))")

# Support: Bottom face (y=0), right edge (x≥1.95, all Z) - fixed in Y direction only
support_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps() && coord[1] >= 2.0 - 0.05 - eps()
        push!(support_nodes, node_id)
    end
end
println("  ✓ Support nodes (Y-fixed): $(length(support_nodes))")

# Z-constraint: Single node to prevent rigid body motion in Z
z_fix_node = Set{Int}()
target = [0.0, 1.0, 0.5]
min_dist = Inf
closest = 1
for node_id = 1:getnnodes(grid)
    dist = norm(grid.nodes[node_id].x - target)
    if dist < min_dist
        global min_dist = dist
        global closest = node_id
    end
end
push!(z_fix_node, closest)
println("  ✓ Z-fix node: 1 (prevents rigid body motion)")

# Force: Semicircle on top face
force_center = [0.0, 1.0, 0.5]
force_radius = 0.1 + eps()

force_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2] - 1.0) < eps()
        dx = coord[1] - force_center[1]
        dz = coord[3] - force_center[3]
        dist = sqrt(dx^2 + dz^2)
        if dist <= force_radius && coord[1] >= force_center[1] - eps()
            push!(force_nodes, node_id)
        end
    end
end
println("  ✓ Force nodes (semicircle): $(length(force_nodes))")

# -----------------------------------------------------------------------------
# 5. BATCH OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING BATCH OPTIMIZATION")
println("Tolerance values: $tolerance_values")
println("="^80)

all_results = Dict{Float64,Any}()

for tol in tolerance_values
    # Format tolerance for folder name
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    results_dir = "./results/06_3D_2x1x1_MBB_$(tol_str)tol_r2.0"
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

    ch_symmetry = apply_sliding_boundary!(K_run, f_run, dh, symmetry_nodes, [1])
    ch_support = apply_sliding_boundary!(K_run, f_run, dh, support_nodes, [2])
    ch_z_fix = apply_sliding_boundary!(K_run, f_run, dh, z_fix_node, [3])

    apply_force!(f_run, dh, collect(force_nodes), [0.0, -1.0, 0.0])

    # Export boundary conditions
    export_boundary_conditions(
        grid,
        dh,
        union(symmetry_nodes, support_nodes, z_fix_node),
        force_nodes,
        joinpath(results_dir, "boundary_conditions"),
    )

    # Optimization parameters
    opt_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-6,
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
        task_name = "3D_MBB_Beam_2x1x1",
    )

    println("  Target volume fraction: $(opt_params.volume_fraction)")
    println("  Tolerance: $(opt_params.tolerance)")
    println("  Filter radius: $(opt_params.filter_radius)")

    # Run optimization
    results = simp_optimize(
        grid,
        dh,
        cellvalues,
        [PointLoad(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
        [ch_symmetry, ch_support, ch_z_fix],
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
# 6. FINAL SUMMARY
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
