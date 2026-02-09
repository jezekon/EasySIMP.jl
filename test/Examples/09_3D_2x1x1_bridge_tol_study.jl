# =============================================================================
# 3D BRIDGE WITH SYMMETRY - SIMP Topology Optimization
# BATCH RUN: Multiple tolerance values
# =============================================================================
#
# Description:
#   Bridge structure under uniform pressure on the top surface.
#   Uses symmetry boundary condition on one face and simple support on
#   the opposite edge. Load applied as uniform pressure on top face.
#
# Problem Visualization (side view, XY plane at z=0.5):
#
#        Y ↑
#          |  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  p = 1.0 MPa
#      1.0 | ●███████████████████████████████████████████████
#          | █                                              █
#          | █           DESIGN DOMAIN                      █
#          | █           2.0 × 1.0 × 1.0                   █
#          | █                                              █
#       0  | ○                                              ━━ Support (U2=0)
#          | ○ ← Symmetry (U1=0)
#          └─────────────────────────────────────────────────────→ X
#            0  ● = Z-fix node [0,1,0.5] (U3=0)             2.0
#
# Boundary Conditions:
#   - Symmetry: Left face (x=0) - U1=0 only
#   - Support: Right bottom edge (x≈2, y=0, all Z) - U2=0 only
#   - Z-fix: Single node at [0, 1, 0.5] - U3=0 (prevents rigid body motion)
#
# Loading:
#   - Uniform pressure on top surface (y=1.0): p = 1.0 MPa downward
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# TOLERANCE VALUES TO TEST
# -----------------------------------------------------------------------------
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
# 4. BOUNDARY CONDITIONS (shared for all runs)
# -----------------------------------------------------------------------------
println("Applying boundary conditions...")

# Symmetry: Left face (x=0) - fixed in X direction only
symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], eps())
println("  ✓ Symmetry nodes (U1=0): $(length(symmetry_nodes))")

# Support: Bottom face (y=0), right edge (x≥1.95, all Z) - fixed in Y only
support_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps() && coord[1] >= 2.0 - 0.05 - eps()
        push!(support_nodes, node_id)
    end
end
println("  ✓ Support nodes (U2=0): $(length(support_nodes))")

# Z-fix: single node at [0, 1, 0.5] to prevent rigid body motion in Z
zfix_target = [0.0, 1.0, 0.5]
zfix_node = argmin(node_id -> norm(grid.nodes[node_id].x - zfix_target), 1:getnnodes(grid))
zfix_nodes = Set([zfix_node])
println("  ✓ Z-fix node (U3=0): node $zfix_node at $(grid.nodes[zfix_node].x)")

# -----------------------------------------------------------------------------
# 5. PRESSURE LOAD ON TOP SURFACE
# -----------------------------------------------------------------------------
pressure_magnitude = 1.0  # [MPa]
top_face_nodes = select_nodes_by_plane(grid, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], eps())
println("  ✓ Top face nodes (pressure): $(length(top_face_nodes))")

# Traction function: uniform pressure downward
traction_fn(x, y, z) = [0.0, -pressure_magnitude, 0.0]

# Create SurfaceTractionLoad (shared for all runs)
pressure_load = SurfaceTractionLoad(dh, grid, top_face_nodes, traction_fn)

# -----------------------------------------------------------------------------
# 6. BATCH OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING BATCH OPTIMIZATION")
println("Tolerance values: $tolerance_values")
println("="^80)

all_results = Dict{Float64,Any}()

for tol in tolerance_values
    # Format tolerance for folder name
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    results_dir = "./results/09_3D_2x1x1_Bridge_$(tol_str)tol_r2.0"
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
    ch_zfix = apply_sliding_boundary!(K_run, f_run, dh, zfix_nodes, [3])

    # Export boundary conditions
    export_boundary_conditions(
        grid,
        dh,
        union(symmetry_nodes, support_nodes, zfix_nodes),
        top_face_nodes,
        joinpath(results_dir, "boundary_conditions"),
    )

    # Optimization parameters
    opt_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-6,
        ν = ν,
        p = 3.0,
        volume_fraction = 0.4,
        max_iterations = 3000,
        tolerance = tol,
        filter_radius = 2.0,
        move_limit = 0.2,
        damping = 0.5,
        use_cache = true,
        export_interval = 3000,
        export_path = results_dir,
        task_name = "3D_Bridge_2x1x1",
    )

    println("  Target volume fraction: $(opt_params.volume_fraction)")
    println("  Tolerance: $(opt_params.tolerance)")
    println("  Filter radius: $(opt_params.filter_radius)")

    # Run optimization
    results = simp_optimize(
        grid,
        dh,
        cellvalues,
        [pressure_load],
        [ch_symmetry, ch_support, ch_zfix],
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

# Single thread computation:
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia -t 1 --project=. test/Examples/09_3D_2x1x1_bridge_tol_study.jl
