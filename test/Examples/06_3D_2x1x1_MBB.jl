# =============================================================================
# 3D MBB BEAM WITH SYMMETRY - SIMP Topology Optimization
# =============================================================================
#
# Description:
#   Classic MBB (Messerschmitt-Bölkow-Blohm) beam benchmark problem.
#   Uses symmetry boundary condition on one face and simple support on
#   the opposite edge. Load applied via semicircular region on top edge.
#
# Boundary Conditions:
#   - Symmetry: Left face (x=0) - U1=0 only
#   - Support: Right bottom edge (x=2, y=0, all Z) - U2=0 only
#
# Loading:
#   - Semicircle load: Center at (0, 1, 0.5), radius 0.1 mm
#   - Force direction: [0, -1, 0] N
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra

# -----------------------------------------------------------------------------
# 1. MESH GENERATION
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(Hexahedron, (40, 20, 20), Vec((0.0, 0.0, 0.0)), Vec((2.0, 1.0, 1.0)))
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 200.0
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

# Assemble and apply boundary conditions
assemble_stiffness_matrix_simp!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    fill(0.4, getncells(grid)),
)

ch_symmetry = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
ch_support = apply_sliding_boundary!(K, f, dh, support_nodes, [2])

apply_force!(f, dh, collect(force_nodes), [0.0, -1.0, 0.0])

# -----------------------------------------------------------------------------
# 5. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
results_dir = "./results/06_3D_2x1x1_MBB_4tol_r2.0"
mkpath(results_dir)

opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-6,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.4,
    max_iterations = 2000,
    tolerance = 0.04,
    filter_radius = 2.0,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 20,
    export_path = results_dir,
    task_name = "3D_MBB_Beam_2x1x1",  # Task name for logging
)

println("\nOptimization parameters:")
println("  Task name: $(opt_params.task_name)")
println("  Volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 6. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
export_boundary_conditions(
    grid,
    dh,
    union(symmetry_nodes, support_nodes),
    force_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 7. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\nStarting optimization...\n")
results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
    [ch_symmetry, ch_support],
    opt_params,
)

# -----------------------------------------------------------------------------
# 8. EXPORT FINAL RESULTS
# -----------------------------------------------------------------------------
println("\nExporting final results...")
results_data = create_results_data(grid, dh, results)
export_results_vtu(results_data, joinpath(results_dir, "final"))

println("\n" * "="^80)
println("OPTIMIZATION COMPLETED")
println("="^80)
println("Final compliance: $(results.compliance)")
println("Final volume fraction: $(results.volume / calculate_volume(grid))")
println("Iterations: $(results.iterations)")
println("Converged: $(results.converged)")
println("\nOutput files:")
println("  $(results_dir)/optimization_progress.csv  (iteration data)")
println("  $(results_dir)/optimization_summary.txt   (final summary)")
println("  $(results_dir)/final_results.vtu          (ParaView visualization)")
println("="^80)
