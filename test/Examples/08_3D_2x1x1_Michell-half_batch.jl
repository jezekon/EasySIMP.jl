# =============================================================================
# 3D MICHELL-TYPE BEAM (HALF) - SIMP Topology Optimization
# Single run with tolerance checkpoint export
# =============================================================================
#
# Description:
#   Michell-type beam problem using one symmetry plane in Z.
#   Two corner supports on bottom face at z=0 side, symmetry at z=1.
#   Single run with multi-tolerance checkpoint export.
#
# Problem Visualization (side view, XY plane at z=0):
#
#        Y ↑
#          |
#      1.0 |████████████████████████████████████████████████
#          |█                                              █
#          |█         DESIGN DOMAIN                        █
#          |█         2.0 × 1.0 × 1.0                      █
#          |█                                              █
#       0  |████████████████████████████████████████████████
#          △△                   ↓ F                      △△
#          U2=0          circle r=0.1                    U2=0
#       (4×4 elem)    at [1,0,0.5], [0,-1,0]          (4×4 elem)
#          └─────────────────────────────────────────────────→ X
#          0                   1.0                        2.0
#
#        (Z dimension: 0 to 1.0, symmetry at z=1.0)
#        2 corner supports: 4×4 elements each at (x=0,z=0), (x=2,z=0)
#
# Boundary Conditions:
#   - Support: 2 corners on bottom face (y=0) at z=0 side, 4×4 elements each - U2=0
#   - Symmetry plane XY at z=1.0: U3=0
#   - Symmetry plane ZY at x=1.0: U1=0
#   - Point load: Circular region at [1,0,0.5], radius 0.1 - F = [0, -1, 0] N
#
# Tolerance Checkpoints:
#   Single run with tol=0.01, results exported when change first drops below:
#   [0.16, 0.08, 0.04, 0.02, 0.01] → final_results_16tol.vtu, etc.
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# 1. MESH GENERATION
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(Hexahedron, (40, 20, 20), Vec((0.0, 0.0, 0.0)), Vec((2.0, 1.0, 1.0)))
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# Mesh parameters
xmax, ymax, zmax = 2.0, 1.0, 1.0
dx = xmax / 40  # 0.05
dy = ymax / 20  # 0.05
dz = zmax / 20  # 0.05

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
# 4. BOUNDARY CONDITIONS - TWO CORNER SUPPORTS ON BOTTOM FACE (z=0 side)
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

# Corner size: 4 elements × 0.05 = 0.20
corner_size = 0.20

# Support at [0,0,0] corner (x≈0, z≈0)
support_left = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]
    if abs(y) < eps() && x <= corner_size + eps() && z <= corner_size + eps()
        push!(support_left, node_id)
    end
end
println("  ✓ Support left [0,0,0] (4×4 elem): $(length(support_left)) nodes")

# Support at [2,0,0] corner (x≈xmax, z≈0)
support_right = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]
    if abs(y) < eps() && x >= xmax - corner_size - eps() && z <= corner_size + eps()
        push!(support_right, node_id)
    end
end
println("  ✓ Support right [2,0,0] (4×4 elem): $(length(support_right)) nodes")

# -----------------------------------------------------------------------------
# 5. FORCE REGION AND SYMMETRY PLANES
# -----------------------------------------------------------------------------
# Force: Circular region on bottom face (y=0)
force_center = [1.0, 0.0, 0.5]
force_radius = 0.1 + eps()

force_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps()
        dx = coord[1] - force_center[1]
        dz = coord[3] - force_center[3]
        dist = sqrt(dx^2 + dz^2)
        if dist <= force_radius
            push!(force_nodes, node_id)
        end
    end
end
println("  ✓ Force nodes (circle r=$(force_radius)): $(length(force_nodes))")

# Symmetry plane XY at z = 1.0: U3 = 0
symmetry_z_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1e-6)
println("  ✓ Symmetry plane z=1.0 (U3=0): $(length(symmetry_z_nodes)) nodes")

# Symmetry plane ZY at x = 1.0: U1 = 0
symmetry_x_nodes = select_nodes_by_plane(grid, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6)
println("  ✓ Symmetry plane x=1.0 (U1=0): $(length(symmetry_x_nodes)) nodes")

# -----------------------------------------------------------------------------
# 6. RESULTS DIRECTORY
# -----------------------------------------------------------------------------
results_dir = "./results/08_michell-half"
mkpath(results_dir)

# -----------------------------------------------------------------------------
# 7. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
all_support_nodes = union(support_left, support_right, symmetry_z_nodes, symmetry_x_nodes)

export_boundary_conditions(
    grid,
    dh,
    all_support_nodes,
    force_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 8. APPLY BOUNDARY CONDITIONS AND FORCES
# -----------------------------------------------------------------------------
println("\nApplying boundary conditions...")

assemble_stiffness_matrix_simp!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    fill(0.4, getncells(grid)),
)

# Corner supports: U2=0 on 2 corners of bottom face (z=0 side)
ch_support_left = apply_sliding_boundary!(K, f, dh, support_left, [2])
ch_support_right = apply_sliding_boundary!(K, f, dh, support_right, [2])

# Symmetry plane z=1.0: U3=0
ch_symmetry_z = apply_sliding_boundary!(K, f, dh, symmetry_z_nodes, [3])

# Symmetry plane x=1.0: U1=0
ch_symmetry_x = apply_sliding_boundary!(K, f, dh, symmetry_x_nodes, [1])

# Point load: F = [0, -1, 0] distributed on circular region
apply_force!(f, dh, collect(force_nodes), [0.0, -1.0, 0.0])
println("  ✓ Applied force: [0, -1, 0] N on $(length(force_nodes)) nodes")

# -----------------------------------------------------------------------------
# 9. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
tolerance_values = [0.16, 0.08, 0.04, 0.02, 0.01]

opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-9,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.4,
    max_iterations = 3000,
    tolerance = 0.01,
    filter_radius = 2.0,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 3000,
    export_path = results_dir,
    task_name = "3D_Michell-half",
    tolerance_checkpoints = tolerance_values,
)

println("\nOptimization parameters:")
println("  Task name: $(opt_params.task_name)")
println("  Target volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Convergence tolerance: $(opt_params.tolerance)")
println("  Filter radius: $(opt_params.filter_radius)")
println("  Tolerance checkpoints: $(tolerance_values)")

# -----------------------------------------------------------------------------
# 10. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING 3D MICHELL-TYPE BEAM (HALF) OPTIMIZATION")
println("Single run with tolerance checkpoint export")
println("="^80)

results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [PointLoad(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
    [ch_support_left, ch_support_right, ch_symmetry_z, ch_symmetry_x],
    opt_params,
)

# -----------------------------------------------------------------------------
# 11. EXPORT FINAL RESULTS
# -----------------------------------------------------------------------------
println("\nExporting final results...")
results_data = create_results_data(grid, dh, results)
export_results_vtu(results_data, joinpath(results_dir, "final"))

# -----------------------------------------------------------------------------
# 12. SUMMARY
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("OPTIMIZATION COMPLETED")
println("="^80)
println("\nFinal Results:")
println("  Compliance: $(round(results.compliance, digits=6))")
println("  Volume fraction: $(round(results.volume / calculate_volume(grid), digits=4))")
println("  Iterations: $(results.iterations)")
println("  Converged: $(results.converged)")

println("\nProblem Setup:")
println("  • Domain: 2.0 × 1.0 × 1.0")
println("  • Support left: $(length(support_left)) nodes, 1 corner at [0,0,0] (U2=0)")
println("  • Support right: $(length(support_right)) nodes, 1 corner at [2,0,0] (U2=0)")
println("  • Force: [0, -1, 0] N on circular region (r=0.1) at [1, 0, 0.5]")
println("  • Symmetry plane z=1.0: $(length(symmetry_z_nodes)) nodes (U3=0)")
println("  • Symmetry plane x=1.0: $(length(symmetry_x_nodes)) nodes (U1=0)")

println("\nTolerance Checkpoint Files:")
for tol in tolerance_values
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    println("  $(results_dir)/final_results_$(tol_str)tol.vtu  (change < $(tol))")
end

println("\nOutput files:")
println("  $(results_dir)/optimization_progress.csv  (iteration data)")
println("  $(results_dir)/optimization_summary.txt   (final summary)")
println("  $(results_dir)/final_results.vtu          (ParaView visualization)")
println("="^80)
