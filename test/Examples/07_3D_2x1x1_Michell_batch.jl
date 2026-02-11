# =============================================================================
# 3D MICHELL-TYPE BEAM - SIMP Topology Optimization
# Single run with tolerance checkpoint export
# =============================================================================
#
# Description:
#   Michell-type beam problem with four corner supports on the bottom face
#   and a central point load. Demonstrates multi-tolerance checkpoint export
#   from a single optimization run.
#
# Problem Visualization (side view, XY plane at z=0.5):
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
#       (corners)     at [1,0,0.5], [0,-1,0]          (corners)
#          └─────────────────────────────────────────────────→ X
#          0                   1.0                        2.0
#
#        (Z dimension: 0 to 1.0, perpendicular to page)
#        4 corner supports: 3×3 elements each at (x=0,z=0), (x=0,z=1),
#                           (x=2,z=0), (x=2,z=1)
#
# Boundary Conditions:
#   - Support: 4 corners on bottom face (y=0), 3×3 elements each - U2=0
#   - Local constraint at base near [0,0,0] (y=0, x<=0.15, z=0): U3=0
#   - Local constraint at base near [0,0,0] (y=0, z<=0.15, x=0): U1=0
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
# 4. BOUNDARY CONDITIONS - FOUR CORNER SUPPORTS ON BOTTOM FACE
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

# Corner size: 3 elements × 0.05 = 0.15
corner_size = 0.15

# Support left: 2 corners at x=0 (z=0 and z=zmax)
support_left = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]
    if abs(y) < eps() && x <= corner_size + eps()
        # Bottom-left corner (x≈0, z≈0)
        in_corner1 = z <= corner_size + eps()
        # Top-left corner (x≈0, z≈zmax)
        in_corner2 = z >= zmax - corner_size - eps()
        if in_corner1 || in_corner2
            push!(support_left, node_id)
        end
    end
end
println("  ✓ Support left (2 corners, 3×3 elem): $(length(support_left)) nodes")

# Support right: 2 corners at x=xmax (z=0 and z=zmax)
support_right = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]
    if abs(y) < eps() && x >= xmax - corner_size - eps()
        # Bottom-right corner (x≈xmax, z≈0)
        in_corner1 = z <= corner_size + eps()
        # Top-right corner (x≈xmax, z≈zmax)
        in_corner2 = z >= zmax - corner_size - eps()
        if in_corner1 || in_corner2
            push!(support_right, node_id)
        end
    end
end
println("  ✓ Support right (2 corners, 3×3 elem): $(length(support_right)) nodes")

# -----------------------------------------------------------------------------
# 5. FORCE REGION AND LOCAL CONSTRAINTS
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

# Local constraint near origin: y=0, z=0, x in <0, 0.15> → U3=0
local_z_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]
    if abs(y) < eps() && x <= corner_size + eps() && abs(z) < eps()
        push!(local_z_nodes, node_id)
    end
end
println("  ✓ Local constraint z=0 edge (U3=0): $(length(local_z_nodes)) nodes")

# Local constraint near origin: y=0, x=0, z in <0, 0.15> → U1=0
local_x_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]
    if abs(y) < eps() && z <= corner_size + eps() && abs(x) < eps()
        push!(local_x_nodes, node_id)
    end
end
println("  ✓ Local constraint x=0 edge (U1=0): $(length(local_x_nodes)) nodes")

# -----------------------------------------------------------------------------
# 6. RESULTS DIRECTORY
# -----------------------------------------------------------------------------
results_dir = "./results/07_michell_type"
mkpath(results_dir)

# -----------------------------------------------------------------------------
# 7. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
all_support_nodes = union(support_left, support_right, local_z_nodes, local_x_nodes)

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

# Corner supports: U2=0 on 4 corners of bottom face
ch_support_left = apply_sliding_boundary!(K, f, dh, support_left, [2])
ch_support_right = apply_sliding_boundary!(K, f, dh, support_right, [2])

# Local constraint z=0 edge: U3=0
ch_local_z = apply_sliding_boundary!(K, f, dh, local_z_nodes, [3])

# Local constraint x=0 edge: U1=0
ch_local_x = apply_sliding_boundary!(K, f, dh, local_x_nodes, [1])

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
    task_name = "3D_Michell_Type",
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
println("STARTING 3D MICHELL-TYPE BEAM OPTIMIZATION")
println("Single run with tolerance checkpoint export")
println("="^80)

results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [PointLoad(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
    [ch_support_left, ch_support_right, ch_local_z, ch_local_x],
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
println("  • Support left: $(length(support_left)) nodes, 2 corners at x=0 (U2=0)")
println("  • Support right: $(length(support_right)) nodes, 2 corners at x=2 (U2=0)")
println("  • Force: [0, -1, 0] N on circular region (r=0.1) at [1, 0, 0.5]")
println(
    "  • Symmetry plane z=0.5: REMOVED → local constraint z=0 edge: $(length(local_z_nodes)) nodes (U3=0)",
)
println(
    "  • Symmetry plane x=1.0: REMOVED → local constraint x=0 edge: $(length(local_x_nodes)) nodes (U1=0)",
)

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
