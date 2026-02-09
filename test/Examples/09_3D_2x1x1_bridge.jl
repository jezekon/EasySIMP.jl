# =============================================================================
# 3D BRIDGE WITH SYMMETRY - SIMP Topology Optimization
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
#        (Z dimension: 0 to 1.0, perpendicular to page)
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

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
const TOLERANCE = 0.08          # Convergence tolerance
const FILTER_RADIUS = 2.0      # Filter radius (× element size)

# Build results directory name from tolerance and filter radius
tol_str = replace(@sprintf("%.2f", TOLERANCE), "." => "")[(end-1):end]  # e.g. "08"
results_dir = "./results/09_3D_2x1x1_Bridge_$(tol_str)tol_r$(FILTER_RADIUS)"

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

# Create SurfaceTractionLoad
pressure_load = SurfaceTractionLoad(dh, grid, top_face_nodes, traction_fn)

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
ch_support = apply_sliding_boundary!(K, f, dh, support_nodes, [2])  # Fix Y only
ch_zfix = apply_sliding_boundary!(K, f, dh, zfix_nodes, [3])        # Fix Z only

# -----------------------------------------------------------------------------
# 6. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
mkpath(results_dir)

opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-6,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.4,
    max_iterations = 3000,
    tolerance = TOLERANCE,
    filter_radius = FILTER_RADIUS,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 3000,
    export_path = results_dir,
    task_name = "3D_Bridge_2x1x1",
)

println("\nOptimization parameters:")
println("  Task name: $(opt_params.task_name)")
println("  Volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 7. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
export_boundary_conditions(
    grid,
    dh,
    union(symmetry_nodes, support_nodes, zfix_nodes),
    top_face_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 8. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\nStarting optimization...\n")
results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [pressure_load],
    [ch_symmetry, ch_support, ch_zfix],
    opt_params,
)

# -----------------------------------------------------------------------------
# 9. EXPORT FINAL RESULTS
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

# Single thread computation:
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia -t 1 --project=. test/Examples/09_3D_2x1x1_bridge.jl
