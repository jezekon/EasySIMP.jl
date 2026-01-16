# =============================================================================
# 3D WHEEL - SIMP Topology Optimization
# =============================================================================
#
# Description:
#   Hollow cylinder (wheel) under tangential traction load.
#   Based on benchmark from Wegert et al. (2025), adapted for SIMP method.
#
# Geometry (top view, looking down Z-axis):
#
#              Y ↑
#                |       Arc fixations (5x on outer rim)
#           _____|_____
#          /  ·  |  ·  \     r_outer = 1.0
#         / ·    |    · \
#        |   ____|____   |
#        |  /    |    \  |   r_inner = 0.1
#        | |     O-----|--→ X
#        |  \________/  |    Tangential traction g = 100(-y, x, 0)
#         \ ·        · /     on inner cylinder surface
#          \_·____|__·_/
#                |
#
#        Z dimension: -0.15 to +0.15 (thickness = 0.3)
#
# Boundary Conditions:
#   - Fixed support: 5 arcs (15° each) on outer cylinder at r = 1.0
#     Angles (CCW from +X): 15°-30°, 82.5°-97.5°, 150°-165°, 217.5°-232.5°, 307.5°-322.5°
#   - Tangential load: Inner cylinder (r = 0.1), g(x,y,z) = 100*(-y, x, 0)
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra

# -----------------------------------------------------------------------------
# 1. GEOMETRY PARAMETERS
# -----------------------------------------------------------------------------
const R_INNER = 0.1      # Inner radius (hub)
const R_OUTER = 1.0      # Outer radius (rim)
const THICKNESS = 0.3    # Z-dimension (-0.15 to +0.15)

# -----------------------------------------------------------------------------
# 2. MESH IMPORT
# -----------------------------------------------------------------------------
println("Importing wheel mesh...")
mesh_path = "data/Wheel_3d_coarse.msh"

if !isfile(mesh_path)
    error(
        "Mesh file not found: $mesh_path\nGenerate mesh using Gmsh with Wheel_3d.geo file.",
    )
end

grid = import_mesh(mesh_path)
println("  ✓ Imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 3. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 1.0       # Young's modulus (normalized)
ν = 0.3        # Poisson's ratio
material_model = create_simp_material_model(E0, ν, 1e-9, 3.0)

# -----------------------------------------------------------------------------
# 4. FEM SETUP
# -----------------------------------------------------------------------------
println("Setting up FEM problem...")
dh, cellvalues, K, f = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 5. BOUNDARY CONDITIONS - FIXED ARCS ON OUTER CYLINDER
# -----------------------------------------------------------------------------
println("Applying boundary conditions...")

# 5 fixation arcs on outer cylinder (each spans 15°)
# Angles in standard convention (CCW from +X axis)
arc_angles = [
    (15.0, 30.0),       # Arc 1: upper right quadrant
    (82.5, 97.5),       # Arc 2: upper middle (near +Y)
    (150.0, 165.0),     # Arc 3: upper left quadrant
    (217.5, 232.5),     # Arc 4: lower left quadrant
    (307.5, 322.5),     # Arc 5: lower right quadrant
]

fixed_nodes = Set{Int}()
axis_point = [0.0, 0.0, 0.0]
axis_normal = [0.0, 0.0, 1.0]
arc_tolerance = 0.01   # Tolerance comparable to mesh element size

for (a_start, a_end) in arc_angles
    arc_nodes = select_nodes_by_arc(
        grid,
        axis_point,
        axis_normal,
        R_OUTER,
        a_start,
        a_end,
        arc_tolerance,
    )
    union!(fixed_nodes, arc_nodes)
end
println("  ✓ Fixed nodes (5 arcs): $(length(fixed_nodes))")

# -----------------------------------------------------------------------------
# 6. NEUMANN BC - TANGENTIAL TRACTION ON INNER CYLINDER
# -----------------------------------------------------------------------------
inner_nodes =
    select_nodes_by_cylinder(grid, axis_point, axis_normal, R_INNER, arc_tolerance)
println("  ✓ Inner cylinder nodes: $(length(inner_nodes))")

# Tangential traction function: g(x,y,z) = 100*(-y, x, 0)
traction_magnitude = 100.0
g(x, y, z) = [traction_magnitude * (-y), traction_magnitude * x, 0.0]

# -----------------------------------------------------------------------------
# 7. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
results_dir = "./results/07_wheel"
mkpath(results_dir)

opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-9,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.3,
    max_iterations = 300,
    tolerance = 0.08,
    filter_radius = 1.4,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 10,
    export_path = results_dir,
    task_name = "Wheel_3D",
)

println("\nOptimization parameters:")
println("  Task name: $(opt_params.task_name)")
println("  Volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 8. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
export_boundary_conditions(
    grid,
    dh,
    fixed_nodes,
    inner_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 9. ASSEMBLE AND APPLY BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
assemble_stiffness_matrix_simp!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    fill(0.3, getncells(grid)),
)

ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)

# Apply tangential traction using surface integration
inner_facets = get_boundary_facets(grid, inner_nodes)
apply_surface_traction!(f, dh, grid, inner_facets, g)

# Create load condition for optimization loop
traction_load = SurfaceTractionLoad(dh, grid, inner_nodes, g)

# -----------------------------------------------------------------------------
# 10. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\nStarting optimization...\n")
results = simp_optimize(grid, dh, cellvalues, [traction_load], [ch_fixed], opt_params)

# -----------------------------------------------------------------------------
# 11. EXPORT FINAL RESULTS
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
