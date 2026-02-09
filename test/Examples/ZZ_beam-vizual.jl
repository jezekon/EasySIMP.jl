# =============================================================================
# BASIC CANTILEVER BEAM - SIMP Topology Optimization (6×4×1 mm, 1 mm elements)
# =============================================================================
# 
# Description:
#   Classic cantilever beam problem with fixed support on one end and 
#   point load on the free end. Small domain for testing purposes.
#
#
# Problem Visualization (side view):
#
#        Y ↑ ▓
#          | ▓
#       4  | ▓██████████████
#          | ▓█            █
#          | ▓█  DESIGN    █
#          | ▓█  DOMAIN    █
#          | ▓█  6×4×1 mm  █
#       0  | ▓██████████████
#          | ▓              ↓ F = 1.0 N (downward)
#          | ▓ ← Fixed support (all DOFs = 0)
#          └──────────────────→ X
#            0              6
#
#        (Z dimension: 0 to 1 mm, perpendicular to page)
#
#   Discretization: 6 × 4 × 1 = 24 hexahedral elements
#   Element size: 1 × 1 × 1 mm
#
# Boundary Conditions:
#   - Fixed support: Left face (x=0) - all DOFs constrained (U1=U2=U3=0)
#   - Point load: Right end (x=6, y=0, z=0.5) - F = [0, -1, 0] N
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra

# -----------------------------------------------------------------------------
# 1. MESH GENERATION
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(
    Hexahedron,
    (6, 4, 1),                      # Elements in X, Y, Z
    Vec((0.0, 0.0, 0.0)),           # Lower corner
    Vec((6.0, 4.0, 1.0)),           # Upper corner
)
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 200.0          # Young's modulus [MPa]
ν = 0.3             # Poisson's ratio
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

# Fixed support: left face (x = 0)
fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
println("  ✓ Fixed nodes: $(length(fixed_nodes))")

# Force application point: right end, bottom, Z-center
force_nodes = select_nodes_by_circle(grid, [6.0, 0.0, 0.5], [1.0, 0.0, 0.0], 0.1)
if isempty(force_nodes)
    # Fallback: find closest node
    target = [6.0, 0.0, 0.5]
    local min_dist = Inf
    local closest = 1
    for node_id = 1:getnnodes(grid)
        dist = norm(grid.nodes[node_id].x - target)
        if dist < min_dist
            min_dist = dist
            closest = node_id
        end
    end
    force_nodes = Set([closest])
end
println("  ✓ Force nodes: $(length(force_nodes))")

# Assemble and apply boundary conditions
assemble_stiffness_matrix_simp!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    fill(0.4, getncells(grid)),
)
ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
apply_force!(f, dh, collect(force_nodes), [0.0, -1.0, 0.0])

# -----------------------------------------------------------------------------
# 5. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
vf = 0.5
results_dir = "./results/ZZ_beam-vizual_6x4x1_vf$(vf)"
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-6,
    ν = ν,
    p = 3.0,
    volume_fraction = vf,
    max_iterations = 200,
    tolerance = 0.01,
    filter_radius = 1.5,            # 1.5× element edge length
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 50,
    export_path = results_dir,
)

println("\nOptimization parameters:")
println("  Volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")
println("  Number of elements: $(getncells(grid))")

# -----------------------------------------------------------------------------
# 6. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\nStarting optimization...\n")
results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [PointLoad(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
    [ch_fixed],
    opt_params,
)

# -----------------------------------------------------------------------------
# 7. EXPORT FINAL RESULTS
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
println("\nResults saved to: $results_dir")
println("="^80)
