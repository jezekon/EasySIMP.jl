# =============================================================================
# CANTILEVER BEAM WITH BODY FORCES (ACCELERATION)
# =============================================================================
#
# Description:
#   Demonstrates topology optimization with body forces due to acceleration.
#   This is useful for design under inertial loads (e.g., vibration, impact,
#   or gravitational loading).
#
# Problem Visualization (side view):
#
#        Y ↑ ○ F = 1000 N (point force)
#          | ○↓
#      20  | ○████████████████████████████████████████████████
#          | ○█  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓                             █
#          | ○█    DESIGN DOMAIN                             █
#          | ○█    60 × 20 × 4 mm                            █
#          | ○█  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓  (Body force: 6 m/s²)       █
#       0  | ○████████████████████████████████████████████████
#          | ○                                               ━━ ← Y-support (U2=0)
#          | ○ ← Sliding support (U1=0, free in Y,Z)
#          └─────────────────────────────────────────────────────→ X
#            0                                               60
#
#        (Z dimension: 0 to 4 mm, perpendicular to page)
#
# Boundary Conditions:
#   - Sliding support: Left face (x=0) - U1=0 only (can slide in Y,Z)
#   - Point support: Right end (x=60, y=0, z=2) - U2=0 only
#   - Point load: Top left (x=0, y=20, z=2) - F = [0, -1000, 0] N
#   - Body force: 6 m/s² acceleration in Y direction (distributed)
#
# Loading:
#   - Point force: Concentrated 1000 N at top left corner
#   - Body force: Distributed inertial load f = ρ × a (density-dependent)
#                 Acts on entire volume, scales with local density
#
# Optimization Goal:
#   - Minimize energy under combined point and body forces
#   - Target volume fraction: 40%
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra

# -----------------------------------------------------------------------------
# 1. MESH GENERATION
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(Hexahedron, (60, 20, 4), Vec((0.0, 0.0, 0.0)), Vec((60.0, 20.0, 4.0)))
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 2.4e3          # Young's modulus [MPa] = [N/mm²]
ν = 0.35            # Poisson's ratio
ρ = 1.04e-6         # Density [kg/mm³]
material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

println("Material properties:")
println("  E₀ = $(E0/1e3) GPa")
println("  ν = $ν")
println("  ρ = $ρ kg/mm³")

# -----------------------------------------------------------------------------
# 3. FEM SETUP
# -----------------------------------------------------------------------------
println("\nSetting up FEM problem...")
dh, cellvalues, K, f = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 4. BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("Applying boundary conditions...")

# Sliding constraint: left face (x = 0) - fixed only in X
sliding_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
println("  ✓ Sliding nodes: $(length(sliding_nodes))")

# Point support: right end - fixed in Y
support_nodes = select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [0.0, 1.0, 0.0], 0.5)
if isempty(support_nodes)
    target = [60.0, 0.0, 2.0]
    local min_dist = Inf
    local closest = 1
    for node_id = 1:getnnodes(grid)
        dist = norm(grid.nodes[node_id].x - target)
        if dist < min_dist
            min_dist = dist
            closest = node_id
        end
    end
    support_nodes = Set([closest])
end
println("  ✓ Support nodes: $(length(support_nodes))")

# Force application: top left
force_nodes = select_nodes_by_circle(grid, [0.0, 20.0, 2.0], [1.0, 0.0, 0.0], 1.0)
if isempty(force_nodes)
    target = [0.0, 20.0, 2.0]
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

ch_sliding = apply_sliding_boundary!(K, f, dh, sliding_nodes, [1])
ch_support = apply_sliding_boundary!(K, f, dh, support_nodes, [2])

# Apply point force
apply_force!(f, dh, collect(force_nodes), [0.0, -1000.0, 0.0])  # 1000 N downward
println("  ✓ Point force: 1000 N")

# -----------------------------------------------------------------------------
# 5. ACCELERATION (BODY FORCE)
# -----------------------------------------------------------------------------
acceleration_vector = [0.0, 6000.0, 0.0]  # 6 m/s² = 6000 mm/s² in Y direction
acceleration_data = (acceleration_vector, ρ)

println("\nBody force configuration:")
println("  Acceleration: $(acceleration_vector[2]/1000) m/s² in Y direction")
println("  Mass = ρ × volume (varies with density distribution)")

# -----------------------------------------------------------------------------
# 6. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
results_dir = "./results/03_with_acceleration"
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-6,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.4,
    max_iterations = 100,
    tolerance = 0.01,
    filter_radius = 2.0,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 5,
    export_path = results_dir,
)

println("\nOptimization parameters:")
println("  Volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 7. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\nStarting optimization with combined point and body forces...\n")
results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [PointLoad(dh, collect(force_nodes), [0.0, -1000.0, 0.0])],
    [ch_sliding, ch_support],
    opt_params,
    acceleration_data,  # Add acceleration
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
println("Final energy: $(results.energy)")
println("Final volume fraction: $(results.volume / calculate_volume(grid))")
println("Iterations: $(results.iterations)")
println("Converged: $(results.converged)")
println("\nResults saved to: $results_dir")
println("\nNote: Body forces scale with density distribution")
println("="^80)
