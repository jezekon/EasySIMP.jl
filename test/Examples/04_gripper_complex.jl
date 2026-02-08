# =============================================================================
# COMPLEX GRIPPER - SIMP Topology Optimization
# =============================================================================
#
# Description:
#   Real-world complex optimization case demonstrating multiple loading 
#   conditions, symmetry constraints, and body forces. This gripper design
#   must support camera equipment and mounting legs under acceleration.
#
# Geometry:
#   - Imported mesh: stul14.vtu (complex 3D geometry)
#   - Symmetry: YZ plane at x = 0
#
# Boundary Conditions:
#   - Fixed support: Circular region at [0, 75, 115] with radius 16.11 mm
#   - Symmetry plane: x = 0 (fixed X displacement only)
#   - Load 1 (Legs): Distributed on plane z = -90, Force = 13 N downward
#   - Load 2 (Camera): Circular region z = 5, radius 21.5 mm, Force = 0.5 N
#   - Body force: 6 m/s² acceleration in Y direction
#
# Optimization Goal:
#   - Minimize compliance under multiple loads and acceleration
#   - Target volume fraction: 30%
#   - Material: Polymer (E = 2.4 GPa, ρ = 1.04 g/cm³)
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra

# -----------------------------------------------------------------------------
# 1. MESH IMPORT
# -----------------------------------------------------------------------------
println("Importing gripper mesh...")
mesh_path = "data/stul14.vtu"

if !isfile(mesh_path)
    error(
        "Mesh file not found: $mesh_path\n" *
        "Please ensure the mesh file exists before running this example.",
    )
end

grid = import_mesh(mesh_path)
println("  ✓ Imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 2.4e3          # Young's modulus [MPa] = [N/mm²]
ν = 0.35            # Poisson's ratio
ρ = 1.04e-6         # Density [kg/mm³] (1.04 g/cm³)
material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

println("\nMaterial properties (Polymer):")
println("  E₀ = $(E0/1e3) GPa")
println("  ν = $ν")
println("  ρ = $(ρ*1e9) g/cm³")

# -----------------------------------------------------------------------------
# 3. FEM SETUP
# -----------------------------------------------------------------------------
println("\nSetting up FEM problem...")
dh, cellvalues, K, f = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 4. BOUNDARY CONDITIONS - NODE SELECTION
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

# Fixed support: circular region on mounting surface
fixed_nodes = select_nodes_by_circle(
    grid,
    [0.0, 75.0, 115.0],      # Center point
    [0.0, -1.0, 0.0],        # Normal vector (pointing down)
    16.11,                    # Radius [mm]
    1e-3,                      # Tolerance
)
println("  ✓ Fixed support nodes: $(length(fixed_nodes))")

# Symmetry plane: YZ plane at x = 0
symmetry_nodes = select_nodes_by_plane(
    grid,
    [0.0, 0.0, 0.0],         # Point on plane
    [1.0, 0.0, 0.0],         # Normal vector (X direction)
    1e-3,                      # Tolerance
)
println("  ✓ Symmetry plane nodes: $(length(symmetry_nodes))")

# Load 1: Legs support - plane at z = -90
leg_nodes = select_nodes_by_plane(
    grid,
    [0.0, 0.0, -90.0],       # Point on plane
    [0.0, 0.0, 1.0],         # Normal vector (Z direction)
    1.0,                       # Tolerance (larger for robustness)
)
println("  ✓ Leg mounting nodes: $(length(leg_nodes))")

# Load 2: Camera support - circular region at z = 5
camera_nodes = select_nodes_by_circle(
    grid,
    [0.0, 0.0, 5.0],         # Center point
    [0.0, 0.0, 1.0],         # Normal vector (Z direction)
    21.5,                     # Radius [mm]
    1e-3,                      # Tolerance
)
println("  ✓ Camera mounting nodes: $(length(camera_nodes))")

# -----------------------------------------------------------------------------
# 5. FALLBACK FOR EMPTY NODE SETS
# -----------------------------------------------------------------------------
# Handle cases where geometric selection fails

if isempty(fixed_nodes)
    println("  ⚠ Warning: No fixed nodes found, using closest node to [0, 75, 115]")
    target = [0.0, 75.0, 115.0]
    local min_dist = Inf
    local closest = 1
    for node_id = 1:getnnodes(grid)
        dist = norm(grid.nodes[node_id].x - target)
        if dist < min_dist
            min_dist = dist
            closest = node_id
        end
    end
    fixed_nodes = Set([closest])
end

if isempty(symmetry_nodes)
    println("  ⚠ Warning: No symmetry nodes found, searching near x = 0 plane")
    symmetry_nodes = Set{Int}()
    for node_id = 1:getnnodes(grid)
        node_coord = grid.nodes[node_id].x
        if abs(node_coord[1]) < 2.0  # Within 2mm of x=0
            push!(symmetry_nodes, node_id)
        end
    end
    println("  ✓ Found $(length(symmetry_nodes)) nodes near x = 0")
end

if isempty(leg_nodes)
    println("  ⚠ Warning: No leg nodes found, searching near z = -90")
    leg_nodes = Set{Int}()
    for node_id = 1:getnnodes(grid)
        node_coord = grid.nodes[node_id].x
        if length(node_coord) >= 3 && abs(node_coord[3] - (-90.0)) < 5.0
            push!(leg_nodes, node_id)
        end
    end
    println("  ✓ Found $(length(leg_nodes)) nodes near z = -90")
end

if isempty(camera_nodes)
    println("  ⚠ Warning: No camera nodes found, using closest node to [0, 0, 5]")
    target = [0.0, 0.0, 5.0]
    local min_dist = Inf
    local closest = 1
    for node_id = 1:getnnodes(grid)
        dist = norm(grid.nodes[node_id].x - target)
        if dist < min_dist
            min_dist = dist
            closest = node_id
        end
    end
    camera_nodes = Set([closest])
end

# -----------------------------------------------------------------------------
# 6. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
println("\nExporting boundary conditions for ParaView inspection...")
results_dir = "./results/04_gripper_complex"
mkpath(results_dir)

all_force_nodes = union(leg_nodes, camera_nodes)
all_constraint_nodes = union(fixed_nodes, symmetry_nodes)
export_boundary_conditions(
    grid,
    dh,
    all_constraint_nodes,
    all_force_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 7. APPLY BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("\nApplying boundary conditions...")

assemble_stiffness_matrix_simp!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    fill(0.3, getncells(grid)),
)

# Fixed support: all DOFs constrained
ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
println("  ✓ Fixed support: $(length(fixed_nodes)) nodes (all DOFs)")

# Symmetry: X direction only
ch_symmetry = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
println("  ✓ Symmetry plane: $(length(symmetry_nodes)) nodes (X-direction only)")

# -----------------------------------------------------------------------------
# 8. APPLY FORCES
# -----------------------------------------------------------------------------
println("\nApplying forces...")

# Force 1: Legs - 13 N total downward (Z direction)
# F = π(14² - 7.5²) × 3 × 0.00985 ≈ 13 N
apply_force!(f, dh, collect(leg_nodes), [0.0, 0.0, -13000.0])  # [mN]
println("  ✓ Leg force: 13 N downward ($(length(leg_nodes)) nodes)")

# Force 2: Camera - 0.5 N total downward (Z direction)
# F = π(21.5² - 17²) × 0.001852 × 0.5 ≈ 0.5 N
apply_force!(f, dh, collect(camera_nodes), [0.0, 0.0, -500.0])  # [mN]
println("  ✓ Camera force: 0.5 N downward ($(length(camera_nodes)) nodes)")

# Body force: 6 m/s² acceleration in Y direction
acceleration_vector = [0.0, 6000.0, 0.0]  # [mm/s²]
acceleration_data = (acceleration_vector, ρ)
println("  ✓ Acceleration: 6 m/s² in Y direction (body force)")

# -----------------------------------------------------------------------------
# 9. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-6,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.3,
    max_iterations = 200,
    tolerance = 0.01,
    filter_radius = 1.5,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 5,
    export_path = results_dir,
)

println("\nOptimization parameters:")
println("  Target volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius) × element_size")
println("  Move limit: $(opt_params.move_limit)")
println("  Convergence tolerance: $(opt_params.tolerance)")

# -----------------------------------------------------------------------------
# 10. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING COMPLEX GRIPPER OPTIMIZATION")
println("="^80)
println("Multiple loads + Symmetry + Body forces")
println()

forces_list = [
    PointLoad(dh, collect(leg_nodes), [0.0, 0.0, -13000.0]),      # Legs: 13 N
    PointLoad(dh, collect(camera_nodes), [0.0, 0.0, -500.0]),      # Camera: 0.5 N
]

results = simp_optimize(
    grid,
    dh,
    cellvalues,
    forces_list,
    [ch_fixed, ch_symmetry],
    opt_params,
    acceleration_data,
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
println("  Compliance: $(round(results.compliance, digits=4))")
println("  Volume fraction: $(round(results.volume / calculate_volume(grid), digits=4))")
println("  Iterations: $(results.iterations)")
println("  Converged: $(results.converged)")

println("\nLoading Conditions:")
println("  • Leg support: 13 N distributed load")
println("  • Camera mount: 0.5 N distributed load")
println("  • Body acceleration: 6 m/s² in Y direction")
println("  • Fixed support: Circular region (r = 16.11 mm)")
println("  • Symmetry: YZ plane at x = 0")

println("\nResults Location:")
println("  $results_dir/")
println("  ├── boundary_conditions.vtu  (BC visualization)")
println("  ├── iter_XXXX_results.vtu    (intermediate results)")
println("  └── final_results.vtu         (final optimized design)")

println("\n" * "="^80)
println("To visualize in ParaView:")
println("  1. Load boundary_conditions.vtu to verify BC placement")
println("  2. Load iter_*.vtu or final_results.vtu for density field")
println("  3. Apply 'Threshold' filter: density > 0.3 for solid regions")
println("="^80)
