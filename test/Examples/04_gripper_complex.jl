# =============================================================================
# COMPLEX GRIPPER - SIMP Topology Optimization
# =============================================================================
#
# Description:
#   Real-world complex optimization case demonstrating multiple loading 
#   conditions, symmetry constraints, and body forces. This gripper design
#   must support camera equipment and mounting legs under acceleration.
#   The script checks the result against a reference run.
#
# Units: N, mm, MPa, t (energy in N·mm = mJ)
#
# Geometry:
#   - Imported mesh: stul14.vtu (complex 3D geometry)
#   - Symmetry: YZ plane at x = 0
#
# Boundary Conditions:
#   - Fixed support: Circular region at [0, 75, 115] with radius 16.11 mm
#   - Symmetry plane: x = 0 (fixed X displacement only)
#   - Load 1 (Legs): Uniform traction on plane z = -90, total force 13 N downward
#   - Load 2 (Camera): Uniform traction on circular region z = 5, radius 21.5 mm,
#     total force 0.5 N downward
#   - Body force: 6 m/s² acceleration in Y direction, acting on every cell in
#     proportion to its density
#
# Optimization Goal:
#   - Minimize energy under multiple loads and acceleration
#   - Target volume fraction: 30%
#   - Material: Polymer (E = 2.4 GPa, ρ = 1.04 g/cm³)
#
# The body force makes the load design-dependent. Two options are needed for it:
#   - rho_lin = 0.1: linear stiffness below ρ = 0.1, so that near-void cells carry
#     their own weight (Bruyneel & Duysinx 2005)
#   - adaptive_move = true: per-element move limits; with a fixed move limit the OC
#     update oscillates with period two and does not converge
#
# Reference run (Julia 1.10.10, 1 thread, 1 BLAS thread):
#   116 iterations, energy U = 0.11597536895754954 N·mm
# This script: U = 0.11597536895867937 N·mm (facets summed in another order).
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra
using Test

# One BLAS thread as in the reference run: the summation order of the solver,
# and with it the last bits of every iteration, depends on the number of threads
LinearAlgebra.BLAS.set_num_threads(1)

# -----------------------------------------------------------------------------
# 1. MESH IMPORT
# -----------------------------------------------------------------------------
println("Importing gripper mesh...")
mesh_path = joinpath(@__DIR__, "..", "..", "data", "stul14.vtu")

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
ρ = 1.04e-9         # Density [t/mm³] (1.04 g/cm³)

println("\nMaterial properties (Polymer):")
println("  E₀ = $(E0/1e3) GPa")
println("  ν = $ν")
println("  ρ = $(round(ρ * 1e9, digits = 3)) g/cm³")

# -----------------------------------------------------------------------------
# 3. FEM SETUP
# -----------------------------------------------------------------------------
println("\nSetting up FEM problem...")
dh, cellvalues = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 4. BOUNDARY CONDITIONS - NODE SELECTION
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")
tol = 1e-6                    # Coordinate tolerance [mm]

# Fixed support: circular region on mounting surface
fixed_nodes = select_nodes_by_circle(
    grid,
    [0.0, 75.0, 115.0],      # Center point
    [0.0, -1.0, 0.0],        # Normal vector (pointing down)
    16.11,                    # Radius [mm]
    tol,                      # Tolerance
)
println("  ✓ Fixed support nodes: $(length(fixed_nodes))")

# Symmetry plane: YZ plane at x = 0
symmetry_nodes = select_nodes_by_plane(
    grid,
    [0.0, 0.0, 0.0],         # Point on plane
    [1.0, 0.0, 0.0],         # Normal vector (X direction)
    tol,                      # Tolerance
)
println("  ✓ Symmetry plane nodes: $(length(symmetry_nodes))")

# Load 1: Legs support - plane at z = -90
leg_nodes = select_nodes_by_plane(
    grid,
    [0.0, 0.0, -90.0],       # Point on plane
    [0.0, 0.0, 1.0],         # Normal vector (Z direction)
    tol,                      # Tolerance
)
println("  ✓ Leg mounting nodes: $(length(leg_nodes))")

# Load 2: Camera support - circular region at z = 5
camera_nodes = select_nodes_by_circle(
    grid,
    [0.0, 0.0, 5.0],         # Center point
    [0.0, 0.0, 1.0],         # Normal vector (Z direction)
    21.5,                     # Radius [mm]
    tol,                      # Tolerance
)
println("  ✓ Camera mounting nodes: $(length(camera_nodes))")

# -----------------------------------------------------------------------------
# 5. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
println("\nExporting boundary conditions for ParaView inspection...")
results_dir = "./results/04_gripper_complex"
mkpath(results_dir)

all_force_nodes = union(leg_nodes, camera_nodes)
all_constraint_nodes = union(fixed_nodes, symmetry_nodes)
export_boundary_conditions(
    grid,
    all_constraint_nodes,
    all_force_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 6. APPLY BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("\nApplying boundary conditions...")

# Fixed support: all DOFs constrained
ch_fixed = create_constraint_handler(dh, fixed_nodes)
println("  ✓ Fixed support: $(length(fixed_nodes)) nodes (all DOFs)")

# Symmetry: X direction only
ch_symmetry = create_constraint_handler(dh, symmetry_nodes, [1])
println("  ✓ Symmetry plane: $(length(symmetry_nodes)) nodes (X-direction only)")

# -----------------------------------------------------------------------------
# 7. APPLY FORCES
# -----------------------------------------------------------------------------
println("\nApplying forces...")

# Force 1: Legs - 13 N total downward (Z direction), uniform traction
# F = π(14² - 7.5²) × 3 × 0.00985 ≈ 13 N
leg_load = SurfaceTractionLoad(dh, grid, leg_nodes, [0.0, 0.0, -13.0])  # [N]
println("  ✓ Leg force: 13 N downward ($(length(leg_load.boundary_facets)) facets)")

# Force 2: Camera - 0.5 N total downward (Z direction), uniform traction
# F = π(21.5² - 17²) × 0.001852 × 0.5 ≈ 0.5 N
camera_load = SurfaceTractionLoad(dh, grid, camera_nodes, [0.0, 0.0, -0.5])  # [N]
println("  ✓ Camera force: 0.5 N downward ($(length(camera_load.boundary_facets)) facets)")

# Body force: 6 m/s² acceleration in Y direction
acceleration_vector = [0.0, 6000.0, 0.0]  # [mm/s²]
acceleration_data = (acceleration_vector, ρ)
println("  ✓ Acceleration: 6 m/s² in Y direction (body force)")

# -----------------------------------------------------------------------------
# 8. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-9 * E0,
    ν = ν,
    p = 3.0,
    rho_lin = 0.1,                 # Linear stiffness below ρ = 0.1 (body force)
    volume_fraction = 0.3,
    max_iterations = 5000,         # Stop by the tolerance only
    tolerance = 0.01,
    filter_radius = 1.5,           # × mean element size (3.939 mm)
    move_limit = 0.2,
    damping = 0.5,
    adaptive_move = true,          # Per-element move limits (body force)
    use_cache = true,
    export_interval = 0,           # No intermediate export
    export_path = results_dir,
)

println("\nOptimization parameters:")
println("  Target volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius) × element_size")
println("  Move limit: $(opt_params.move_limit)")
println("  Convergence tolerance: $(opt_params.tolerance)")

# -----------------------------------------------------------------------------
# 9. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING COMPLEX GRIPPER OPTIMIZATION")
println("="^80)
println("Multiple loads + Symmetry + Body forces")
println()

forces_list = [
    leg_load,         # Legs: 13 N
    camera_load,      # Camera: 0.5 N
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
# 10. EXPORT FINAL RESULTS
# -----------------------------------------------------------------------------
println("\nExporting final results...")
results_data = create_results_data(grid, dh, cellvalues, results)
export_results_vtu(results_data, joinpath(results_dir, "final"))

# -----------------------------------------------------------------------------
# 11. SUMMARY
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("OPTIMIZATION COMPLETED")
println("="^80)
println("\nFinal Results:")
println("  Energy: $(results.energy) N·mm")
println("  Volume fraction: $(round(results.volume / calculate_volume(grid), digits=4))")
println("  Iterations: $(results.iterations)")
println("  Converged: $(results.converged)")

println("\nLoading Conditions:")
println("  • Leg support: 13 N uniform traction")
println("  • Camera mount: 0.5 N uniform traction")
println("  • Body acceleration: 6 m/s² in Y direction")
println("  • Fixed support: Circular region (r = 16.11 mm)")
println("  • Symmetry: YZ plane at x = 0")

println("\nResults Location:")
println("  $results_dir/")
println("  ├── boundary_conditions.vtu  (BC visualization)")
println("  └── final.vtu                (final optimized design)")

println("\n" * "="^80)
println("To visualize in ParaView:")
println("  1. Load boundary_conditions.vtu to verify BC placement")
println("  2. Load final.vtu for density field")
println("  3. Apply 'Threshold' filter: density > 0.3 for solid regions")
println("="^80)

# -----------------------------------------------------------------------------
# 12. COMPARISON WITH THE REFERENCE RUN
# -----------------------------------------------------------------------------
@testset "Gripper vs. reference run" begin
    @test results.converged
    @test results.iterations == 116
    @test isapprox(results.energy, 0.11597536895754954; rtol = 1e-6)
end
