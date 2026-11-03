# =============================================================================
# BASIC CANTILEVER BEAM - SIMP Topology Optimization
# =============================================================================
# 
# Description:
#   Classic cantilever beam problem with fixed support on one end and 
#   a load at the bottom of the free end. Demonstrates basic SIMP optimization
#   setup. The script checks the result against a reference run.
#
#
# Problem Visualization (side view):
#
#        Y ↑ ▓
#          | ▓
#      20  | ▓████████████████████████████████████████████████
#          | ▓█                                              █
#          | ▓█         DESIGN DOMAIN                        █
#          | ▓█         60 × 20 × 4 mm                       █
#          | ▓█                                              █
#       0  | ▓████████████████████████████████████████████████
#          | ▓                                               ↓ F = 1.0 N (downward, y ≤ 1)
#          | ▓ ← Fixed support (all DOFs = 0)
#          └─────────────────────────────────────────────────────→ X
#            0                                               60
#
#        (Z dimension: 0 to 4 mm, perpendicular to page)
#
# Boundary Conditions:
#   - Fixed support: Left face (x=0) - all DOFs constrained (U1=U2=U3=0)
#   - Surface load: Strip of the right face (x=60, 0 ≤ y ≤ 1, 0 ≤ z ≤ 4, area 4 mm²),
#     uniform traction with total force F = [0, -1, 0] N
#
# Optimization Goal:
#   - Minimize energy (maximize stiffness)
#   - Target volume fraction: 40%
#   - Material will be removed from low-stress regions
#
# Reference run (Julia 1.10.10, 1 thread, 1 BLAS thread):
#   77 iterations, energy U = 30.13042187926461 N·mm
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
# 1. MESH GENERATION
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(
    Hexahedron,
    (60, 20, 4),                   # Elements in X, Y, Z
    Vec((0.0, 0.0, 0.0)),          # Lower corner
    Vec((60.0, 20.0, 4.0)),        # Upper corner
)
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 1.0            # Young's modulus [MPa]
ν = 0.3             # Poisson's ratio

# -----------------------------------------------------------------------------
# 3. FEM SETUP
# -----------------------------------------------------------------------------
println("Setting up FEM problem...")
dh, cellvalues = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 4. BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("Applying boundary conditions...")

tol = 1e-6          # Coordinate tolerance [mm]

# Fixed support: left face (x = 0)
fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], tol)
println("  ✓ Fixed nodes: $(length(fixed_nodes))")

# Force application: strip y <= 1 of the right face (x = 60)
force_nodes = select_nodes(grid, x -> abs(x[1] - 60.0) < tol && x[2] <= 1.0 + tol)
println("  ✓ Force nodes: $(length(force_nodes))")

ch_fixed = create_constraint_handler(dh, fixed_nodes)

# Total force 1 N in -y as a uniform traction on the boundary facets of the strip
load = SurfaceTractionLoad(dh, grid, force_nodes, [0.0, -1.0, 0.0])

# -----------------------------------------------------------------------------
# 5. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
results_dir = "./results/01_basic_cantilever"
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-9 * E0,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.4,
    max_iterations = 5000,         # Stop by the tolerance only
    tolerance = 0.01,
    filter_radius = 1.5,           # × element size (1 mm)
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 0,           # No intermediate export
    export_path = results_dir,
)

println("\nOptimization parameters:")
println("  Volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 6. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\nStarting optimization...\n")
results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [load],
    [ch_fixed],
    opt_params,
)

# -----------------------------------------------------------------------------
# 7. EXPORT FINAL RESULTS
# -----------------------------------------------------------------------------
println("\nExporting final results...")
results_data = create_results_data(grid, dh, cellvalues, results)
export_results_vtu(results_data, joinpath(results_dir, "final"))

println("\n" * "="^80)
println("OPTIMIZATION COMPLETED")
println("="^80)
println("Final energy: $(results.energy)")
println("Final volume fraction: $(results.volume / calculate_volume(grid))")
println("Iterations: $(results.iterations)")
println("Converged: $(results.converged)")
println("\nResults saved to: $results_dir")
println("="^80)

# -----------------------------------------------------------------------------
# 8. COMPARISON WITH THE REFERENCE RUN
# -----------------------------------------------------------------------------
@testset "Cantilever vs. reference run" begin
    @test results.converged
    @test results.iterations == 77
    @test isapprox(results.energy, 30.13042187926461; rtol = 1e-6)
end
