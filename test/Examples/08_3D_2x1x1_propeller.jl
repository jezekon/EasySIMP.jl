# =============================================================================
# 3D BEAM WITH PROPELLER LOADING - SIMP Topology Optimization
# =============================================================================
#
# Description:
#   3D beam optimization with fixed support on one end and 4 forces
#   forming a "propeller" (torsional) pattern on the free end.
#   Forces are rotationally symmetric, creating a twisting moment.
#
# Geometry:
#   - Domain: 2.0 × 1.0 × 1.0 (X × Y × Z)
#   - Mesh: 40 × 20 × 20 hexahedral elements (element size 0.05)
#   - Fixed support: Entire YZ face at x = 0 (all DOFs constrained)
#   - Load: 4 forces on the free end (x ≈ 2.0), each on a 3×3 element patch
#
# Cross-section at x = 2.0 (view from +X axis):
#
#       z=1  ┌─F4→─────────F1↓─┐
#            │                  │
#            │        ×         │  (× = beam axis)
#            │                  │
#       z=0  └──F3↑─────────F2←┘
#            y=0                y=1
#
# Force Patches (each 0.15 × 0.15 = 3×3 elements):
#   F1: z=1.0 face, x∈[1.85,2.0], y∈[0.85,1.0], dir=[0, 0,-1]
#   F2: y=1.0 face, x∈[1.85,2.0], z∈[0.0,0.15], dir=[0,-1, 0]
#   F3: z=0.0 face, x∈[1.85,2.0], y∈[0.0,0.15], dir=[0, 0,+1]
#   F4: y=0.0 face, x∈[1.85,2.0], z∈[0.85,1.0], dir=[0,+1, 0]
#
# Optimization Goal:
#   - Minimize compliance under torsional (propeller) loading
#   - Target volume fraction: 40%
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
results_dir = "./results/08_3D_2x1x1_propeller_$(tol_str)tol_r$(FILTER_RADIUS)"

# -----------------------------------------------------------------------------
# 1. MESH GENERATION
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(Hexahedron, (40, 20, 20), Vec((0.0, 0.0, 0.0)), Vec((2.0, 1.0, 1.0)))
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

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
# 4. BOUNDARY CONDITIONS - FIXED SUPPORT AT x = 0
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], eps())
println("  ✓ Fixed support nodes (x=0): $(length(fixed_nodes))")

# -----------------------------------------------------------------------------
# 5. FORCE APPLICATION - 4 PROPELLER FORCES
# -----------------------------------------------------------------------------
println("\nSelecting force nodes (propeller pattern)...")

# Helper: select nodes on a given face plane within a rectangular patch
function select_patch_nodes(
    grid,
    plane_coord::Float64,
    plane_dim::Int,
    range1::Tuple{Float64,Float64},
    dim1::Int,
    range2::Tuple{Float64,Float64},
    dim2::Int,
)
    nodes = Set{Int}()
    for node_id = 1:getnnodes(grid)
        coord = grid.nodes[node_id].x
        if abs(coord[plane_dim] - plane_coord) < eps() &&
           coord[dim1] >= range1[1] - eps() &&
           coord[dim1] <= range1[2] + eps() &&
           coord[dim2] >= range2[1] - eps() &&
           coord[dim2] <= range2[2] + eps()
            push!(nodes, node_id)
        end
    end
    return nodes
end

# F1: z=1.0 face, x∈[1.85,2.0], y∈[0.85,1.0], dir=[0,0,-1]
f1_nodes = select_patch_nodes(grid, 1.0, 3, (1.85, 2.0), 1, (0.85, 1.0), 2)
println("  ✓ F1 nodes (z=1, top-right):    $(length(f1_nodes))  → [0, 0,-1]")

# F2: y=1.0 face, x∈[1.85,2.0], z∈[0.0,0.15], dir=[0,-1,0]
f2_nodes = select_patch_nodes(grid, 1.0, 2, (1.85, 2.0), 1, (0.0, 0.15), 3)
println("  ✓ F2 nodes (y=1, bottom-right): $(length(f2_nodes))  → [0,-1, 0]")

# F3: z=0.0 face, x∈[1.85,2.0], y∈[0.0,0.15], dir=[0,0,+1]
f3_nodes = select_patch_nodes(grid, 0.0, 3, (1.85, 2.0), 1, (0.0, 0.15), 2)
println("  ✓ F3 nodes (z=0, bottom-left):  $(length(f3_nodes))  → [0, 0,+1]")

# F4: y=0.0 face, x∈[1.85,2.0], z∈[0.85,1.0], dir=[0,+1,0]
f4_nodes = select_patch_nodes(grid, 0.0, 2, (1.85, 2.0), 1, (0.85, 1.0), 3)
println("  ✓ F4 nodes (y=0, top-left):     $(length(f4_nodes))  → [0,+1, 0]")

# Verify all patches found nodes
for (name, ns) in [("F1", f1_nodes), ("F2", f2_nodes), ("F3", f3_nodes), ("F4", f4_nodes)]
    isempty(ns) &&
        error("No nodes found for $name! Check mesh resolution and patch coordinates.")
end

# -----------------------------------------------------------------------------
# 6. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
mkpath(results_dir)

all_force_nodes = union(f1_nodes, f2_nodes, f3_nodes, f4_nodes)
export_boundary_conditions(
    grid,
    dh,
    fixed_nodes,
    all_force_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 7. APPLY BOUNDARY CONDITIONS AND FORCES
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

ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
println("  ✓ Fixed support: $(length(fixed_nodes)) nodes (all DOFs)")

# Apply propeller forces (total force = -1.0 per patch)
apply_force!(f, dh, collect(f1_nodes), [0.0, 0.0, -1.0])
apply_force!(f, dh, collect(f2_nodes), [0.0, -1.0, 0.0])
apply_force!(f, dh, collect(f3_nodes), [0.0, 0.0, 1.0])
apply_force!(f, dh, collect(f4_nodes), [0.0, 1.0, 0.0])
println("  ✓ Applied 4 propeller forces (1.0 N each)")

# -----------------------------------------------------------------------------
# 8. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-9,
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
    task_name = "3D_Propeller_2x1x1",
)

println("\nOptimization parameters:")
println("  Task name: $(opt_params.task_name)")
println("  Target volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Tolerance: $(opt_params.tolerance)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 9. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING 3D BEAM OPTIMIZATION (propeller loading)")
println("="^80)

results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [
        PointLoad(dh, collect(f1_nodes), [0.0, 0.0, -1.0]),
        PointLoad(dh, collect(f2_nodes), [0.0, -1.0, 0.0]),
        PointLoad(dh, collect(f3_nodes), [0.0, 0.0, 1.0]),
        PointLoad(dh, collect(f4_nodes), [0.0, 1.0, 0.0]),
    ],
    [ch_fixed],
    opt_params,
)

# -----------------------------------------------------------------------------
# 10. EXPORT FINAL RESULTS
# -----------------------------------------------------------------------------
println("\nExporting final results...")
results_data = create_results_data(grid, dh, results)
export_results_vtu(results_data, joinpath(results_dir, "final"))

# -----------------------------------------------------------------------------
# 11. SUMMARY
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
println("  • Fixed: Entire YZ face at x = 0")
println("  • Load: 4 propeller forces (3×3 element patches on free end)")
println("    F1: z=1.0, y∈[0.85,1.0] → [0, 0,-1]")
println("    F2: y=1.0, z∈[0.0,0.15] → [0,-1, 0]")
println("    F3: z=0.0, y∈[0.0,0.15] → [0, 0,+1]")
println("    F4: y=0.0, z∈[0.85,1.0] → [0,+1, 0]")

println("\nResults Location:")
println("  $results_dir/")
println("="^80)
