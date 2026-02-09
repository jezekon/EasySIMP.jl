# =============================================================================
# 3D BEAM WITH 4-CORNER FIXATION - SIMP Topology Optimization
# =============================================================================
#
# Description:
#   3D beam optimization with 4 corner fixations on one end and 
#   circular load on the opposite end. Based on Gridap benchmark problem.
#
# Geometry:
#   - Domain: 2.0 × 1.0 × 1.0 (X × Y × Z)
#   - Fixed support: 4 corners at x=0 face (0.3 × 0.3 squares)
#   - Load: Circular region at x=2.0 face center, radius 0.1, force in -Z
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra

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
# 4. BOUNDARY CONDITIONS - 4 CORNER FIXATIONS
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

xmax, ymax, zmax = 2.0, 1.0, 1.0
fix_size = 0.3

# Select nodes at x=0 face within 4 corner regions
fixed_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]

    if abs(x) < 1e-6  # x ≈ 0
        # Bottom-left corner (y≈0, z≈0)
        in_corner1 = (y <= fix_size + 1e-6) && (z <= fix_size + 1e-6)
        # Bottom-right corner (y≈ymax, z≈0)
        in_corner2 = (y >= ymax - fix_size - 1e-6) && (z <= fix_size + 1e-6)
        # Top-left corner (y≈0, z≈zmax)
        in_corner3 = (y <= fix_size + 1e-6) && (z >= zmax - fix_size - 1e-6)
        # Top-right corner (y≈ymax, z≈zmax)
        in_corner4 = (y >= ymax - fix_size - 1e-6) && (z >= zmax - fix_size - 1e-6)

        if in_corner1 || in_corner2 || in_corner3 || in_corner4
            push!(fixed_nodes, node_id)
        end
    end
end
println("  ✓ Fixed corner nodes: $(length(fixed_nodes))")

# -----------------------------------------------------------------------------
# 5. FORCE APPLICATION - CIRCULAR REGION
# -----------------------------------------------------------------------------
load_radius = 0.1
force_center = [xmax, ymax/2, zmax/2]

# Select nodes in circular region at x=xmax
force_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]

    if abs(x - xmax) < 1e-6  # x ≈ xmax
        dist_sq = (y - force_center[2])^2 + (z - force_center[3])^2
        if dist_sq <= load_radius^2 + 1e-6
            push!(force_nodes, node_id)
        end
    end
end

if isempty(force_nodes)
    println("  ⚠ No force nodes found, using closest to center")
    min_dist = Inf
    closest = 1
    for node_id = 1:getnnodes(grid)
        dist = norm(grid.nodes[node_id].x - force_center)
        if dist < min_dist
            min_dist = dist
            closest = node_id
        end
    end
    force_nodes = Set([closest])
end
println("  ✓ Force nodes (circular): $(length(force_nodes))")

# -----------------------------------------------------------------------------
# 6. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
results_dir = "./results/05_3D_2x1x1_4Legs_8tol_r2.0"
mkpath(results_dir)

export_boundary_conditions(
    grid,
    dh,
    fixed_nodes,
    force_nodes,
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
println("  ✓ Fixed support: $(length(fixed_nodes)) nodes")

# Force in -Z direction (total force = -1.0)
apply_force!(f, dh, collect(force_nodes), [0.0, 0.0, -1.0])
println("  ✓ Applied force: [0, 0, -1] N")

# -----------------------------------------------------------------------------
# 8. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-9,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.4,
    max_iterations = 2000,
    tolerance = 0.08,
    filter_radius = 2.0,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 10,
    export_path = results_dir,
)

println("\nOptimization parameters:")
println("  Target volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 9. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING 3D BEAM OPTIMIZATION (4-corner fixation)")
println("="^80)

results = simp_optimize(
    grid,
    dh,
    cellvalues,
    [PointLoad(dh, collect(force_nodes), [0.0, 0.0, -1.0])],
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
println("  • Fixed: 4 corners at x=0 (0.3×0.3 each)")
println("  • Load: Circular region r=0.1 at x=2.0 center, F=[0,0,-1]")

println("\nResults Location:")
println("  $results_dir/")
println("="^80)

# Single thread computation:
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia -t 1 --project=. test/Examples/05_3D_2x1x1_4Legs.jl
