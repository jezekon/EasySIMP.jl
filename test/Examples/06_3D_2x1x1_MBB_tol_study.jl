# =============================================================================
# 3D MBB BEAM WITH SYMMETRY - SIMP Topology Optimization
# BATCH RUN: Multiple tolerance values
# =============================================================================
#
# Description:
#   MBB (Messerschmitt-Bölkow-Blohm) beam with symmetry plane on the left,
#   roller support on the bottom-right, and a point load on the top-left.
#   Batch tolerance study.
#
# Problem Visualization (side view, XY plane at z=0.5):
#
#        Y ↑
#          |↓ F [0,-1,0] semicircle r=0.1 at [0.,1.,0.5]
#      1.0 |████████████████████████████████████████████████
#          |█                                              █
#          |█         DESIGN DOMAIN                        █
#  symmetry|█         2.0 × 1.0 × 1.0                      █
#  U1=0    |█                                              █
#          |█                                              █
#          |█                                              █
#       0  |████████████████████████████████████████████████
#          └─────────────────────────────────────────────▲▲▲→ X
#          0                   1.0                     ≥1.95
#                                                   U2=0 (roller)
#
#        (Z dimension: 0 to 1.0, perpendicular to page)
#        Symmetry plane: x=0, full YZ face, U1=0
#        Roller support: y=0, x≥1.95, all Z, U2=0
#        Z-fix: single node at [0,1,0.5], U3=0 (prevents rigid body motion)
#
# Boundary Conditions:
#   - Symmetry: Left face (x=0) - U1=0 (sliding in Y,Z)
#   - Roller support: Bottom face (y=0), right edge (x≥1.95) - U2=0
#   - Z-constraint: Single node at [0,1,0.5] - U3=0
#   - Point load: Semicircle on top face at [0,1,0.5], radius 0.1 - F = [0, -1, 0] N
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra
using Printf
using Dates

# -----------------------------------------------------------------------------
# TOLERANCE VALUES TO TEST
# -----------------------------------------------------------------------------
# First run triggers JIT compilation and is slower - duplicate 0.16 ensures
# the second run (and all subsequent) give consistent timing results.
tolerance_values = [0.16, 0.16, 0.08, 0.04, 0.02, 0.01]

# Storage for results
struct BatchResult
    compliance::Float64
    volume_fraction::Float64
    iterations::Int
    converged::Bool
    elapsed_time::Float64
    results_dir::String
end

all_results = Dict{Float64,BatchResult}()

# -----------------------------------------------------------------------------
# 1. MESH GENERATION (shared for all runs)
# -----------------------------------------------------------------------------
println("Generating mesh...")
grid = generate_grid(Hexahedron, (40, 20, 20), Vec((0.0, 0.0, 0.0)), Vec((2.0, 1.0, 1.0)))
println("  ✓ Generated: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# Mesh parameters
xmax, ymax, zmax = 2.0, 1.0, 1.0

# -----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 1.0
ν = 0.3
material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

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
# 4. BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("\nApplying boundary conditions...")

# Symmetry: Left face (x=0) - fixed in X direction only
symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], eps())
println("  ✓ Symmetry nodes (X-fixed): $(length(symmetry_nodes))")

# Support: Bottom face (y=0), right edge (x≥1.95, all Z) - fixed in Y direction only
support_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps() && coord[1] >= 2.0 - 0.05 - eps()
        push!(support_nodes, node_id)
    end
end
println("  ✓ Support nodes (Y-fixed): $(length(support_nodes))")

# Z-constraint: Single node to prevent rigid body motion in Z
z_fix_node = Set{Int}()
target = [0.0, 1.0, 0.5]
min_dist = Inf
closest = 1
for node_id = 1:getnnodes(grid)
    dist = norm(grid.nodes[node_id].x - target)
    if dist < min_dist
        global min_dist = dist
        global closest = node_id
    end
end
push!(z_fix_node, closest)
println("  ✓ Z-fix node: 1 (prevents rigid body motion)")

# Force: Semicircle on top face
force_center = [0.0, 1.0, 0.5]
force_radius = 0.1 + eps()

force_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2] - 1.0) < eps()
        dx = coord[1] - force_center[1]
        dz = coord[3] - force_center[3]
        dist = sqrt(dx^2 + dz^2)
        if dist <= force_radius && coord[1] >= force_center[1] - eps()
            push!(force_nodes, node_id)
        end
    end
end
println("  ✓ Force nodes (semicircle): $(length(force_nodes))")

# -----------------------------------------------------------------------------
# 5. BATCH OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING BATCH OPTIMIZATION")
println("Tolerance values: $tolerance_values")
println("="^80)

total_mesh_volume = calculate_volume(grid)

for tol in tolerance_values
    # Format tolerance for folder name
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    results_dir = "./results/06_3D_2x1x1_MBB_$(tol_str)tol_r2.0"
    mkpath(results_dir)

    println("\n" * "="^80)
    println("RUNNING: tolerance = $tol (folder: $(tol_str)tol)")
    println("="^80)

    # Export boundary conditions (only for first run)
    if tol == tolerance_values[1]
        export_boundary_conditions(
            grid,
            dh,
            union(symmetry_nodes, support_nodes, z_fix_node),
            force_nodes,
            joinpath(results_dir, "boundary_conditions"),
        )
        println("  ✓ Saved: boundary_conditions.vtu")
    end

    # Fresh K and f for each run
    K_run = allocate_matrix(dh)
    f_run = zeros(ndofs(dh))

    # Assemble and apply BC
    assemble_stiffness_matrix_simp!(
        K_run,
        f_run,
        dh,
        cellvalues,
        material_model,
        fill(0.4, getncells(grid)),
    )

    ch_symmetry = apply_sliding_boundary!(K_run, f_run, dh, symmetry_nodes, [1])
    ch_support = apply_sliding_boundary!(K_run, f_run, dh, support_nodes, [2])
    ch_z_fix = apply_sliding_boundary!(K_run, f_run, dh, z_fix_node, [3])

    apply_force!(f_run, dh, collect(force_nodes), [0.0, -1.0, 0.0])

    # Optimization parameters
    opt_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-6,
        ν = ν,
        p = 3.0,
        volume_fraction = 0.4,
        max_iterations = 3000,
        tolerance = tol,
        filter_radius = 2.0,
        move_limit = 0.2,
        damping = 0.5,
        use_cache = true,
        export_interval = 3000,
        export_path = results_dir,
        task_name = "3D_MBB_Beam_$(tol_str)tol",
    )

    println("  Target volume fraction: $(opt_params.volume_fraction)")
    println("  Tolerance: $(opt_params.tolerance)")
    println("  Filter radius: $(opt_params.filter_radius)")

    # Run optimization with timing
    t_start = time()
    results = simp_optimize(
        grid,
        dh,
        cellvalues,
        [PointLoad(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
        [ch_symmetry, ch_support, ch_z_fix],
        opt_params,
    )
    elapsed = time() - t_start

    # Export final results
    results_data = create_results_data(grid, dh, results)
    export_results_vtu(
        results_data,
        joinpath(results_dir, "3D_2x1x1_MBB_$(tol_str)tol_r2.0-SIMP"),
    )

    vol_frac = results.volume / total_mesh_volume

    # Store results
    all_results[tol] = BatchResult(
        results.compliance,
        vol_frac,
        results.iterations,
        results.converged,
        elapsed,
        results_dir,
    )

    # Write per-tolerance summary txt file
    summary_path = joinpath(results_dir, "optimization_summary.txt")
    open(summary_path, "w") do io
        println(io, "=" ^ 60)
        println(io, "SIMP TOPOLOGY OPTIMIZATION SUMMARY")
        println(io, "=" ^ 60)
        println(io)
        println(io, "Task name:           3D_MBB_Beam_$(tol_str)tol")
        println(io, "Tolerance:           $tol")
        println(io, "Iterations:          $(results.iterations)")
        println(io, "Total time:          $(round(elapsed, digits=2)) s")
        println(io, "Converged:           $(results.converged ? "Yes" : "No")")
        println(io)
        println(io, "Final compliance:    $(results.compliance)")
        println(io, "Final volume frac.:  $(round(vol_frac, digits=6))")
        println(io, "Final volume:        $(results.volume)")
        println(io)
        println(io, "Generated:           $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "=" ^ 60)
    end

    println("  ✓ Summary saved: $summary_path")
    @printf(
        "  Result: C=%.6f, Vf=%.4f, Iter=%d, Time=%.1fs, Conv=%s\n",
        results.compliance,
        vol_frac,
        results.iterations,
        elapsed,
        results.converged ? "Yes" : "No"
    )

    # Clean up memory between runs
    GC.gc(true)
end

# -----------------------------------------------------------------------------
# 6. FINAL SUMMARY
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("BATCH OPTIMIZATION COMPLETED")
println("="^80)
println("\nSummary of all runs:")
println("-"^90)
@printf(
    "%-10s | %-12s | %-12s | %-10s | %-10s | %-10s\n",
    "Tolerance",
    "Compliance",
    "Vol.Frac",
    "Iterations",
    "Time [s]",
    "Converged"
)
println("-"^90)

for tol in sort(collect(keys(all_results)))
    r = all_results[tol]
    @printf(
        "%-10.4f | %-12.6f | %-12.4f | %-10d | %-10.1f | %-10s\n",
        tol,
        r.compliance,
        r.volume_fraction,
        r.iterations,
        r.elapsed_time,
        r.converged ? "Yes" : "No"
    )
end
println("-"^90)

println("\nResults saved to:")
for tol in sort(collect(keys(all_results)))
    println("  $(all_results[tol].results_dir)/")
end
println("="^80)

# Write global summary txt file
global_summary_path = "./results/06_mbb_beam_batch_summary.txt"
open(global_summary_path, "w") do io
    println(io, "=" ^ 90)
    println(io, "BATCH TOLERANCE STUDY - 3D MBB BEAM WITH SYMMETRY")
    println(io, "=" ^ 90)
    println(io)
    println(
        io,
        "Problem: 2.0 × 1.0 × 1.0, symmetry at x=0, roller at bottom-right, top-left load",
    )
    println(io, "Mesh: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
    println(io, "Material: E₀ = $E0, ν = $ν")
    println(io, "Volume fraction: 0.4")
    println(io, "Filter radius: 2.0")
    println(io)
    println(io, "-" ^ 90)
    @printf(
        io,
        "%-10s | %-12s | %-12s | %-10s | %-10s | %-10s\n",
        "Tolerance",
        "Compliance",
        "Vol.Frac",
        "Iterations",
        "Time [s]",
        "Converged"
    )
    println(io, "-" ^ 90)
    for tol in sort(collect(keys(all_results)))
        r = all_results[tol]
        @printf(
            io,
            "%-10.4f | %-12.6f | %-12.4f | %-10d | %-10.1f | %-10s\n",
            tol,
            r.compliance,
            r.volume_fraction,
            r.iterations,
            r.elapsed_time,
            r.converged ? "Yes" : "No"
        )
    end
    println(io, "-" ^ 90)
    println(io)
    println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "=" ^ 90)
end
println("\nGlobal summary: $global_summary_path")
println("="^80)

# Single thread computation:
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia -t 1 --project=. test/Examples/06_3D_2x1x1_MBB_tol_study.jl
