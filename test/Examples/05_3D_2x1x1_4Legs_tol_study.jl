# =============================================================================
# 3D BEAM WITH 4-CORNER FIXATION - SIMP Topology Optimization
# BATCH RUN: Multiple tolerance values
# =============================================================================
#
# Description:
#   Cantilever-type beam with four corner fixations on the left face (x=0)
#   and a point load on the right face center. Batch tolerance study.
#
# Problem Visualization (front view, XY plane at z=0.5):
#
#        Y ↑
#          |
#      1.0 ▓██████████████████████████████████████████████████
#          ▓█                                                █
#          ▓█         DESIGN DOMAIN                          █
#          |█         2.0 × 1.0 × 1.0                        █
#          |█                                                █ ↓ F [0,0,-1]
#          |█                                                █  circle r=0.1
#          ▓█                                                █  Force at [2,0.5,0.5]
#          ▓█                                                █
#       0  ▓██████████████████████████████████████████████████
#          └─────────────────────────────────────────────────────→ X
#          0                   1.0                          2.0
#
#        (Z dimension: 0 to 1.0, perpendicular to page)
#        4 corner fixations on left face (x=0): fix_size=0.3
#        ▓▓ = U1=U2=U3=0 at corners (y≤0.3,z≤0.3), (y≥0.7,z≤0.3),
#             (y≤0.3,z≥0.7), (y≥0.7,z≥0.7)
#
# Problem Visualization (left face view, YZ plane at x=0):
#
#        Z ↑
#          |
#      1.0 |▓▓▓▓▓▓▓▓▓▓             ▓▓▓▓▓▓▓▓▓▓
#          |▓▓▓▓▓▓▓▓▓▓             ▓▓▓▓▓▓▓▓▓▓
#          |▓▓▓▓▓▓▓▓▓▓             ▓▓▓▓▓▓▓▓▓▓
#          |   0.3                      0.3
#          |                                    
#          |   0.3                      0.3
#          |▓▓▓▓▓▓▓▓▓▓             ▓▓▓▓▓▓▓▓▓▓
#          |▓▓▓▓▓▓▓▓▓▓             ▓▓▓▓▓▓▓▓▓▓
#       0  |▓▓▓▓▓▓▓▓▓▓             ▓▓▓▓▓▓▓▓▓▓
#          └────────────────────────────────────→ Y
#          0         0.3        0.7           1.0
#          ▓▓ = Fixed support U1=U2=U3=0
#
# Boundary Conditions:
#   - Fixed support: 4 corners on left face (x=0), 0.3×0.3 each - U1=U2=U3=0
#   - Point load: Circular region at [2.0, 0.5, 0.5], radius 0.1 - F = [0, 0, -1] N
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
    energy::Float64
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

fix_size = 0.3

fixed_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]

    if abs(x) < 1e-6
        in_corner1 = (y <= fix_size + 1e-6) && (z <= fix_size + 1e-6)
        in_corner2 = (y >= ymax - fix_size - 1e-6) && (z <= fix_size + 1e-6)
        in_corner3 = (y <= fix_size + 1e-6) && (z >= zmax - fix_size - 1e-6)
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

force_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    x, y, z = coord[1], coord[2], coord[3]

    if abs(x - xmax) < 1e-6
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
# 6. BATCH OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING BATCH OPTIMIZATION")
println("Tolerance values: $tolerance_values")
println("="^80)

total_mesh_volume = calculate_volume(grid)

for tol in tolerance_values
    # Format tolerance for folder name (e.g., 0.01 -> "01", 0.08 -> "08", 0.16 -> "16")
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    results_dir = "./results/05_3D_2x1x1_4Legs_$(tol_str)tol_r2.0"
    mkpath(results_dir)

    println("\n" * "="^80)
    println("RUNNING: tolerance = $tol (folder: $(tol_str)tol)")
    println("="^80)

    # Export boundary conditions (only for first run)
    if tol == tolerance_values[1]
        export_boundary_conditions(
            grid,
            dh,
            fixed_nodes,
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

    ch_fixed = apply_fixed_boundary!(K_run, f_run, dh, fixed_nodes)
    apply_force!(f_run, dh, collect(force_nodes), [0.0, 0.0, -1.0])

    # Optimization parameters
    opt_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-9,
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
        task_name = "3D_4Legs_$(tol_str)tol",
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
        [PointLoad(dh, collect(force_nodes), [0.0, 0.0, -1.0])],
        [ch_fixed],
        opt_params,
    )
    elapsed = time() - t_start

    # Export final results
    results_data = create_results_data(grid, dh, results)
    export_results_vtu(
        results_data,
        joinpath(results_dir, "3D_2x1x1_4Legs_$(tol_str)tol_r2.0-SIMP"),
    )

    vol_frac = results.volume / total_mesh_volume

    # Store results
    all_results[tol] = BatchResult(
        results.energy,
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
        println(io, "Task name:           3D_4Legs_$(tol_str)tol")
        println(io, "Tolerance:           $tol")
        println(io, "Iterations:          $(results.iterations)")
        println(io, "Total time:          $(round(elapsed, digits=2)) s")
        println(io, "Converged:           $(results.converged ? "Yes" : "No")")
        println(io)
        println(io, "Final energy:        $(results.energy)")
        println(io, "Final volume frac.:  $(round(vol_frac, digits=6))")
        println(io, "Final volume:        $(results.volume)")
        println(io)
        println(io, "Generated:           $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "=" ^ 60)
    end

    println("  ✓ Summary saved: $summary_path")
    @printf(
        "  Result: C=%.6f, Vf=%.4f, Iter=%d, Time=%.1fs, Conv=%s\n",
        results.energy,
        vol_frac,
        results.iterations,
        elapsed,
        results.converged ? "Yes" : "No"
    )

    # Clean up memory between runs
    GC.gc(true)
end

# -----------------------------------------------------------------------------
# 7. FINAL SUMMARY
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("BATCH OPTIMIZATION COMPLETED")
println("="^80)
println("\nSummary of all runs:")
println("-"^90)
@printf(
    "%-10s | %-12s | %-12s | %-10s | %-10s | %-10s\n",
    "Tolerance",
    "Energy",
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
        r.energy,
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
global_summary_path = "./results/05_4legs_batch_summary.txt"
open(global_summary_path, "w") do io
    println(io, "=" ^ 90)
    println(io, "BATCH TOLERANCE STUDY - 3D BEAM WITH 4-CORNER FIXATION")
    println(io, "=" ^ 90)
    println(io)
    println(
        io,
        "Problem: 2.0 × 1.0 × 1.0, four corner fixations on left face (0.3×0.3 each), point load at right face center",
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
        "Energy",
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
            r.energy,
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
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia -t 1 --project=. test/Examples/05_3D_2x1x1_4Legs_tol_study.jl
