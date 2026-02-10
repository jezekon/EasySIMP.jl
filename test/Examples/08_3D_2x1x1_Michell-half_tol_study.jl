# =============================================================================
# 3D MICHELL-TYPE BEAM (HALF) - SIMP Topology Optimization
# BATCH RUN: Multiple tolerance values
# =============================================================================
#
# Description:
#   Michell-type beam problem using one symmetry plane in Z.
#   Two corner supports on bottom face at z=0 side, symmetry at z=1.
#   Batch tolerance study.
#
# Problem Visualization (side view, XY plane at z=0):
#
#        Y ↑
#          |
#      1.0 |████████████████████████████████████████████████
#          |█                                              █
#          |█         DESIGN DOMAIN                        █
#          |█         2.0 × 1.0 × 1.0                      █
#          |█                                              █
#       0  |████████████████████████████████████████████████
#          △△                   ↓ F                      △△
#          U2=0          circle r=0.1                    U2=0
#       (4×4 elem)    at [1,0,0.5], [0,-1,0]          (4×4 elem)
#          └─────────────────────────────────────────────────→ X
#          0                   1.0                        2.0
#
#        (Z dimension: 0 to 1.0, symmetry at z=1.0)
#        2 corner supports: 4×4 elements each at (x=0,z=0), (x=2,z=0)
#
# Boundary Conditions:
#   - Support: 2 corners on bottom face (y=0) at z=0 side, 4×4 elements each - U2=0
#   - Symmetry plane XY at z=1.0: U3=0
#   - Symmetry plane ZY at x=1.0: U1=0
#   - Point load: Circular region at [1,0,0.5], radius 0.1 - F = [0, -1, 0] N
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
tolerance_values = [0.16, 0.08, 0.04, 0.02, 0.01]

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
# 4. BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

# Support: 2 corner regions on bottom face (y=0) at z=0 side, each 4×4 elements = 0.20×0.20
corner_size = 0.20

# Support at [0,0,0] corner (x≈0, z≈0)
support_left = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps() &&
       coord[1] <= corner_size + eps() &&
       coord[3] <= corner_size + eps()
        push!(support_left, node_id)
    end
end
println("  ✓ Support left [0,0,0] (4×4 elem): $(length(support_left)) nodes")

# Support at [2,0,0] corner (x≈xmax, z≈0)
support_right = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps() &&
       coord[1] >= xmax - corner_size - eps() &&
       coord[3] <= corner_size + eps()
        push!(support_right, node_id)
    end
end
println("  ✓ Support right [2,0,0] (4×4 elem): $(length(support_right)) nodes")

# Force: Circular region on bottom face (y=0)
force_center = [1.0, 0.0, 1.0]
force_radius = 0.2 + eps()

force_nodes = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps()
        dx = coord[1] - force_center[1]
        dz = coord[3] - force_center[3]
        dist = sqrt(dx^2 + dz^2)
        if dist <= force_radius
            push!(force_nodes, node_id)
        end
    end
end
println("  ✓ Force nodes (circle r=$(force_radius)): $(length(force_nodes))")

# Symmetry plane XY at z = 1.0: U3 = 0
symmetry_z_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1e-6)
println("  ✓ Symmetry plane z=1.0 (U3=0): $(length(symmetry_z_nodes)) nodes")

# Symmetry plane ZY at x = 1.0: U1 = 0
symmetry_x_nodes = select_nodes_by_plane(grid, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6)
println("  ✓ Symmetry plane x=1.0 (U1=0): $(length(symmetry_x_nodes)) nodes")

# -----------------------------------------------------------------------------
# 5. BATCH OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
total_mesh_volume = calculate_volume(grid)

for tol in tolerance_values
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    results_dir = "./results/08_3D_2x1x1_Michell-half_$(tol_str)tol_r2.0"
    mkpath(results_dir)

    println("\n" * "="^80)
    println("RUNNING: Tolerance = $tol  →  $results_dir")
    println("="^80)

    # Export boundary conditions (only for first run)
    if tol == tolerance_values[1]
        all_support_nodes =
            union(support_left, support_right, symmetry_z_nodes, symmetry_x_nodes)
        export_boundary_conditions(
            grid,
            dh,
            all_support_nodes,
            force_nodes,
            joinpath(results_dir, "boundary_conditions"),
        )
        println("  ✓ Saved: boundary_conditions.vtu")
    end

    # Re-assemble and apply boundary conditions for each run
    K_run = allocate_matrix(dh)
    f_run = zeros(ndofs(dh))

    assemble_stiffness_matrix_simp!(
        K_run,
        f_run,
        dh,
        cellvalues,
        material_model,
        fill(0.4, getncells(grid)),
    )

    ch_support_left = apply_sliding_boundary!(K_run, f_run, dh, support_left, [2])
    ch_support_right = apply_sliding_boundary!(K_run, f_run, dh, support_right, [2])
    ch_symmetry_z = apply_sliding_boundary!(K_run, f_run, dh, symmetry_z_nodes, [3])
    ch_symmetry_x = apply_sliding_boundary!(K_run, f_run, dh, symmetry_x_nodes, [1])

    apply_force!(f_run, dh, collect(force_nodes), [0.0, -1.0, 0.0])

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
        task_name = "3D_Michell-half_$(tol_str)tol",
    )

    # Run optimization with timing
    t_start = time()
    results = simp_optimize(
        grid,
        dh,
        cellvalues,
        [PointLoad(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
        [ch_support_left, ch_support_right, ch_symmetry_z, ch_symmetry_x],
        opt_params,
    )
    elapsed = time() - t_start

    # Export final results
    results_data = create_results_data(grid, dh, results)
    export_results_vtu(results_data, joinpath(results_dir, "final"))

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
        println(io, "Task name:           3D_Michell-half_$(tol_str)tol")
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
global_summary_path = "./results/08_michell-half_batch_summary.txt"
open(global_summary_path, "w") do io
    println(io, "=" ^ 90)
    println(io, "BATCH TOLERANCE STUDY - 3D MICHELL-TYPE BEAM (HALF)")
    println(io, "=" ^ 90)
    println(io)
    println(
        io,
        "Problem: 2.0 × 1.0 × 1.0, two corner supports (4×4 elem) at z=0, symmetry z=1.0",
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
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia -t 1 --project=. test/Examples/08_3D_2x1x1_Michell-half_tol_study.jl
