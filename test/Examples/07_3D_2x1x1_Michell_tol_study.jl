# =============================================================================
# 3D MICHELL-TYPE BEAM - SIMP Topology Optimization
# BATCH RUN: Multiple tolerance values
# =============================================================================
#
# Description:
#   Michell-type beam problem with four corner supports on the bottom face
#   and a central point load. Batch tolerance study.
#
# Problem Visualization (side view, XY plane at z=0.5):
#
#        Y ↑
#          |
#      1.0 |████████████████████████████████████████████████
#          |█                                              █
#          |█         DESIGN DOMAIN                        █
#          |█         2.0 × 1.0 × 1.0                      █
#          |█                                              █
#       0  |████████████████████████████████████████████████
#          ▓▓                   ↓ F                      ▓▓
#       U1=U2=U3=0      circle r=0.1               U1=U2=U3=0
#       (corners)     at [1,0,0.5], [0,-1,0]          (corners)
#          └─────────────────────────────────────────────────→ X
#          0                   1.0                        2.0
#
#        (Z dimension: 0 to 1.0, perpendicular to page)
#        4 corner supports: 3×3 elements each at (x=0,z=0), (x=0,z=1),
#                           (x=2,z=0), (x=2,z=1)
#
# Boundary Conditions:
#   - Fixed support: 4 corners on bottom face (y=0), 3×3 elements each - U1=U2=U3=0
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
# 4. BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

# Fixed support: 4 corner regions on bottom face (y=0), each 3×3 elements = 0.15×0.15
# Corner size: 3 elements × 0.05 = 0.15
corner_size = 0.15

# Support left: 2 corners at x=0 (z=0 and z=zmax) - fixed U1=U2=U3=0
support_left = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps() && coord[1] <= corner_size + eps()
        # Bottom-left corner (x≈0, z≈0)
        in_corner1 = coord[3] <= corner_size + eps()
        # Top-left corner (x≈0, z≈zmax)
        in_corner2 = coord[3] >= zmax - corner_size - eps()
        if in_corner1 || in_corner2
            push!(support_left, node_id)
        end
    end
end
println("  ✓ Support left (2 corners, 3×3 elem, fixed): $(length(support_left)) nodes")

# Support right: 2 corners at x=xmax (z=0 and z=zmax) - fixed U1=U2=U3=0
support_right = Set{Int}()
for node_id = 1:getnnodes(grid)
    coord = grid.nodes[node_id].x
    if abs(coord[2]) < eps() && coord[1] >= xmax - corner_size - eps()
        # Bottom-right corner (x≈xmax, z≈0)
        in_corner1 = coord[3] <= corner_size + eps()
        # Top-right corner (x≈xmax, z≈zmax)
        in_corner2 = coord[3] >= zmax - corner_size - eps()
        if in_corner1 || in_corner2
            push!(support_right, node_id)
        end
    end
end
println("  ✓ Support right (2 corners, 3×3 elem, fixed): $(length(support_right)) nodes")

# Force: Circular region on bottom face (y=0)
force_center = [1.0, 0.0, 0.5]
force_radius = 0.1 + eps()

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

# -----------------------------------------------------------------------------
# 5. BATCH OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
total_mesh_volume = calculate_volume(grid)

for tol in tolerance_values
    tol_str = @sprintf("%02d", round(Int, tol * 100))
    results_dir = "./results/07_3D_2x1x1_Michell_$(tol_str)tol_r2.0"
    mkpath(results_dir)

    println("\n" * "="^80)
    println("RUNNING: Tolerance = $tol  →  $results_dir")
    println("="^80)

    # Export boundary conditions (only for first run)
    if tol == tolerance_values[1]
        all_support_nodes = union(support_left, support_right)
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

    ch_support_left = apply_fixed_boundary!(K_run, f_run, dh, support_left)
    ch_support_right = apply_fixed_boundary!(K_run, f_run, dh, support_right)

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
        task_name = "3D_Michell_Type_$(tol_str)tol",
    )

    # Run optimization with timing
    t_start = time()
    results = simp_optimize(
        grid,
        dh,
        cellvalues,
        [PointLoad(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
        [ch_support_left, ch_support_right],
        opt_params,
    )
    elapsed = time() - t_start

    # Export final results
    results_data = create_results_data(grid, dh, results)
    export_results_vtu(
        results_data,
        joinpath(results_dir, "3D_2x1x1_Michell_$(tol_str)tol_r2.0-SIMP"),
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
        println(io, "Task name:           3D_Michell_Type_$(tol_str)tol")
        println(io, "Tolerance:           $tol")
        println(io, "Iterations:          $(results.iterations)")
        println(io, "Total time:          $(round(elapsed, digits=2)) s")
        println(io, "Converged:           $(results.converged ? "Yes" : "No")")
        println(io)
        println(io, "Final energy:    $(results.energy)")
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
global_summary_path = "./results/07_michell_type_batch_summary.txt"
open(global_summary_path, "w") do io
    println(io, "=" ^ 90)
    println(io, "BATCH TOLERANCE STUDY - 3D MICHELL-TYPE BEAM")
    println(io, "=" ^ 90)
    println(io)
    println(
        io,
        "Problem: 2.0 × 1.0 × 1.0, four corner fixed supports (3×3 elem), central point load",
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
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia -t 1 --project=. test/Examples/07_3D_2x1x1_Michell_tol_study.jl
