# =============================================================================
# 3D WHEEL - SIMP Topology Optimization (OPRAVENÁ VERZE)
# =============================================================================
#
# OPRAVY:
#   1. Zvětšená tolerance pro select_nodes_by_arc (z 1e-2 na 0.05)
#   2. Přidána diagnostika pro ověření správnosti BC
#   3. Opravena trakce - použití správné surface integration
#
# Description:
#   Hollow cylinder (wheel) under tangential traction load.
#   Based on benchmark from Wegert et al. (2025), adapted for SIMP method.
#
# Geometry (top view, looking down Z-axis):
#
#              Y ↑
#                |       Arc fixations (5x on outer rim)
#           _____|_____
#          /  ·  |  ·  \     r_outer = 1.0
#         / ·    |    · \    
#        |   ____|____   |
#        |  /    |    \  |   r_inner = 0.1
#        | |     O-----|--→ X
#        |  \________/  |    Tangential traction g = 100(-y, x, 0)
#         \ ·        · /     on inner cylinder surface
#          \_·____|__·_/
#                |
#
#        Z dimension: -0.15 to +0.15 (thickness = 0.3)
#
# Boundary Conditions (from reference Wheel3D_CutFEM.jl):
#   - Fixed support: 5 arcs (15° each) on outer cylinder at r = 1.0
#     Upper half (y > 0): 15°-30°, 82.5°-97.5°, 150°-165°
#     Lower half (y < 0): 217.5°-232.5° (= 127.5°-142.5° mirrored)
#                         307.5°-322.5° (= 37.5°-52.5° mirrored)
#   - Tangential load: Inner cylinder (r = 0.1), g(x,y,z) = 100*(-y, x, 0)
#
# =============================================================================

using EasySIMP
using Ferrite
using LinearAlgebra

# -----------------------------------------------------------------------------
# 1. GEOMETRY PARAMETERS
# -----------------------------------------------------------------------------
const R_INNER = 0.1      # Inner radius (hub)
const R_OUTER = 1.0      # Outer radius (rim)
const THICKNESS = 0.3    # Z-dimension (-0.15 to +0.15)

# -----------------------------------------------------------------------------
# 2. MESH IMPORT
# -----------------------------------------------------------------------------
println("Importing wheel mesh...")
mesh_path = "data/Wheel_3d_coarse.msh"

if !isfile(mesh_path)
    error(
        "Mesh file not found: $mesh_path\n" *
        "Generate mesh using Gmsh with Wheel_3d.geo file.",
    )
end

grid = import_mesh(mesh_path)
println("  ✓ Imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")

# -----------------------------------------------------------------------------
# 3. MATERIAL PROPERTIES
# -----------------------------------------------------------------------------
E0 = 1.0               # Young's modulus (normalized)
ν = 0.3                # Poisson's ratio
material_model = create_simp_material_model(E0, ν, 1e-9, 3.0)

println("\nMaterial properties:")
println("  E₀ = $E0")
println("  ν = $ν")

# -----------------------------------------------------------------------------
# 4. FEM SETUP
# -----------------------------------------------------------------------------
println("\nSetting up FEM problem...")
dh, cellvalues, K, f = setup_problem(grid)
println("  ✓ DOFs: $(ndofs(dh))")

# -----------------------------------------------------------------------------
# 5. BOUNDARY CONDITIONS - FIXED ARCS ON OUTER CYLINDER
# -----------------------------------------------------------------------------
println("\nSelecting boundary condition nodes...")

# 5 fixation arcs on outer cylinder (each spans 15°)
# Angles in standard convention (CCW from +X axis)
arc_angles = [
    (15.0, 30.0),       # Arc 1: upper right quadrant
    (82.5, 97.5),       # Arc 2: upper middle (near +Y)
    (150.0, 165.0),     # Arc 3: upper left quadrant
    (217.5, 232.5),     # Arc 4: lower left quadrant  
    (307.5, 322.5),     # Arc 5: lower right quadrant
]

fixed_nodes = Set{Int}()
axis_point = [0.0, 0.0, 0.0]
axis_normal = [0.0, 0.0, 1.0]

# OPRAVA: Zvětšená tolerance z 1e-2 na 0.05
# Mesh má element size ~0.037-0.08, potřebujeme toleranci srovnatelnou s velikostí elementu
arc_tolerance = 0.05

println("\n  Arc selection with tolerance = $arc_tolerance:")

for (i, (a_start, a_end)) in enumerate(arc_angles)
    arc_nodes = select_nodes_by_arc(
        grid,
        axis_point,
        axis_normal,
        R_OUTER,
        a_start,
        a_end,
        arc_tolerance,
    )
    union!(fixed_nodes, arc_nodes)

    # Diagnostika: vypsat souřadnice a úhly vybraných uzlů
    if !isempty(arc_nodes)
        angles = Float64[]
        for node_id in arc_nodes
            coord = grid.nodes[node_id].x
            angle = rad2deg(atan(coord[2], coord[1]))
            if angle < 0
                angle += 360
            end
            push!(angles, angle)
        end
        min_angle = minimum(angles)
        max_angle = maximum(angles)
        println(
            "  Arc $i ($(a_start)°-$(a_end)°): $(length(arc_nodes)) nodes, actual range: $(round(min_angle, digits=1))°-$(round(max_angle, digits=1))°",
        )
    else
        println("  Arc $i ($(a_start)°-$(a_end)°): 0 nodes ⚠ WARNING!")
    end
end

println("\n  ✓ Total fixed nodes: $(length(fixed_nodes))")

# Validace: kontrola že máme dostatek uzlů
expected_min_nodes = 30  # Očekáváme minimálně 6 uzlů na arc × 5 arcs
if length(fixed_nodes) < expected_min_nodes
    println(
        "  ⚠ WARNING: Only $(length(fixed_nodes)) fixed nodes found (expected >= $expected_min_nodes)",
    )
    println("    This may indicate tolerance issues with the mesh.")
end

# Fallback pokud stále nemáme dostatek uzlů
if isempty(fixed_nodes) || length(fixed_nodes) < 5
    println("  ⚠ CRITICAL: Too few nodes, using alternative selection method")

    # Alternativní metoda: přímý výběr podle reference implementace
    fixed_nodes = Set{Int}()
    for node_id = 1:getnnodes(grid)
        coord = grid.nodes[node_id].x
        x, y = coord[1], coord[2]
        r = sqrt(x^2 + y^2)

        # Pouze uzly na vnějším válci
        if abs(r - R_OUTER) > arc_tolerance
            continue
        end

        # Reference podmínky (z Wheel3D_CutFEM.jl)
        # Arc 1: 15°-30° (horní)
        if (cos(30*π/180) <= x <= cos(15*π/180)) &&
           abs(y - sqrt(max(0, R_OUTER^2 - x^2))) < arc_tolerance
            push!(fixed_nodes, node_id)
            # Arc 2: 82.5°-97.5° (horní)
        elseif (cos(97.5*π/180) <= x <= cos(82.5*π/180)) &&
               abs(y - sqrt(max(0, R_OUTER^2 - x^2))) < arc_tolerance
            push!(fixed_nodes, node_id)
            # Arc 3: 150°-165° (horní)
        elseif (cos(165*π/180) <= x <= cos(150*π/180)) &&
               abs(y - sqrt(max(0, R_OUTER^2 - x^2))) < arc_tolerance
            push!(fixed_nodes, node_id)
            # Arc 4: 127.5°-142.5° na dolní polovině
        elseif (cos(142.5*π/180) <= x <= cos(127.5*π/180)) &&
               abs(y - (-sqrt(max(0, R_OUTER^2 - x^2)))) < arc_tolerance
            push!(fixed_nodes, node_id)
            # Arc 5: 37.5°-52.5° na dolní polovině
        elseif (cos(52.5*π/180) <= x <= cos(37.5*π/180)) &&
               abs(y - (-sqrt(max(0, R_OUTER^2 - x^2)))) < arc_tolerance
            push!(fixed_nodes, node_id)
        end
    end
    println("  Alternative selection found: $(length(fixed_nodes)) nodes")
end

# -----------------------------------------------------------------------------
# 6. NEUMANN BC - TANGENTIAL TRACTION ON INNER CYLINDER
# -----------------------------------------------------------------------------
inner_nodes = select_nodes_by_cylinder(
    grid,
    axis_point,
    axis_normal,
    R_INNER,
    arc_tolerance,  # Použít stejnou toleranci
)
println("  ✓ Inner cylinder nodes: $(length(inner_nodes))")

if isempty(inner_nodes)
    println("  ⚠ Warning: No inner cylinder nodes found, trying larger tolerance")
    inner_nodes = select_nodes_by_cylinder(grid, axis_point, axis_normal, R_INNER, 0.1)
    println("  Found with larger tolerance: $(length(inner_nodes)) nodes")
end

# Tangential traction function
traction_magnitude = 100.0
g(x, y, z) = [traction_magnitude * (-y), traction_magnitude * x, 0.0]

# -----------------------------------------------------------------------------
# 7. EXPORT BOUNDARY CONDITIONS FOR VISUALIZATION
# -----------------------------------------------------------------------------
results_dir = "./results/07_wheel_fixed"
mkpath(results_dir)

export_boundary_conditions(
    grid,
    dh,
    fixed_nodes,
    inner_nodes,
    joinpath(results_dir, "boundary_conditions"),
)
println("  ✓ Saved: boundary_conditions.vtu")

# -----------------------------------------------------------------------------
# 8. APPLY BOUNDARY CONDITIONS
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

ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
println("  ✓ Fixed support: $(length(fixed_nodes)) nodes (all DOFs)")

# Apply tangential traction using surface integration
inner_facets = get_boundary_facets(grid, inner_nodes)
apply_surface_traction!(f, dh, grid, inner_facets, g)

# -----------------------------------------------------------------------------
# 9. OPTIMIZATION PARAMETERS
# -----------------------------------------------------------------------------
opt_params = OptimizationParameters(
    E0 = E0,
    Emin = 1e-9,
    ν = ν,
    p = 3.0,
    volume_fraction = 0.3,
    max_iterations = 300,
    tolerance = 0.08,
    filter_radius = 2.0,
    move_limit = 0.2,
    damping = 0.5,
    use_cache = true,
    export_interval = 10,
    export_path = results_dir,
    task_name = "Wheel_3D_Fixed",
)

println("\nOptimization parameters:")
println("  Target volume fraction: $(opt_params.volume_fraction)")
println("  Max iterations: $(opt_params.max_iterations)")
println("  Filter radius: $(opt_params.filter_radius)")

# -----------------------------------------------------------------------------
# 10. PREPARE FORCES FOR OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
# Approximation: average traction distributed to nodes
# forces_list = Tuple{DofHandler,Vector{Int},Vector{Float64}}[]
# for node_id in inner_nodes
#     coord = grid.nodes[node_id].x
#     node_traction = g(coord[1], coord[2], coord[3]) ./ length(inner_nodes)
#     push!(forces_list, (dh, [node_id], node_traction))
# end

# Tangential traction function: g(x,y,z) = 100*(-y, x, 0)
traction_magnitude = 100.0
g(x, y, z) = [traction_magnitude * (-y), traction_magnitude * x, 0.0]

traction_load = SurfaceTractionLoad(dh, grid, inner_nodes, g)

# Load list for optimization
loads = [traction_load]

# -----------------------------------------------------------------------------
# 11. RUN OPTIMIZATION
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("STARTING WHEEL TOPOLOGY OPTIMIZATION")
println("="^80)
println("Fixed: $(length(fixed_nodes)) nodes on 5 arcs")
println("Load: $(length(inner_nodes)) nodes on inner cylinder")
println()

# results = simp_optimize(grid, dh, cellvalues, forces_list, [ch_fixed], opt_params)
results = simp_optimize(grid, dh, cellvalues, loads, [ch_fixed], opt_params)

# -----------------------------------------------------------------------------
# 12. EXPORT FINAL RESULTS
# -----------------------------------------------------------------------------
println("\nExporting final results...")
results_data = create_results_data(grid, dh, results)
export_results_vtu(results_data, joinpath(results_dir, "final"))

# -----------------------------------------------------------------------------
# 13. SUMMARY
# -----------------------------------------------------------------------------
println("\n" * "="^80)
println("OPTIMIZATION COMPLETED")
println("="^80)
println("\nFinal Results:")
println("  Compliance: $(round(results.compliance, digits=6))")
println("  Volume fraction: $(round(results.volume / calculate_volume(grid), digits=4))")
println("  Iterations: $(results.iterations)")
println("  Converged: $(results.converged)")

println("\nResults saved to: $results_dir/")
println("="^80)
