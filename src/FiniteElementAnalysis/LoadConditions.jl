# src/FiniteElementAnalysis/LoadConditions.jl
"""
LoadConditions.jl

Abstract load condition types for SIMP topology optimization.
Supports point loads and surface traction, uniform or with position-dependent functions.
"""

# =============================================================================
# ABSTRACT BASE TYPE
# =============================================================================

"""
    AbstractLoadCondition

Abstract base type for all load conditions in topology optimization.
Concrete implementations must define `apply_load_condition!(f, load)`, which adds the
load to the global force vector `f`. Loads do not depend on the design, so
`simp_optimize` assembles them once before the optimization loop.
"""
abstract type AbstractLoadCondition end

# =============================================================================
# POINT LOAD
# =============================================================================

"""
    PointLoad

Constant point force distributed equally across specified nodes.

# Fields
- `dh::DofHandler`: Degree of freedom handler
- `nodes::Vector{Int}`: Vector of node IDs where force is applied
- `force_vector::Vector{Float64}`: Force vector [Fx, Fy, Fz] in Newtons

# Example
```julia
load = PointLoad(dh, [1, 2, 3], [0.0, -1000.0, 0.0])
```
"""
struct PointLoad <: AbstractLoadCondition
    dh::DofHandler
    nodes::Vector{Int}
    force_vector::Vector{Float64}
end

# =============================================================================
# SURFACE TRACTION LOAD
# =============================================================================

"""
    SurfaceTractionLoad

Surface traction on boundary facets using proper Gauss quadrature integration.
Most accurate method for distributed loads on surfaces.

# Fields
- `dh::DofHandler`: Degree of freedom handler
- `grid::Grid`: Computational mesh
- `boundary_facets::Vector{Tuple{Int,Int}}`: (cell_id, local_face_id) pairs
- `traction_function::Function`: Function (x, y, z) -> [Tx, Ty, Tz]

# Constructors
- `SurfaceTractionLoad(dh, grid, nodes, traction_fn)`: position-dependent traction
- `SurfaceTractionLoad(dh, grid, nodes, force_vector)`: uniform traction t = F / A whose
  resultant over the selected surface (area A) is the force vector F [Fx, Fy, Fz]

The facets are the boundary facets whose nodes all lie in `nodes`
(see `get_boundary_facets`).

# Example
```julia
# Tangential traction on inner cylinder
g(x, y, z) = [100.0 * (-y), 100.0 * x, 0.0]
load = SurfaceTractionLoad(dh, grid, inner_nodes, g)

# Total force of 13 N in -z, spread uniformly over the surface
load = SurfaceTractionLoad(dh, grid, leg_nodes, [0.0, 0.0, -13.0])
```
"""
struct SurfaceTractionLoad <: AbstractLoadCondition
    dh::DofHandler
    grid::Grid
    boundary_facets::Vector{Tuple{Int,Int}}
    traction_function::Function

    # Constructor from node set
    function SurfaceTractionLoad(
        dh::DofHandler,
        grid::Grid,
        nodes::Set{Int},
        traction_fn::Function,
    )
        facets = get_boundary_facets(grid, nodes)
        new(dh, grid, facets, traction_fn)
    end

    # Constructor from node set and total force (uniform traction)
    function SurfaceTractionLoad(
        dh::DofHandler,
        grid::Grid,
        nodes::Set{Int},
        force_vector::Vector{Float64},
    )
        facets = get_boundary_facets(grid, nodes)
        area = calculate_surface_area(grid, facets)
        if area <= 0.0
            error("No boundary facets with nonzero area for surface traction.")
        end

        traction = Vec{3}(Tuple(force_vector ./ area))
        println("Uniform traction $(traction) on area $(area)")
        new(dh, grid, facets, (x, y, z) -> traction)
    end
end

# =============================================================================
# APPLY FUNCTIONS
# =============================================================================

"""
    apply_load_condition!(f, load::AbstractLoadCondition)

Apply load condition to global force vector.
"""
function apply_load_condition!(f::Vector{Float64}, load::PointLoad)
    apply_force!(f, load.dh, load.nodes, load.force_vector)
end

function apply_load_condition!(f::Vector{Float64}, load::SurfaceTractionLoad)
    apply_surface_traction!(
        f,
        load.dh,
        load.grid,
        load.boundary_facets,
        load.traction_function,
    )
end
