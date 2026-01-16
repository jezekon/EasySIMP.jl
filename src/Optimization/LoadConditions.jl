# src/Optimization/LoadConditions.jl
"""
LoadConditions.jl

Abstract load condition types for SIMP topology optimization.
Supports point loads, nodal traction, and surface traction with position-dependent functions.
"""

export AbstractLoadCondition, PointLoad, SurfaceTractionLoad
export apply_load_condition!

# =============================================================================
# ABSTRACT BASE TYPE
# =============================================================================

"""
    AbstractLoadCondition

Abstract base type for all load conditions in topology optimization.
Concrete implementations must define `apply_load_condition!(f, load)`.
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

# Constructor from tuple (backward compatibility with legacy format)
PointLoad(t::Tuple{DofHandler,Vector{Int},Vector{Float64}}) = PointLoad(t[1], t[2], t[3])

"""
    SurfaceTractionLoad

Position-dependent surface traction using proper Gauss quadrature integration.
Most accurate method for distributed loads on surfaces.

# Fields
- `dh::DofHandler`: Degree of freedom handler
- `grid::Grid`: Computational mesh
- `boundary_facets::Set{Tuple{Int,Int}}`: Set of (cell_id, local_face_id) tuples
- `traction_function::Function`: Function (x, y, z) -> [Tx, Ty, Tz]

# Example
```julia
# Tangential traction on inner cylinder
g(x, y, z) = [100.0 * (-y), 100.0 * x, 0.0]
inner_facets = get_boundary_facets(grid, inner_nodes)
load = SurfaceTractionLoad(dh, grid, inner_facets, g)
```
"""
struct SurfaceTractionLoad <: AbstractLoadCondition
    dh::DofHandler
    grid::Grid
    boundary_facets::Set{Tuple{Int,Int}}
    traction_function::Function
end

# Convenience constructor from node set
function SurfaceTractionLoad(
    dh::DofHandler,
    grid::Grid,
    nodes::Set{Int},
    traction_fn::Function,
)
    facets = get_boundary_facets(grid, nodes)
    return SurfaceTractionLoad(dh, grid, facets, traction_fn)
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

# Backward compatibility: handle tuple as PointLoad
function apply_load_condition!(
    f::Vector{Float64},
    load::Tuple{DofHandler,Vector{Int},Vector{Float64}},
)
    apply_force!(f, load[1], load[2], load[3])
end

# =============================================================================
# HELPER: Convert legacy format to new format
# =============================================================================

"""
    convert_to_load_condition(load)

Convert various load formats to AbstractLoadCondition.
Provides backward compatibility with tuple format.

# Supported input types
- `AbstractLoadCondition`: Returned as-is
- `Tuple{DofHandler, Vector{Int}, Vector{Float64}}`: Converted to PointLoad
"""
function convert_to_load_condition(load::AbstractLoadCondition)
    return load
end

function convert_to_load_condition(load::Tuple{DofHandler,Vector{Int},Vector{Float64}})
    return PointLoad(load[1], load[2], load[3])
end
