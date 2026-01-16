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

# =============================================================================
# SURFACE TRACTION LOAD
# =============================================================================

"""
    SurfaceTractionLoad

Position-dependent surface traction using proper Gauss quadrature integration.
Most accurate method for distributed loads on surfaces.
FacetValues are cached for performance to avoid memory fragmentation.

# Fields
- `dh::DofHandler`: Degree of freedom handler
- `grid::Grid`: Computational mesh
- `boundary_facets::Set{Tuple{Int,Int}}`: Set of (cell_id, local_face_id) tuples
- `traction_function::Function`: Function (x, y, z) -> [Tx, Ty, Tz]
- `facevalues::FacetValues`: Cached FacetValues for integration

# Example
```julia
# Tangential traction on inner cylinder
g(x, y, z) = [100.0 * (-y), 100.0 * x, 0.0]
inner_facets = get_boundary_facets(grid, inner_nodes)
load = SurfaceTractionLoad(dh, grid, inner_nodes, g)
```
"""
struct SurfaceTractionLoad <: AbstractLoadCondition
    dh::DofHandler
    grid::Grid
    boundary_facets::Set{Tuple{Int,Int}}
    traction_function::Function
    facevalues::FacetValues

    # Constructor from node set
    function SurfaceTractionLoad(
        dh::DofHandler,
        grid::Grid,
        nodes::Set{Int},
        traction_fn::Function,
    )
        facets = get_boundary_facets(grid, nodes)

        # Create FacetValues once for reuse
        cell_type = typeof(getcells(grid, 1))

        ip, qr = if cell_type <: Ferrite.Hexahedron
            Lagrange{RefHexahedron,1}()^3, FacetQuadratureRule{RefHexahedron}(2)
        elseif cell_type <: Ferrite.Tetrahedron
            Lagrange{RefTetrahedron,1}()^3, FacetQuadratureRule{RefTetrahedron}(2)
        else
            error("Unsupported cell type: $cell_type")
        end

        new(dh, grid, facets, traction_fn, FacetValues(qr, ip))
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
    facevalues = load.facevalues
    grid = load.grid
    dh = load.dh

    n_basefuncs = getnbasefunctions(facevalues)
    fe = zeros(n_basefuncs)

    # Iterate over boundary facets
    for (cell_id, local_face_id) in load.boundary_facets
        # Get cell and reinitialize face values
        coords = getcoordinates(grid, cell_id)
        reinit!(facevalues, coords, local_face_id)
        fill!(fe, 0.0)

        # Integrate traction over the face
        for q_point = 1:getnquadpoints(facevalues)
            dΓ = getdetJdV(facevalues, q_point)

            # Get spatial coordinates at quadrature point
            x_qp = spatial_coordinate(facevalues, q_point, coords)

            # Evaluate traction at this point
            traction = load.traction_function(x_qp[1], x_qp[2], x_qp[3])

            # Assemble: fe += N^T * traction * dΓ
            for i = 1:n_basefuncs
                N = shape_value(facevalues, q_point, i)
                fe[i] += (N ⋅ traction) * dΓ
            end
        end

        # Get global DOFs and assemble
        cell_dofs = celldofs(dh, cell_id)
        for (i, dof) in enumerate(cell_dofs)
            f[dof] += fe[i]
        end
    end
end
