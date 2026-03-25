module FiniteElementAnalysis

using Ferrite
using LinearAlgebra
using SparseArrays
using StaticArrays
using ..Utils

export create_material_model,
    setup_problem,
    get_node_dofs,
    apply_variable_density_volume_force!,
    apply_fixed_boundary!,
    apply_sliding_boundary!,
    apply_force!,
    create_simp_material_model,
    assemble_stiffness_matrix_simp!,
    assemble_element_stiffness_matrix,
    assemble_element_stiffness_matrix!,
    calculate_stresses_simp,
    get_boundary_facets,
    apply_surface_traction!,
    AbstractLoadCondition,
    PointLoad,
    SurfaceTractionLoad,
    apply_load_condition!,
    calculate_element_volumes,
    create_volume_quadrature,
    initialize_element_cache

include("SelectNodesForBC.jl")
include("LoadConditions.jl")
export select_nodes_by_plane,
    select_nodes_by_circle, select_nodes_by_cylinder, select_nodes_by_arc

# =============================================================================
# UNIFIED MATERIAL CALCULATIONS
# =============================================================================

"""
    compute_lame_parameters(youngs_modulus::Float64, poissons_ratio::Float64)

Compute Lamé parameters (λ, μ) from Young's modulus and Poisson's ratio.

# Arguments
- `youngs_modulus`: Young's modulus E [Pa]
- `poissons_ratio`: Poisson's ratio ν [-]

# Returns
- `Tuple{Float64, Float64}`: (λ, μ) - Lamé's first and second parameters
"""
function compute_lame_parameters(youngs_modulus::Float64, poissons_ratio::Float64)
    λ = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    μ = youngs_modulus / (2 * (1 + poissons_ratio))
    return λ, μ
end

"""
    constitutive_relation(ε, λ, μ)

Apply linear elastic constitutive relation (Hooke's law).
"""
function constitutive_relation(ε, λ, μ)
    return λ * tr(ε) * one(ε) + 2μ * ε
end

# =============================================================================
# MATERIAL MODEL CREATION
# =============================================================================

"""
    create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)

Create material constants for a linearly elastic material.

# Returns
- `Tuple{Float64, Float64}`: (λ, μ) - Lamé parameters
"""
function create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)
    return compute_lame_parameters(youngs_modulus, poissons_ratio)
end

"""
    create_simp_material_model(E0, nu, Emin=1e-6, p=3.0)

Create a SIMP (Solid Isotropic Material with Penalization) material model.

# Arguments
- `E0`: Base material Young's modulus
- `nu`: Poisson's ratio
- `Emin`: Minimum Young's modulus (default: 1e-6)
- `p`: Penalization power (default: 3.0)

# Returns
- Function mapping density ρ to Lamé parameters (λ, μ)

# SIMP Model
E(ρ) = Emin + (E0 - Emin) * ρ^p
"""
function create_simp_material_model(
    E0::Float64,
    nu::Float64,
    Emin::Float64 = 1e-6,
    p::Float64 = 3.0,
)
    function material_for_density(density::Float64)
        E = Emin + (E0 - Emin) * density^p
        return compute_lame_parameters(E, nu)
    end

    return material_for_density
end

# =============================================================================
# PROBLEM SETUP
# =============================================================================

"""
    setup_problem(grid::Grid, interpolation_order::Int=1)

Set up the finite element problem for the given grid.
Automatically detects cell type (tetrahedron or hexahedron).

# Returns
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `K`: Allocated sparse stiffness matrix
- `f`: Allocated force vector
"""
function setup_problem(grid::Grid, interpolation_order::Int = 1)
    dim = 3

    cell_type = typeof(getcells(grid, 1))

    if cell_type <: Ferrite.Hexahedron
        println("Setting up problem with hexahedral elements")
        ip = Lagrange{RefHexahedron,interpolation_order}()^dim
        qr = QuadratureRule{RefHexahedron}(2)
    else
        println("Setting up problem with tetrahedral elements")
        ip = Lagrange{RefTetrahedron,interpolation_order}()^dim
        qr = QuadratureRule{RefTetrahedron}(2)
    end

    cellvalues = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    n_dofs = ndofs(dh)
    print_success("FEM setup complete: $n_dofs DOFs")
    K = allocate_matrix(dh)
    f = zeros(n_dofs)

    return dh, cellvalues, K, f
end

# =============================================================================
# STIFFNESS MATRIX ASSEMBLY
# =============================================================================

"""
    assemble_element_stiffness_matrix!(ke, cellvalues, λ, μ)

Compute element stiffness matrix in-place (no allocation).

# Arguments
- `ke`: Pre-allocated element stiffness matrix (modified in-place)
- `cellvalues`: CellValues object (must be reinitialized)
- `λ`: Lamé's first parameter
- `μ`: Lamé's second parameter
"""
function assemble_element_stiffness_matrix!(ke::Matrix{Float64}, cellvalues, λ, μ)
    n_basefuncs = getnbasefunctions(cellvalues)
    fill!(ke, 0.0)

    for q_point = 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)

        for i = 1:n_basefuncs
            ∇Ni = shape_gradient(cellvalues, q_point, i)
            εi = symmetric(∇Ni)

            for j = 1:n_basefuncs
                ∇Nj = shape_gradient(cellvalues, q_point, j)
                εj = symmetric(∇Nj)
                σ = constitutive_relation(εj, λ, μ)
                ke[i, j] += (εi ⊡ σ) * dΩ
            end
        end
    end
end

"""
    assemble_element_stiffness_matrix(cellvalues, λ, μ)

Compute element stiffness matrix (allocating version).

# Arguments
- `cellvalues`: CellValues object (must be reinitialized)
- `λ`: Lamé's first parameter
- `μ`: Lamé's second parameter

# Returns
- `ke`: Element stiffness matrix
"""
function assemble_element_stiffness_matrix(cellvalues, λ, μ)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    assemble_element_stiffness_matrix!(ke, cellvalues, λ, μ)
    return ke
end

"""
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)

Assemble global stiffness matrix using SIMP material model.
Simple allocating version for one-off assembly (tests, initial setup).

# Arguments
- `K`: Global stiffness matrix (modified in-place)
- `f`: Global force vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `material_model`: Function mapping density to (λ, μ)
- `density_data`: Vector of density values for each cell
"""
function assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        density = density_data[cellid(cell)]
        λ, μ = material_model(density)
        assemble_element_stiffness_matrix!(ke, cellvalues, λ, μ)
        assemble!(assembler, celldofs(cell), ke, fe)
    end
end

# =============================================================================
# NODE DOF MAPPING
# =============================================================================

"""
    get_node_dofs(dh::DofHandler)

Build mapping from node IDs to their global DOF indices.

# Returns
- `Dict{Int, Vector{Int}}`: Node ID -> Vector of DOF indices
"""
function get_node_dofs(dh::DofHandler)
    node_to_dofs = Dict{Int,Vector{Int}}()

    for cell in CellIterator(dh)
        cell_nodes = cell.nodes
        cell_dofs = celldofs(cell)

        nodes_per_cell = length(cell_nodes)
        dofs_per_node = length(cell_dofs) ÷ nodes_per_cell

        for (local_node_idx, global_node_idx) in enumerate(cell_nodes)
            start_dof = (local_node_idx - 1) * dofs_per_node + 1
            end_dof = local_node_idx * dofs_per_node
            local_dofs = cell_dofs[start_dof:end_dof]

            if !haskey(node_to_dofs, global_node_idx)
                node_to_dofs[global_node_idx] = local_dofs
            end
        end
    end

    return node_to_dofs
end

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

"""
    apply_fixed_boundary!(K, f, dh, nodes)

Apply fixed boundary conditions (all DOFs = 0) to specified nodes.

# Returns
- `ConstraintHandler` with applied constraints
"""
function apply_fixed_boundary!(K, f, dh, nodes)
    dim = 3

    ch = ConstraintHandler(dh)

    for d = 1:dim
        dbc = Dirichlet(:u, nodes, (x, t) -> 0.0, d)
        add!(ch, dbc)
    end

    close!(ch)
    update!(ch, 0.0)
    apply!(K, f, ch)

    println("Applied fixed boundary conditions to $(length(nodes)) nodes")
    return ch
end

"""
    apply_sliding_boundary!(K, f, dh, nodes, fixed_dofs)

Apply sliding boundary conditions to specified nodes.

# Arguments
- `K`: Global stiffness matrix
- `f`: Global force vector
- `dh`: DofHandler
- `nodes`: Set or Array of node IDs
- `fixed_dofs`: Array of direction indices to fix (1=X, 2=Y, 3=Z)

# Returns
- `ConstraintHandler` with applied constraints
"""
function apply_sliding_boundary!(K, f, dh, nodes, fixed_dofs)
    ch = ConstraintHandler(dh)

    for d in fixed_dofs
        dbc = Dirichlet(:u, nodes, (x, t) -> 0.0, d)
        add!(ch, dbc)
    end

    close!(ch)
    update!(ch, 0.0)
    apply!(K, f, ch)

    println("Applied sliding boundary to $(length(nodes)) nodes, fixed DOFs: $fixed_dofs")
    return ch
end

# =============================================================================
# FORCE APPLICATION
# =============================================================================

"""
    apply_force!(f, dh, nodes, force_vector)

Apply a force distributed equally across specified nodes.

# Arguments
- `f`: Global force vector (modified in-place)
- `dh`: DofHandler
- `nodes`: Array or Set of node IDs
- `force_vector`: Force vector [Fx, Fy, Fz] in Newtons
"""
function apply_force!(f, dh, nodes, force_vector)
    if isempty(nodes)
        error("No nodes provided for force application.")
    end

    node_to_dofs = get_node_dofs(dh)
    force_per_node = force_vector ./ length(nodes)

    for node_id in nodes
        if haskey(node_to_dofs, node_id)
            dofs = node_to_dofs[node_id]

            for (i, component) in enumerate(force_per_node)
                if i <= length(dofs)
                    f[dofs[i]] += component
                end
            end
        end
    end
end

"""
    apply_surface_traction!(f, dh, grid, boundary_facets, traction_function)

Apply position-dependent surface traction using Gauss quadrature.

# Arguments
- `f`: Global force vector (modified in-place)
- `dh`: DofHandler
- `grid`: Computational mesh
- `boundary_facets`: Set of (cell_id, local_face_id) tuples
- `traction_function`: Function (x, y, z) -> [Tx, Ty, Tz]
"""
function apply_surface_traction!(
    f,
    dh::DofHandler,
    grid::Grid,
    boundary_facets,
    traction_function::Function,
)
    cell_type = typeof(getcells(grid, 1))

    if cell_type <: Ferrite.Hexahedron
        ip = Lagrange{RefHexahedron,1}()^3
        qr_face = FacetQuadratureRule{RefHexahedron}(2)
    elseif cell_type <: Ferrite.Tetrahedron
        ip = Lagrange{RefTetrahedron,1}()^3
        qr_face = FacetQuadratureRule{RefTetrahedron}(2)
    else
        error("Unsupported cell type: $cell_type")
    end

    facevalues = FacetValues(qr_face, ip)

    n_basefuncs = getnbasefunctions(facevalues)
    fe = zeros(n_basefuncs)

    total_force = zeros(3)

    for (cell_id, local_face_id) in boundary_facets
        coords = getcoordinates(grid, cell_id)
        reinit!(facevalues, coords, local_face_id)

        fill!(fe, 0.0)

        for q_point = 1:getnquadpoints(facevalues)
            dΓ = getdetJdV(facevalues, q_point)
            x_qp = spatial_coordinate(facevalues, q_point, coords)
            traction = traction_function(x_qp[1], x_qp[2], x_qp[3])

            for i = 1:n_basefuncs
                N = shape_value(facevalues, q_point, i)
                fe[i] += (N ⋅ traction) * dΓ
            end

            total_force .+= traction * dΓ
        end

        cell_dofs = celldofs(dh, cell_id)
        for (i, dof) in enumerate(cell_dofs)
            f[dof] += fe[i]
        end
    end
end

"""
    get_boundary_facets(grid, nodes)

Identify boundary facets containing all nodes from the given set.

# Returns
- `Set{Tuple{Int,Int}}`: Set of (cell_id, local_face_id) pairs
"""
function get_boundary_facets(grid::Grid, nodes::Set{Int})
    boundary_facets = Set{Tuple{Int,Int}}()

    for cell_id = 1:getncells(grid)
        cell = getcells(grid, cell_id)
        face_nodes_list = get_face_nodes(cell)

        for (local_face_id, face_nodes) in enumerate(face_nodes_list)
            global_face_nodes = [cell.nodes[i] for i in face_nodes]

            if all(n -> n in nodes, global_face_nodes)
                push!(boundary_facets, (cell_id, local_face_id))
            end
        end
    end

    println("Found $(length(boundary_facets)) boundary facets")
    return boundary_facets
end

function get_face_nodes(cell::Ferrite.Tetrahedron)
    return [(1, 2, 3), (1, 2, 4), (2, 3, 4), (1, 3, 4)]
end

function get_face_nodes(cell::Ferrite.Hexahedron)
    return [
        (1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)
    ]
end

"""
    apply_variable_density_volume_force!(f, dh, cellvalues, body_force_vector, density_data)

Apply volume forces with variable density distribution (for SIMP).
"""
function apply_variable_density_volume_force!(f, dh, cellvalues, body_force_vector, density_data)
    n_basefuncs = getnbasefunctions(cellvalues)
    fe_body = zeros(n_basefuncs)
    total_force_applied = zeros(3)

    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        density = density_data[cell_id]

        if density < 1e-6
            continue
        end

        reinit!(cellvalues, cell)
        fill!(fe_body, 0.0)

        cell_dofs = celldofs(cell)

        for q_point = 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i = 1:n_basefuncs
                N_vec = shape_value(cellvalues, q_point, i)
                dofs_per_node = 3
                dof_component = mod(i - 1, dofs_per_node) + 1
                N_scalar = N_vec[dof_component]

                body_force_contribution = density * body_force_vector[dof_component] * N_scalar * dΩ
                fe_body[i] += body_force_contribution
                total_force_applied[dof_component] += body_force_contribution
            end
        end

        for (local_dof, global_dof) in enumerate(cell_dofs)
            f[global_dof] += fe_body[local_dof]
        end
    end

    println("Applied variable density volume force")
    println("  Total force applied: $total_force_applied N")
end

# =============================================================================
# STRESS CALCULATION
# =============================================================================

"""
    calculate_stress_at_quadrature_points(u_cell, cellvalues, λ, μ)

Calculate stresses at quadrature points for a single element.
"""
function calculate_stress_at_quadrature_points(u_cell, cellvalues, λ, μ)
    n_qpoints = getnquadpoints(cellvalues)
    cell_stresses = Vector{SymmetricTensor{2,3,Float64}}(undef, n_qpoints)
    avg_stress = zero(SymmetricTensor{2,3,Float64})

    for q_point = 1:n_qpoints
        grad_u = function_gradient(cellvalues, q_point, u_cell)
        ε = symmetric(grad_u)
        σ = constitutive_relation(ε, λ, μ)
        cell_stresses[q_point] = σ
        avg_stress += σ
    end

    if n_qpoints > 0
        avg_stress = avg_stress / n_qpoints
    end

    return cell_stresses, avg_stress
end

"""
    calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)

Calculate stress field using SIMP material model.

# Returns
- `stress_field`: Dict of stress tensors per cell
- `max_von_mises`: Maximum von Mises stress
- `max_vm_cell_id`: Cell ID with maximum von Mises stress
"""
function calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)
    stress_field = Dict{Int,Vector{SymmetricTensor{2,3,Float64}}}()
    max_von_mises = 0.0
    max_vm_cell_id = 0

    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        cell_dofs = celldofs(cell)
        u_cell = u[cell_dofs]

        density = density_data[cell_id]
        λ, μ = material_model(density)

        reinit!(cellvalues, cell)

        cell_stresses, avg_stress = calculate_stress_at_quadrature_points(u_cell, cellvalues, λ, μ)

        cell_von_mises = sqrt(3/2 * (dev(avg_stress) ⊡ dev(avg_stress)))

        if cell_von_mises > max_von_mises
            max_von_mises = cell_von_mises
            max_vm_cell_id = cell_id
        end

        stress_field[cell_id] = cell_stresses
    end

    println("Maximum von Mises stress: $max_von_mises at cell $max_vm_cell_id")

    return stress_field, max_von_mises, max_vm_cell_id
end

# =============================================================================
# CACHED ASSEMBLY FUNCTIONS
# =============================================================================

"""
    initialize_element_cache(dh, cellvalues, material_model, n_cells)

Initialize element stiffness matrix cache with unit matrices.
"""
function initialize_element_cache(dh, cellvalues, material_model, n_cells)
    n_basefuncs = getnbasefunctions(cellvalues)
    unit_matrices = Vector{Matrix{Float64}}(undef, n_cells)
    λ_unit, μ_unit = material_model(1.0)

    print_info("Computing element stiffness matrix cache...")

    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        reinit!(cellvalues, cell)
        ke_unit = assemble_element_stiffness_matrix(cellvalues, λ_unit, μ_unit)
        unit_matrices[cell_id] = ke_unit
    end

    # Store unit material parameters for scaling
    cache = Dict{Symbol,Any}()
    cache[:unit_matrices] = unit_matrices
    cache[:λ_unit] = λ_unit
    cache[:μ_unit] = μ_unit

    print_success("Element cache initialized: $(n_cells) unit matrices")
    return cache
end

"""
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities, cache, ke_buffer, fe_buffer)

Assemble global stiffness matrix with in-place operations to avoid allocations.
"""
function assemble_stiffness_matrix_simp!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    densities,
    cache,
    ke_buffer,
    fe_buffer,
)
    if cache !== nothing
        assemble_with_cache!(
            K,
            f,
            dh,
            material_model,
            densities,
            cache,
            ke_buffer,
            fe_buffer,
        )
    else
        assemble_variable_material!(
            K,
            f,
            dh,
            cellvalues,
            material_model,
            densities,
            ke_buffer,
            fe_buffer,
        )
    end
end

"""
    assemble_with_cache!(K, f, dh, material_model, densities, cache, ke_buffer, fe_buffer)

Assembly using cached unit matrices with in-place scaling.
"""
function assemble_with_cache!(
    K,
    f,
    dh,
    material_model,
    densities,
    cache,
    ke_buffer,
    fe_buffer,
)
    unit_matrices = cache[:unit_matrices]
    λ_unit = cache[:λ_unit]

    fill!(fe_buffer, 0.0)
    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        density = densities[cell_id]

        # Get current material parameters
        λ_current, _ = material_model(density)
        scaling_factor = λ_current / λ_unit

        # In-place scaling: ke_buffer = scaling_factor * unit_matrix
        unit_ke = unit_matrices[cell_id]
        @inbounds for j = 1:size(ke_buffer, 2)
            for i = 1:size(ke_buffer, 1)
                ke_buffer[i, j] = scaling_factor * unit_ke[i, j]
            end
        end

        assemble!(assembler, celldofs(cell), ke_buffer, fe_buffer)
    end
end

"""
    assemble_variable_material!(K, f, dh, cellvalues, material_model, densities, ke_buffer, fe_buffer)

Assembly for variable material properties without caching.
"""
function assemble_variable_material!(
    K,
    f,
    dh,
    cellvalues,
    material_model,
    densities,
    ke_buffer,
    fe_buffer,
)
    fill!(fe_buffer, 0.0)
    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)

        cell_id = cellid(cell)
        density = densities[cell_id]
        λ, μ = material_model(density)

        # Compute element stiffness into buffer
        assemble_element_stiffness_matrix!(ke_buffer, cellvalues, λ, μ)
        assemble!(assembler, celldofs(cell), ke_buffer, fe_buffer)
    end
end

# =============================================================================
# ELEMENT VOLUMES AND VOLUME QUADRATURE
# =============================================================================

"""
    calculate_element_volumes(grid, cellvalues)

Calculate volume of each element in the grid.
"""
function calculate_element_volumes(grid::Grid, cellvalues)
    n_cells = getncells(grid)
    element_volumes = zeros(n_cells)

    for cell_idx = 1:n_cells
        coords = getcoordinates(grid, cell_idx)
        reinit!(cellvalues, coords)

        volume = 0.0
        for q_point = 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            volume += dΩ
        end
        element_volumes[cell_idx] = volume
    end

    return element_volumes
end

"""
    create_volume_quadrature(grid)

Create CellValues for volume calculation with 3rd order quadrature.
"""
function create_volume_quadrature(grid::Grid{dim}) where {dim}
    cell = getcells(grid, 1)

    if cell isa Hexahedron
        ip = Lagrange{RefHexahedron,1}()
        qr = QuadratureRule{RefHexahedron}(3)
    elseif cell isa Tetrahedron
        ip = Lagrange{RefTetrahedron,1}()
        qr = QuadratureRule{RefTetrahedron}(3)
    elseif cell isa Quadrilateral && dim == 2
        ip = Lagrange{RefQuadrilateral,1}()
        qr = QuadratureRule{RefQuadrilateral}(3)
    elseif cell isa Triangle && dim == 2
        ip = Lagrange{RefTriangle,1}()
        qr = QuadratureRule{RefTriangle}(3)
    else
        error("Unsupported cell type: $(typeof(cell))")
    end

    return CellValues(qr, ip)
end

end # module
