module FiniteElementAnalysis

using Ferrite
using LinearAlgebra
using SparseArrays
using StaticArrays
using ..Utils

export create_material_model, setup_problem,
       get_node_dofs, apply_variable_density_volume_force!,
       apply_fixed_boundary!, apply_sliding_boundary!, apply_force!, solve_system,
       calculate_stresses, create_simp_material_model, assemble_stiffness_matrix_simp!,
       calculate_stresses_simp, solve_system_simp

include("SelectNodesForBC.jl")
export select_nodes_by_plane, select_nodes_by_circle
# ============================================================================
# UNIFIED MATERIAL CALCULATIONS - Odstranění duplicit
# ============================================================================

"""
    compute_lame_parameters(youngs_modulus::Float64, poissons_ratio::Float64)

Unified function to compute Lamé parameters from Young's modulus and Poisson's ratio.
Eliminates duplication across the codebase.
"""
function compute_lame_parameters(youngs_modulus::Float64, poissons_ratio::Float64)
    λ = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    μ = youngs_modulus / (2 * (1 + poissons_ratio))
    return λ, μ
end

"""
    constitutive_relation(ε, λ, μ)

Applies linear elastic relationship between strain and stress (Hooke's law).
"""
function constitutive_relation(ε, λ, μ)
    return λ * tr(ε) * one(ε) + 2μ * ε
end

# ============================================================================
# MATERIAL MODEL CREATION - Zachované původní API
# ============================================================================

"""
    create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)

Creates material constants for a linearly elastic material.

Parameters:
- `youngs_modulus`: Young's modulus in Pa
- `poissons_ratio`: Poisson's ratio

Returns:
- lambda and mu coefficients for Hooke's law
"""
function create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)
    return compute_lame_parameters(youngs_modulus, poissons_ratio)
end

"""
    create_simp_material_model(E0::Float64, nu::Float64, Emin::Float64=1e-6, p::Float64=3.0)

Creates a material model using the SIMP (Solid Isotropic Material with Penalization) approach.

Parameters:
- `E0`: Base material Young's modulus
- `nu`: Poisson's ratio
- `Emin`: Minimum Young's modulus (default: 1e-6)
- `p`: Penalization power (default: 3.0)

Returns:
- Function mapping density to Lamé parameters (λ, μ)
"""
function create_simp_material_model(E0::Float64, nu::Float64, Emin::Float64=1e-6, p::Float64=3.0)
    function material_for_density(density::Float64)
        # SIMP model: E(ρ) = Emin + (E0 - Emin) * ρ^p
        E = Emin + (E0 - Emin) * density^p
        return compute_lame_parameters(E, nu)
    end
    
    return material_for_density
end

# ============================================================================
# PROBLEM SETUP
# ============================================================================

"""
    setup_problem(grid::Grid, interpolation_order::Int=1)

Sets up the finite element problem for the given grid.
Automatically detects the cell type (tetrahedron or hexahedron) and creates appropriate interpolation.
"""
function setup_problem(grid::Grid, interpolation_order::Int=1)
    dim = 3  # 3D problem
    
    cell_type = typeof(getcells(grid, 1))
    
    if cell_type <: Ferrite.Hexahedron
        println("Setting up problem with hexahedral elements")
        ip = Lagrange{RefHexahedron, interpolation_order}()^dim
        qr = QuadratureRule{RefHexahedron}(2)
    else
        println("Setting up problem with tetrahedral elements")
        ip = Lagrange{RefTetrahedron, interpolation_order}()^dim
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

# ============================================================================
# STIFFNESS MATRIX ASSEMBLY - Zachované původní funkce
# ============================================================================

"""
    assemble_element_stiffness_matrix(cellvalues, λ, μ)

Helper function to compute element stiffness matrix for given material properties.
Eliminates code duplication between constant and variable material assembly.
"""
function assemble_element_stiffness_matrix(cellvalues, λ, μ)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        
        for i in 1:n_basefuncs
            ∇Ni = shape_gradient(cellvalues, q_point, i)
            εi = symmetric(∇Ni)
            
            for j in 1:n_basefuncs
                ∇Nj = shape_gradient(cellvalues, q_point, j)
                εj = symmetric(∇Nj)
                σ = constitutive_relation(εj, λ, μ)
                ke[i, j] += (εi ⊡ σ) * dΩ
            end
        end
    end
    
    return ke
end


"""
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data, cache=nothing)

Enhanced SIMP assembly with optional element stiffness matrix caching.

Parameters:
- `K`: global stiffness matrix (modified in-place)
- `f`: global load vector (modified in-place) 
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `material_model`: Function mapping density to material parameters (λ, μ)
- `density_data`: Vector with density values for each cell
- `cache`: Optional Dict for caching unit element matrices (default: nothing)
"""
function assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data, cache=nothing)
    if cache !== nothing && haskey(cache, :unit_matrices)
        assemble_with_cache(K, f, dh, cellvalues, material_model, density_data, cache)
    else
        assemble_variable_material(K, f, dh, cellvalues, material_model, density_data)
        
        # Initialize cache if requested
        if cache !== nothing
            initialize_cache(cache, dh, cellvalues, material_model, length(density_data))
        end
    end
end

"""
    assemble_variable_material(K, f, dh, cellvalues, material_model, density_data)

Assembly for variable material properties without caching.
"""
function assemble_variable_material(K, f, dh, cellvalues, material_model, density_data)
    n_basefuncs = getnbasefunctions(cellvalues)
    fe = zeros(n_basefuncs)
    
    assembler = start_assemble(K, f)
    
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(fe, 0.0)
        
        cell_id = cellid(cell)
        density = density_data[cell_id]
        λ, μ = material_model(density)
        
        ke = assemble_element_stiffness_matrix(cellvalues, λ, μ)
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    
    println("Stiffness matrix assembled successfully with variable material properties")
end

"""
    assemble_with_cache(K, f, dh, cellvalues, material_model, density_data, cache)

Assembly using cached unit matrices for improved performance.
"""
function assemble_with_cache(K, f, dh, cellvalues, material_model, density_data, cache)
    n_basefuncs = getnbasefunctions(cellvalues)
    fe = zeros(n_basefuncs)
    
    unit_matrices = cache[:unit_matrices]
    
    fill!(K.nzval, 0.0)
    assembler = start_assemble(K, f)
    
    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        density = density_data[cell_id]
        
        # Získej skalární faktor z material_model
        λ_current, μ_current = material_model(density)
        λ_unit, μ_unit = material_model(1.0)
        
        # Scaling factor je poměr current/unit
        scaling_factor = λ_current / λ_unit  # λ i μ mají stejný scaling
        
        # Škáluj cached unit matrix
        ke = scaling_factor * unit_matrices[cell_id]
        
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    
    println("Stiffness matrix assembled with cached unit matrices: $(length(density_data)) elements")
end

"""
    initialize_cache(cache, dh, cellvalues, material_model, n_cells)

Initialize element stiffness matrix cache with unit matrices.
FIXED: Uses actual parameters from material_model instead of hardcoded p=3.
"""
function initialize_cache(cache, dh, cellvalues, material_model, n_cells)
    # PŘÍMO EXTRAHUJ PARAMETRY Z OptimizationParameters místo odhadu
    # (parametry se předají z optimization)
    # Toto se upraví v ZMĚNĚ 3
    
    n_basefuncs = getnbasefunctions(cellvalues)
    unit_matrices = Vector{Matrix{Float64}}(undef, n_cells)
    
    print_info("Computing element stiffness matrix cache...")
    
    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        reinit!(cellvalues, cell)
        
        # Použij jednotkové materiálové parametry (budou se později skalovat)
        λ1, μ1 = material_model(1.0)  # Pro hustotu = 1
        ke_unit = assemble_element_stiffness_matrix(cellvalues, λ1, μ1)
        unit_matrices[cell_id] = ke_unit
    end
    
    cache[:unit_matrices] = unit_matrices
    print_success("Element cache initialized: $(n_cells) unit matrices")
end

# ============================================================================
# NODE SELECTION FUNCTIONS
# ============================================================================

function get_node_dofs(dh::DofHandler)
    node_to_dofs = Dict{Int, Vector{Int}}()
    
    # For each cell, get the mapping of nodes to DOFs
    for cell in CellIterator(dh)
        cell_nodes = cell.nodes
        cell_dofs = celldofs(cell)
        
        # Assume that for each node we have 'dim' DOFs (one for each direction)
        # and they are arranged sequentially for each node
        nodes_per_cell = length(cell_nodes)
        dofs_per_node = length(cell_dofs) ÷ nodes_per_cell
        
        # For each node in the cell
        for (local_node_idx, global_node_idx) in enumerate(cell_nodes)
            # Calculate the range of DOFs for this node within the cell
            start_dof = (local_node_idx - 1) * dofs_per_node + 1
            end_dof = local_node_idx * dofs_per_node
            local_dofs = cell_dofs[start_dof:end_dof]
            
            # Add DOFs to the dictionary
            if !haskey(node_to_dofs, global_node_idx)
                node_to_dofs[global_node_idx] = local_dofs
            end
        end
    end
    
    return node_to_dofs
end

"""
    apply_fixed_boundary!(K, f, dh, nodes)

Applies fixed boundary conditions (all DOFs fixed) to the specified nodes.

Parameters:
- `K`: global stiffness matrix
- `f`: global load vector
- `dh`: DofHandler
- `nodes`: Set or Array of node IDs to be fixed

Returns:
- ConstraintHandler with the applied constraints
"""
function apply_fixed_boundary!(K, f, dh, nodes)
    dim = 3  # 3D problem
    
    # Create constraint handler
    ch = ConstraintHandler(dh)
    
    for d in 1:dim
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

Applies sliding boundary conditions to the specified nodes,
allowing movement only in certain directions.

Parameters:
- `K`: global stiffness matrix
- `f`: global load vector
- `dh`: DofHandler
- `nodes`: Set or Array of node IDs for the sliding boundary
- `fixed_dofs`: Array of direction indices to fix (1=x, 2=y, 3=z)

Returns:
- ConstraintHandler with the applied constraints
"""
function apply_sliding_boundary!(K, f, dh, nodes, fixed_dofs)
    # Create constraint handler
    ch = ConstraintHandler(dh)
    
    # Apply Dirichlet boundary conditions only for specified directions
    for d in fixed_dofs
        dbc = Dirichlet(:u, nodes, (x, t) -> 0.0, d)
        add!(ch, dbc)
    end
    
    close!(ch)
    update!(ch, 0.0)
    apply!(K, f, ch)
    
    println("Applied sliding boundary conditions to $(length(nodes)) nodes, fixing DOFs: $fixed_dofs")
    return ch
end

"""
    apply_force!(f, dh, nodes, force_vector)

Applies a force to the specified nodes.

Parameters:
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `nodes`: Array or Set of node IDs where force is applied
- `force_vector`: Force vector [Fx, Fy, Fz] in Newtons

Returns:
- nothing (modifies f in-place)
"""
function apply_force!(f, dh, nodes, force_vector)
    if isempty(nodes)
        error("No nodes provided for force application.")
    end
    
    # Get mapping from nodes to DOFs
    node_to_dofs = get_node_dofs(dh)
    
    # Calculate force per node
    force_per_node = force_vector ./ length(nodes)
    
    # Apply force to each node
    for node_id in nodes
        if haskey(node_to_dofs, node_id)
            dofs = node_to_dofs[node_id]
            
            # Apply force components to respective DOFs
            for (i, component) in enumerate(force_per_node)
                if i <= length(dofs)
                    f[dofs[i]] += component
                end
            end
        end
    end
    
    println("Applied force $force_vector distributed over $(length(nodes)) nodes")
end

"""
    apply_variable_density_volume_force!(f, dh, cellvalues, body_force_vector, density_data)

Applies volume forces with variable density distribution (for SIMP topology optimization).

Parameters:
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `body_force_vector`: Body force per unit mass [Fx, Fy, Fz] in N/kg
- `density_data`: Vector with density values for each cell

Returns:
- nothing (modifies f in-place)
"""
function apply_variable_density_volume_force!(f, dh, cellvalues, body_force_vector, density_data)
    # Number of basis functions per element
    n_basefuncs = getnbasefunctions(cellvalues)
    
    # Element load vector for body forces
    fe_body = zeros(n_basefuncs)
    
    # Track total applied force
    total_force_applied = zeros(3)
    
    # Iterate over all cells
    for cell in CellIterator(dh)
        # Get cell ID and corresponding density
        cell_id = cellid(cell)
        density = density_data[cell_id]
        
        # Skip if density is negligible (for SIMP optimization)
        if density < 1e-6
            continue
        end
        
        # Reinitialize cell values
        reinit!(cellvalues, cell)
        fill!(fe_body, 0.0)
        
        # Get cell DOFs
        cell_dofs = celldofs(cell)
        
        # Integrate body force over element volume
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                # Get vector shape function value
                N_vec = shape_value(cellvalues, q_point, i)
                
                # Calculate DOF component
                dofs_per_node = 3
                dof_component = mod(i - 1, dofs_per_node) + 1
                
                # Extract scalar component
                N_scalar = N_vec[dof_component]
                
                # Apply variable density body force
                body_force_contribution = density * body_force_vector[dof_component] * N_scalar * dΩ
                fe_body[i] += body_force_contribution
                
                # Track total force
                total_force_applied[dof_component] += body_force_contribution
            end
        end
        
        # Add to global load vector
        for (local_dof, global_dof) in enumerate(cell_dofs)
            f[global_dof] += fe_body[local_dof]
        end
    end
    
    println("Applied variable density volume force")
    println("Total force applied: $total_force_applied N")
end

# ============================================================================
# STRESS CALCULATION - Zachované původní funkce s helper functions
# ============================================================================

"""
    calculate_stress_at_quadrature_points(u_cell, cellvalues, λ, μ)

Helper function to calculate stresses at quadrature points.
Eliminates duplication between constant and variable material stress calculations.
"""
function calculate_stress_at_quadrature_points(u_cell, cellvalues, λ, μ)
    n_qpoints = getnquadpoints(cellvalues)
    cell_stresses = Vector{SymmetricTensor{2, 3, Float64}}(undef, n_qpoints)
    avg_stress = zero(SymmetricTensor{2, 3, Float64})
    
    for q_point in 1:n_qpoints
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

Calculate stress field from displacement solution, using variable material properties.

Parameters:
- `u`: displacement vector
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `material_model`: Function mapping density to material parameters (λ, μ)
- `density_data`: Vector with density values for each cell

Returns:
- Tuple with stress field, maximum von Mises stress, and cell ID where max occurs
"""
function calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)
    stress_field = Dict{Int, Vector{SymmetricTensor{2, 3, Float64}}}()
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
        
        # Calculate von Mises stress for this cell
        cell_von_mises = sqrt(3/2 * (dev(avg_stress) ⊡ dev(avg_stress)))
        
        if cell_von_mises > max_von_mises
            max_von_mises = cell_von_mises
            max_vm_cell_id = cell_id
        end
        
        stress_field[cell_id] = cell_stresses
    end
    
    println("Stress calculation complete with variable material properties")
    println("Maximum von Mises stress: $max_von_mises at cell $max_vm_cell_id")
    
    return stress_field, max_von_mises, max_vm_cell_id
end

# ============================================================================
# SYSTEM SOLVER - Zachované původní funkce
# ============================================================================

"""
    solve_system(K, f, dh, cellvalues, λ, μ, constraints...)

Solves the system of linear equations with multiple constraint handlers
and calculates stresses for constant material properties.
"""
function solve_system(K, f, dh, cellvalues, λ, μ, constraints...)
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    println("Solving linear system...")
    
    u = K \ f
    deformation_energy = 0.5 * dot(u, K * u)
    stress_field, max_von_mises, max_stress_cell = calculate_stresses(u, dh, cellvalues, λ, μ)

    println("Analysis complete")
    println("Deformation energy: $deformation_energy J")
    println("Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")
    
    return u, deformation_energy, stress_field, max_von_mises, max_stress_cell
end

"""
    solve_system_simp(K, f, dh, cellvalues, material_model, density_data, constraints...)

Solves the system of linear equations with multiple constraint handlers
and calculates stresses, using variable material properties.
"""
function solve_system_simp(K, f, dh, cellvalues, material_model, density_data, constraints...)
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    println("Solving linear system...")
    
    u = K \ f
    deformation_energy = 0.5 * dot(u, K * u)
    stress_field, max_von_mises, max_stress_cell = calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)
    
    println("Analysis complete")
    println("Deformation energy: $deformation_energy J")
    println("Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")
    
    return u, deformation_energy, stress_field, max_von_mises, max_stress_cell
end

end # module
