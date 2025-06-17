"""
EasySIMP.jl - Main Interface

Simple SIMP topology optimization for Julia using Ferrite.jl
"""

using .MeshImport
using .FiniteElementAnalysis  
using .Optimization
using .PostProcessing
using .Utils

export simp_topology_optimization, SIMPProblem

"""
    SIMPProblem

Complete problem definition for SIMP topology optimization.
"""
mutable struct SIMPProblem
    # Input mesh
    mesh_file::String
    
    # Material properties
    E0::Float64                    # Young's modulus of solid material (Pa)
    ν::Float64                     # Poisson's ratio
    ρ::Float64                     # Material density (kg/m³) 
    
    # Optimization parameters
    volume_fraction::Float64       # Target volume fraction (0-1)
    penalization_power::Float64    # SIMP penalization power (default: 3.0)
    filter_radius::Float64         # Density filter radius (element units)
    max_iterations::Int           # Maximum optimization iterations
    tolerance::Float64            # Convergence tolerance
    
    # Loading and boundary conditions
    point_loads::Vector{Tuple}     # [(node_set, force_vector), ...]
    body_forces::Vector{Tuple}     # [(acceleration_vector, density), ...]  
    fixed_supports::Vector{String} # Node sets with fixed supports (U1=U2=U3=0)
    sliding_supports::Vector{Tuple} # [(node_set, [fixed_dofs]), ...] 
    
    # Output settings
    output_file::String           # Base name for output files
    export_history::Bool          # Export convergence history
    
    # Constructor with defaults
    function SIMPProblem(
        mesh_file::String;
        E0 = 200e9,           # Steel Young's modulus  
        ν = 0.3,
        ρ = 7850.0,           # Steel density
        volume_fraction = 0.5,
        penalization_power = 3.0,
        filter_radius = 1.5,
        max_iterations = 200,
        tolerance = 0.01,
        point_loads = Tuple[],
        body_forces = Tuple[],
        fixed_supports = String[],
        sliding_supports = Tuple[],
        output_file = "simp_results",
        export_history = true
    )
        new(mesh_file, E0, ν, ρ, volume_fraction, penalization_power, filter_radius,
            max_iterations, tolerance, point_loads, body_forces, fixed_supports,
            sliding_supports, output_file, export_history)
    end
end

"""
    simp_topology_optimization(problem::SIMPProblem)

Main function to run SIMP topology optimization.

# Example Usage
```julia
# Define problem  
problem = SIMPProblem(
    "cantilever.vtu",
    volume_fraction = 0.4,
    point_loads = [(\"load_nodes\", [0.0, 0.0, -1000.0])],
    fixed_supports = [\"fixed_nodes\"],
    output_file = "cantilever_optimized"
)

# Run optimization
results = simp_topology_optimization(problem)
```
"""
function simp_topology_optimization(problem::SIMPProblem)
    print_info("=" ^ 60)
    print_info("Starting EasySIMP Topology Optimization")
    print_info("=" ^ 60)
    
    # 1. Import mesh
    print_info("Step 1: Importing mesh...")
    try
        grid = import_mesh(problem.mesh_file)
        print_success("Mesh imported successfully")
        print_data("Elements: $(getncells(grid))")
        print_data("Nodes: $(getnnodes(grid))")
    catch e
        print_error("Failed to import mesh: $e")
        rethrow(e)
    end
    
    # 2. Setup finite element problem
    print_info("Step 2: Setting up FE problem...")
    try
        dh, cellvalues, K, f = setup_problem(grid)
        print_success("FE problem setup complete")
        print_data("DOFs: $(ndofs(dh))")
    catch e
        print_error("Failed to setup FE problem: $e")
        rethrow(e)
    end
    
    # 3. Create material model
    print_info("Step 3: Creating material model...")
    λ, μ = create_material_model(problem.E0, problem.ν)
    material_model = create_simp_material_model(problem.E0, problem.ν)
    print_success("SIMP material model created")
    print_data("Base Young's modulus: $(problem.E0/1e9) GPa")
    print_data("Poisson's ratio: $(problem.ν)")
    print_data("Penalization power: $(problem.penalization_power)")
    
    # 4. Apply boundary conditions and loads
    print_info("Step 4: Applying boundary conditions and loads...")
    constraint_handlers = []
    
    # Fixed supports  
    for support_set in problem.fixed_supports
        if haskey(grid.nodesets, support_set)
            nodes = grid.nodesets[support_set]
            ch = apply_fixed_boundary!(K, f, dh, nodes)
            push!(constraint_handlers, ch)
            print_data("Fixed support applied to node set: $support_set")
        else
            print_warning("Node set '$support_set' not found in mesh")
        end
    end
    
    # Sliding supports
    for (support_set, fixed_dofs) in problem.sliding_supports
        if haskey(grid.nodesets, support_set)
            nodes = grid.nodesets[support_set]
            ch = apply_sliding_boundary!(K, f, dh, nodes, fixed_dofs)
            push!(constraint_handlers, ch)
            print_data("Sliding support applied to node set: $support_set")
        else
            print_warning("Node set '$support_set' not found in mesh")
        end
    end
    
    # Point loads
    for (load_set, force_vector) in problem.point_loads
        if haskey(grid.nodesets, load_set)
            nodes = grid.nodesets[load_set]
            apply_force!(f, dh, nodes, force_vector)
            print_data("Point load applied to node set: $load_set")
        else
            print_warning("Node set '$load_set' not found in mesh")
        end
    end
    
    # Body forces
    for (acceleration, density) in problem.body_forces
        apply_acceleration!(f, dh, cellvalues, acceleration, density)
        print_data("Body force applied: $acceleration")
    end
    
    print_success("Boundary conditions and loads applied")
    
    # 5. Setup optimization parameters
    print_info("Step 5: Setting up optimization...")
    opt_params = OptimizationParameters(
        E0 = problem.E0,
        ν = problem.ν,
        p = problem.penalization_power,
        volume_fraction = problem.volume_fraction,
        max_iterations = problem.max_iterations,
        tolerance = problem.tolerance,
        filter_radius = problem.filter_radius
    )
    
    print_success("Optimization parameters configured")
    print_data("Target volume fraction: $(problem.volume_fraction)")
    print_data("Filter radius: $(problem.filter_radius)")
    print_data("Max iterations: $(problem.max_iterations)")
    
    # 6. Run optimization
    print_info("Step 6: Running topology optimization...")
    try
        results = simp_optimize(
            grid, dh, cellvalues, material_model,
            [(f,)], constraint_handlers, opt_params
        )
        print_success("Optimization completed successfully!")
    catch e
        print_error("Optimization failed: $e")
        rethrow(e)
    end
    
    # 7. Post-processing and export
    print_info("Step 7: Post-processing and exporting results...")
    try
        # Create results data
        results_data = create_results_data(grid, dh, results)
        
        # Export to VTU
        export_results_vtu(results_data, problem.output_file, 
                          include_history = problem.export_history)
        
        # Create summary report
        create_summary_report(results_data, problem.output_file)
        
        print_success("Results exported successfully!")
        print_data("Main results: $(problem.output_file)_results.vtu")
        print_data("Summary report: $(problem.output_file).txt")
        
    catch e
        print_error("Post-processing failed: $e")
        rethrow(e)
    end
    
    # 8. Summary
    print_info("=" ^ 60)
    print_success("EasySIMP Optimization Complete!")
    print_info("=" ^ 60)
    print_data("Final compliance: $(results.compliance)")
    print_data("Final volume fraction: $(results.volume / calculate_volume(grid))")
    print_data("Iterations: $(results.iterations)")
    print_data("Converged: $(results.converged ? "Yes" : "No")")
    
    return results
end

# Convenience functions for common problems

"""
    cantilever_beam(length, height, thickness, volume_fraction; kwargs...)

Create a cantilever beam optimization problem with typical boundary conditions.
"""
function cantilever_beam(
    mesh_file::String,
    volume_fraction::Float64 = 0.4;
    load_magnitude::Float64 = 1000.0,
    kwargs...
)
    return SIMPProblem(
        mesh_file,
        volume_fraction = volume_fraction,
        point_loads = [(\"load_nodes\", [0.0, 0.0, -load_magnitude])],
        fixed_supports = [\"fixed_nodes\"],
        kwargs...
    )
end

"""
    bridge_structure(mesh_file, volume_fraction; kwargs...)

Create a bridge optimization problem with supports at both ends.
"""
function bridge_structure(
    mesh_file::String,
    volume_fraction::Float64 = 0.5;
    load_magnitude::Float64 = 1000.0,
    kwargs...
)
    return SIMPProblem(
        mesh_file,
        volume_fraction = volume_fraction,
        point_loads = [(\"load_nodes\", [0.0, 0.0, -load_magnitude])],
        sliding_supports = [(\"support_left\", [2, 3]), (\"support_right\", [3])], # Pin and roller
        kwargs...
    )
end

"""
    mbb_beam(mesh_file, volume_fraction; kwargs...)

Create an MBB (Messerschmitt-Bölkow-Blohm) beam problem.
"""
function mbb_beam(
    mesh_file::String,
    volume_fraction::Float64 = 0.5;
    load_magnitude::Float64 = 1000.0,
    kwargs...
)
    return SIMPProblem(
        mesh_file,
        volume_fraction = volume_fraction,
        point_loads = [(\"load_nodes\", [0.0, -load_magnitude, 0.0])],
        sliding_supports = [(\"left_support\", [1]), (\"right_support\", [2])],
        kwargs...
    )
end
