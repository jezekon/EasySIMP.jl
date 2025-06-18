using Test
using Ferrite
using EasySIMP
using EasySIMP.MeshImport
using EasySIMP.FiniteElementAnalysis
using EasySIMP.Optimization
using EasySIMP.PostProcessing
using EasySIMP.Utils

@testset "EasySIMP.jl Tests" begin
    
    @testset "Cantilever Beam SIMP Optimization" begin
        print_info("Running Cantilever Beam SIMP Optimization")
        
        # Import mesh
        grid = import_mesh("../data/cantilever_beam.vtu")
        print_success("Mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
        
        # Material properties
        E0 = 200.
        ν = 0.3
        ρ = 7850.0
        λ, μ = create_material_model(E0, ν)
        material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)
        
        # Setup FEM
        dh, cellvalues, K, f = setup_problem(grid)
        print_success("FEM setup complete: $(ndofs(dh)) DOFs")
        
        # Boundary conditions
        fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
        force_nodes = select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [1.0, 0.0, 0.0], 1.0)
        
        if isempty(force_nodes)
            target_point = [60.0, 0.0, 2.0]
            min_dist = Inf
            closest_node = 1
            for node_id = 1:getnnodes(grid)
                node_coord = grid.nodes[node_id].x
                dist = norm(node_coord - target_point)
                if dist < min_dist
                    min_dist = dist
                    closest_node = node_id
                end
            end
            force_nodes = Set([closest_node])
        end
        
        # Apply boundary conditions
        assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, fill(0.4, getncells(grid)))
        ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
        apply_force!(f, dh, collect(force_nodes), [0.0, -1.0, 0.0])
        
        # Optimization parameters
        opt_params = OptimizationParameters(
            E0 = E0,
            Emin = 1e-6,
            ν = ν,
            p = 3.0,
            volume_fraction = 0.4,
            max_iterations = 20,        # ← Zvýšit pro lepší konvergenci
            tolerance = 0.005,          # ← Menší tolerance
            filter_radius = 2.5,
            move_limit = 0.1,          # ← Menší move limit pro stabilitu
            damping = 0.5
        )
                
        # Run optimization
        results = simp_optimize(
            grid, dh, cellvalues,
            [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], [ch_fixed], opt_params
        )
        
        print_success("Optimization completed!")
        print_data("Final compliance: $(results.compliance)")
        print_data("Iterations: $(results.iterations)")
        
        # Export results
        results_data = create_results_data(grid, dh, results)
        export_results_vtu(results_data, "cantilever_beam_simp_test")
        
        print_success("Test completed successfully!")
    end
end
