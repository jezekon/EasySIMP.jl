using Test
using LinearAlgebra
using EasySIMP
using EasySIMP.MeshImport
using EasySIMP.FiniteElementAnalysis
using EasySIMP.Optimization
using EasySIMP.PostProcessing
using EasySIMP.Utils

@testset "EasySIMP.jl Tests" begin
    
    @testset "Cantilever Beam SIMP Optimization" begin
        println("="^60)
        println("Running Cantilever Beam SIMP Optimization Test")
        println("="^60)
        
        # Test configuration
        taskName = "cantilever_beam_simp_test"
        
        # 1. Import mesh from VTU file
        print_info("Step 1: Importing mesh from ../data/cantilever_beam.vtu")
        grid = import_mesh("../data/cantilever_beam.vtu")
        
        @test grid !== nothing
        print_success("Mesh imported successfully")
        print_data("Elements: $(getncells(grid))")
        print_data("Nodes: $(getnnodes(grid))")
        
        # 2. Setup material properties for SIMP optimization
        print_info("Step 2: Setting up SIMP material model")
        E0 = 200e9      # Steel Young's modulus [Pa]
        ν = 0.3         # Poisson's ratio
        ρ = 7850.0      # Steel density [kg/m³]
        Emin = 1e-6     # Minimum Young's modulus (to avoid singularity)
        p = 3.0         # SIMP penalization power
        
        # Create material models
        λ, μ = create_material_model(E0, ν)
        material_model = create_simp_material_model(E0, ν, Emin, p)
        
        print_success("Material model created")
        print_data("Young's modulus: $(E0/1e9) GPa")
        print_data("Poisson's ratio: $(ν)")
        print_data("Penalization power: $(p)")
        
        # 3. Setup FEM problem
        print_info("Step 3: Setting up finite element problem")
        dh, cellvalues, K, f = setup_problem(grid)
        
        @test dh !== nothing
        @test cellvalues !== nothing
        print_success("FEM problem setup complete")
        print_data("DOFs: $(ndofs(dh))")
        
        # 4. Define boundary conditions
        print_info("Step 4: Defining boundary conditions")
        
        # Fixed support: all nodes on plane x = 0 (YZ plane)
        fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
        @test !isempty(fixed_nodes)
        print_data("Fixed nodes (x=0): $(length(fixed_nodes)) nodes")
        
        # Load application: point load at (60, 0, 2) in direction (0, -1, 0)
        # Use a small radius to select nodes near the point
        force_nodes = select_nodes_by_circle(
            grid,
            [60.0, 0.0, 2.0],    # center point
            [1.0, 0.0, 0.0],     # normal to search plane (perpendicular to YZ plane)
            1.0                   # small radius to get nodes near the point
        )
        
        # If no nodes found with circle, try selecting by proximity to the point
        if isempty(force_nodes)
            # Alternative: find closest nodes to the target point
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
            print_warning("No nodes found at exact location, using closest node at distance $(min_dist)")
        end
        
        @test !isempty(force_nodes)
        print_data("Force application nodes: $(length(force_nodes)) nodes")
        
        # 5. Setup optimization parameters
        print_info("Step 5: Setting up optimization parameters")
        volume_fraction = 0.4  # Target volume fraction (40% of material)
        
        opt_params = OptimizationParameters(
            E0 = E0,
            Emin = Emin,
            ν = ν,
            p = p,
            volume_fraction = volume_fraction,
            max_iterations = 50,     # Reduce for faster testing
            tolerance = 0.01,
            filter_radius = 1.5,
            move_limit = 0.2,
            damping = 0.5
        )
        
        print_success("Optimization parameters configured")
        print_data("Target volume fraction: $(volume_fraction)")
        print_data("Max iterations: $(opt_params.max_iterations)")
        print_data("Filter radius: $(opt_params.filter_radius)")
        
        # 6. Initialize density field
        print_info("Step 6: Initializing design variables")
        n_cells = getncells(grid)
        initial_densities = fill(volume_fraction, n_cells)
        
        # 7. Assemble initial stiffness matrix
        print_info("Step 7: Assembling initial stiffness matrix")
        assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, initial_densities)
        
        # 8. Apply boundary conditions and loads
        print_info("Step 8: Applying boundary conditions and loads")
        
        # Apply fixed boundary conditions (all DOFs fixed)
        ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
        
        # Apply point load: F = 1 N in direction (0, -1, 0)
        force_vector = [0.0, -1.0, 0.0]  # 1 N downward
        apply_force!(f, dh, collect(force_nodes), force_vector)
        
        print_success("Boundary conditions and loads applied")
        print_data("Force applied: $(force_vector) N to $(length(force_nodes)) nodes")
        
        # 9. Run SIMP topology optimization
        print_info("Step 9: Running SIMP topology optimization")
        
        try
            # Prepare constraint handlers
            constraint_handlers = [ch_fixed]
            
            # Run optimization
            results = simp_optimize(
                grid, 
                dh, 
                cellvalues, 
                material_model,
                [(f,)],  # Force vector wrapped in tuple
                constraint_handlers, 
                opt_params
            )
            
            @test results !== nothing
            @test hasfield(typeof(results), :densities)
            @test hasfield(typeof(results), :displacements)
            @test hasfield(typeof(results), :compliance)
            @test hasfield(typeof(results), :converged)
            
            print_success("SIMP optimization completed successfully!")
            print_data("Final compliance: $(results.compliance)")
            print_data("Final volume fraction: $(results.volume / calculate_volume(grid))")
            print_data("Iterations: $(results.iterations)")
            print_data("Converged: $(results.converged ? "Yes" : "No")")
            
            # 10. Post-processing and export
            print_info("Step 10: Post-processing and exporting results")
            
            # Create results data structure
            results_data = create_results_data(grid, dh, results)
            
            # Export results to VTU files
            output_file = taskName
            export_results_vtu(results_data, output_file, include_history=true)
            
            # Create summary report
            create_summary_report(results_data, output_file)
            
            print_success("Results exported successfully!")
            print_data("Main results: $(output_file)_results.vtu")
            print_data("Summary report: $(output_file).txt")
            
            # 11. Basic validation tests
            print_info("Step 11: Validating results")
            
            # Test that densities are within bounds
            @test all(0.0 .<= results.densities .<= 1.0)
            
            # Test that volume constraint is approximately satisfied
            final_volume_fraction = results.volume / calculate_volume(grid)
            volume_error = abs(final_volume_fraction - volume_fraction)
            @test volume_error < 0.05  # Allow 5% tolerance
            
            # Test that some optimization occurred (densities not all equal to initial)
            @test !all(results.densities .≈ volume_fraction)
            
            # Test that compliance is positive
            @test results.compliance > 0.0
            
            # Test that displacement field is reasonable
            @test length(results.displacements) == ndofs(dh)
            @test any(results.displacements .!= 0.0)  # Some displacement should occur
            
            print_success("All validation tests passed!")
            
        catch e
            print_error("Optimization failed: $e")
            @test false  # Force test failure
            rethrow(e)
        end
        
        print_info("="^60)
        print_success("Cantilever Beam SIMP Optimization Test Completed Successfully!")
        print_info("="^60)
    end
    
    # @testset "High-Level Interface Test" begin
    #     print_info("Testing high-level SIMPProblem interface")
    #
    #     # Create problem using high-level interface
    #     problem = SIMPProblem(
    #         "../data/cantilever_beam.vtu",
    #         E0 = 200e9,
    #         ν = 0.3,
    #         ρ = 7850.0,
    #         volume_fraction = 0.4,
    #         penalization_power = 3.0,
    #         filter_radius = 1.5,
    #         max_iterations = 20,  # Reduced for testing
    #         tolerance = 0.02,
    #         point_loads = [("force_nodes", [0.0, -1.0, 0.0])],
    #         fixed_supports = ["fixed_nodes"],
    #         output_file = "cantilever_highlevel_test",
    #         export_history = true
    #     )
    #
    #     @test problem.mesh_file == "../data/cantilever_beam.vtu"
    #     @test problem.volume_fraction == 0.4
    #     @test problem.max_iterations == 20
    #
    #     # Note: The high-level interface would need node sets defined in the mesh
    #     # or custom node selection logic to work properly
    #     print_info("High-level interface structure validated")
    # end
end
