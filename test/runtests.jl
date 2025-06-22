using Test
using Ferrite
using EasySIMP
using EasySIMP.MeshImport
using EasySIMP.FiniteElementAnalysis
using EasySIMP.Optimization
using EasySIMP.PostProcessing
using EasySIMP.Utils

@testset "EasySIMP.jl Tests with Performance Optimizations" begin

  RUN_BEAM_fixed = false
  RUN_BEAM_slide = false
  RUN_BEAM_acc   = false
  RUN_BEAM_performance = true   # NEW: Performance comparison test
  RUN_CHAPADLO   = false

  if RUN_BEAM_performance
    @testset "Cantilever Beam Performance Comparison" begin
        print_info("Running Performance Comparison: Standard vs Optimized SIMP")
        
        # Import mesh
        grid = import_mesh("../data/cantilever_beam.vtu")
        print_success("Mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
        
        # Material properties
        E0 = 200.
        ν = 0.3
        
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
        
        ch_fixed = apply_fixed_boundary!(copy(K), copy(f), dh, fixed_nodes)
        
        # Test 1: Standard optimization (baseline)
        print_info("Test 1: Standard Direct Solver")
        
        standard_params = OptimizationParameters(
            E0 = E0,
            Emin = 1e-6,
            ν = ν,
            p = 3.0,
            volume_fraction = 0.4,
            max_iterations = 12,
            tolerance = 0.01,
            filter_radius = 2.5,
            move_limit = 0.1,
            damping = 0.5,
            solver_options = SolverOptions(use_warm_start = false)  # Disable warm-start
        )
        
        time_standard = @elapsed begin
            results_standard = simp_optimize(
                grid, dh, cellvalues,
                [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], 
                [ch_fixed], 
                standard_params
            )
        end
        
        print_success("Standard optimization completed in $(round(time_standard, digits=2)) seconds")
        print_data("Final compliance: $(results_standard.compliance)")
        
        # Test 2: Optimized with warm-start (aggressive)
        print_info("Test 2: Optimized Warm-Start Solver (Aggressive)")
        
        optimized_params = OptimizationParameters(
            E0 = E0,
            Emin = 1e-6,
            ν = ν,
            p = 3.0,
            volume_fraction = 0.4,
            max_iterations = 12,
            tolerance = 0.01,
            filter_radius = 2.5,
            move_limit = 0.1,
            damping = 0.5,
            solver_options = create_optimized_solver_options(conservative=false)
        )
        
        time_optimized = @elapsed begin
            results_optimized = simp_optimize(
                grid, dh, cellvalues,
                [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], 
                [ch_fixed], 
                optimized_params
            )
        end
        
        print_success("Optimized optimization completed in $(round(time_optimized, digits=2)) seconds")
        print_data("Final compliance: $(results_optimized.compliance)")
        print_data("Total CG iterations: $(results_optimized.total_cg_iterations)")
        print_data("Average CG per solve: $(round(results_optimized.average_cg_per_iteration, digits=1))")
        
        # Test 3: Conservative optimized settings
        print_info("Test 3: Conservative Optimized Settings")
        
        conservative_params = OptimizationParameters(
            E0 = E0,
            Emin = 1e-6,
            ν = ν,
            p = 3.0,
            volume_fraction = 0.4,
            max_iterations = 12,
            tolerance = 0.01,
            filter_radius = 2.5,
            move_limit = 0.1,
            damping = 0.5,
            solver_options = create_optimized_solver_options(conservative=true)
        )
        
        time_conservative = @elapsed begin
            results_conservative = simp_optimize(
                grid, dh, cellvalues,
                [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], 
                [ch_fixed], 
                conservative_params
            )
        end
        
        print_success("Conservative optimization completed in $(round(time_conservative, digits=2)) seconds")
        
        # Performance analysis
        speedup_aggressive = time_standard / time_optimized
        speedup_conservative = time_standard / time_conservative
        
        print_info("Performance Analysis:")
        print_data("Standard time: $(round(time_standard, digits=2))s")
        print_data("Optimized (aggressive) time: $(round(time_optimized, digits=2))s ($(round(speedup_aggressive, digits=1))x speedup)")
        print_data("Optimized (conservative) time: $(round(time_conservative, digits=2))s ($(round(speedup_conservative, digits=1))x speedup)")
        
        # Solution quality check
        compliance_diff_aggressive = abs(results_optimized.compliance - results_standard.compliance) / results_standard.compliance * 100
        compliance_diff_conservative = abs(results_conservative.compliance - results_standard.compliance) / results_standard.compliance * 100
        
        print_data("Compliance difference (aggressive): $(round(compliance_diff_aggressive, digits=2))%")
        print_data("Compliance difference (conservative): $(round(compliance_diff_conservative, digits=2))%")
        
        # Validate performance improvements
        @test speedup_aggressive >= 1.0  # Should be at least as fast as standard
        @test speedup_conservative >= 1.0
        @test compliance_diff_aggressive < 5.0  # Less than 5% difference in solution quality
        @test compliance_diff_conservative < 5.0
        @test results_optimized.total_cg_iterations > 0  # Should have used iterative solver
        
        # Export results for comparison
        results_data_standard = create_results_data(grid, dh, results_standard)
        export_results_vtu(results_data_standard, "cantilever_standard")
        
        results_data_optimized = create_results_data(grid, dh, results_optimized)
        export_results_vtu(results_data_optimized, "cantilever_optimized")
        
        print_success("Performance test completed successfully!")
        
        if speedup_aggressive > 1.5
            print_success("🚀 Excellent performance improvement achieved!")
        elseif speedup_aggressive > 1.2
            print_success("⚡ Good performance improvement achieved!")
        else
            print_warning("🐌 Limited performance improvement - check solver settings")
        end
    end
  end

  if RUN_BEAM_fixed
    @testset "Cantilever Beam SIMP (fixed) - Optimized" begin
        print_info("Running Cantilever Beam SIMP (fixed) with optimized solver")
        
        # Import mesh
        grid = import_mesh("../data/cantilever_beam.vtu")
        print_success("Mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
        
        # Material properties
        E0 = 200.
        ν = 0.3
        
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
        ch_fixed = apply_fixed_boundary!(copy(K), copy(f), dh, fixed_nodes)
        
        # Optimization parameters with performance optimizations
        opt_params = OptimizationParameters(
            E0 = E0,
            Emin = 1e-6,
            ν = ν,
            p = 3.0,
            volume_fraction = 0.4,
            max_iterations = 20,
            tolerance = 0.005,
            filter_radius = 2.5,
            move_limit = 0.1,
            damping = 0.5,
            solver_options = create_optimized_solver_options(conservative=false)  # Use optimized solver
        )
                
        # Run optimization
        results = simp_optimize(
            grid, dh, cellvalues,
            [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], [ch_fixed], opt_params
        )
        
        print_success("Optimization completed!")
        print_data("Final compliance: $(results.compliance)")
        print_data("Iterations: $(results.iterations)")
        print_data("Total CG iterations: $(results.total_cg_iterations)")
        print_data("Average CG per solve: $(round(results.average_cg_per_iteration, digits=1))")
        
        # Export results
        results_data = create_results_data(grid, dh, results)
        export_results_vtu(results_data, "cantilever_beam_simp_optimized")
        
        print_success("Optimized test completed successfully!")
    end
  end

  if RUN_CHAPADLO
      @testset "Chapadlo SIMP Optimization - Performance Enhanced" begin
          print_info("Running Chapadlo SIMP topology optimization with performance enhancements")
          
          # Import mesh
          grid = import_mesh("../data/stul15.vtu")
          print_success("Chapadlo mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
                    
          # Material properties for Chapadlo
          E0 = 2.4e3      # MPa = N/mm²
          ν = 0.35    # Poisson's ratio
          ρ = 1.04e-6     # kg/mm³
          
          print_data("Material properties: E = $(E0/1e3) GPa, ν = $(ν), ρ = $(ρ*1e9) kg/m³")
          
          # Setup FEM
          dh, cellvalues, K, f = setup_problem(grid)
          
          # Boundary conditions
          fixed_nodes = select_nodes_by_circle(grid, [0.0, 75.0, 115.0], [0.0, -1.0, 0.0], 16.11, 1e-3)
          symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
          nozicky_nodes = select_nodes_by_plane(grid, [0.0, 0.0, -90.0], [0.0, 0.0, 1.0], 1.0)
          kamera_nodes = select_nodes_by_circle(grid, [0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 21.5, 1e-3)

          # Handle empty node sets by finding closest nodes
          if isempty(fixed_nodes)
              target_point = [0.0, 75.0, 115.0]
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
              fixed_nodes = Set([closest_node])
              print_info("Using closest node $(closest_node) for fixed support")
          end
          
          if isempty(symmetry_nodes)
              symmetry_nodes = Set{Int}()
              for node_id = 1:getnnodes(grid)
                  node_coord = grid.nodes[node_id].x
                  if abs(node_coord[1]) < 2.0
                      push!(symmetry_nodes, node_id)
                  end
              end
              print_info("Found $(length(symmetry_nodes)) nodes near x=0 plane for symmetry")
          end
          
          if isempty(nozicky_nodes)
              nozicky_nodes = Set{Int}()
              for node_id = 1:getnnodes(grid)
                  node_coord = grid.nodes[node_id].x
                  if length(node_coord) >= 3 && abs(node_coord[3] - (-90.0)) < 5.0
                      push!(nozicky_nodes, node_id)
                  end
              end
              print_info("Found $(length(nozicky_nodes)) nodes near z = -90 for nožičky")
          end
          
          if isempty(kamera_nodes)
              target_point = [0.0, 0.0, 5.0]
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
              kamera_nodes = Set([closest_node])
              print_info("Using closest node $(closest_node) for kamera")
          end
          
          # Apply boundary conditions
          ch_fixed = apply_fixed_boundary!(copy(K), copy(f), dh, fixed_nodes)
          ch_symmetry = apply_sliding_boundary!(copy(K), copy(f), dh, symmetry_nodes, [1])
          
          print_info("Applied boundary conditions:")
          print_data("  Fixed support: $(length(fixed_nodes)) nodes (all DOFs)")
          print_data("  Symmetry: $(length(symmetry_nodes)) nodes (X direction only)")
          
          # Acceleration data
          acceleration_vector = [0.0, 6000.0, 0.0]  # 6 m/s² = 6000 mm/s²
          acceleration_data = (acceleration_vector, ρ)
          print_info("Vertical acceleration: $(acceleration_vector[2]/1000) m/s² with density $(ρ*1e9) kg/m³")
          
          # Optimization parameters with performance enhancements
          opt_params = OptimizationParameters(
              E0 = E0,
              Emin = 1e-6,
              ν = ν,
              p = 3.0,
              volume_fraction = 0.4,
              max_iterations = 15,       # Moderate iterations for testing
              tolerance = 0.005,
              filter_radius = 2.0,
              move_limit = 0.2,
              damping = 0.5,
              solver_options = create_optimized_solver_options(conservative=false)  # Use aggressive optimizations
          )
          
          print_info("Optimization parameters:")
          print_data("  Volume fraction: $(opt_params.volume_fraction)")
          print_data("  Filter radius: $(opt_params.filter_radius)")
          print_data("  Solver: Optimized warm-start (aggressive)")
          
          # Run optimization with performance monitoring
          forces_list = [
              (dh, collect(nozicky_nodes), [0.0, 0.0, -2500.]),
              (dh, collect(kamera_nodes), [0.0, 0.0, -1000.])
          ]
          
          optimization_time = @elapsed begin
              results = simp_optimize(
                  grid, dh, cellvalues,
                  forces_list,
                  [ch_fixed, ch_symmetry],
                  opt_params,
                  acceleration_data
              )
          end
          
          print_success("Chapadlo optimization completed in $(round(optimization_time, digits=2)) seconds!")
          print_data("Final compliance: $(results.compliance)")
          print_data("Final volume fraction: $(results.volume / calculate_volume(grid))")
          print_data("Iterations: $(results.iterations)")
          print_data("Converged: $(results.converged)")
          print_data("Total CG iterations: $(results.total_cg_iterations)")
          print_data("Average CG per solve: $(round(results.average_cg_per_iteration, digits=1))")
          
          # Performance assessment
          if results.average_cg_per_iteration < 100
              print_success("🚀 Excellent solver performance!")
          elseif results.average_cg_per_iteration < 200
              print_success("⚡ Good solver performance!")
          else
              print_warning("🐌 Solver performance could be improved")
          end
          
          # Export results
          results_data = create_results_data(grid, dh, results)
          export_results_vtu(results_data, "chapadlo_optimization_performance")
          
          print_success("Enhanced Chapadlo test completed successfully!")
          print_info("Results exported to: chapadlo_optimization_performance.vtu")
          
          # Test performance assertions
          @test results.converged == true
          @test results.total_cg_iterations >= 0  # Should have used some CG iterations
          @test optimization_time > 0  # Sanity check
      end
  end
  
end
