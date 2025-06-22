using Test
using Ferrite
using EasySIMP
using EasySIMP.MeshImport
using EasySIMP.FiniteElementAnalysis
using EasySIMP.Optimization
using EasySIMP.PostProcessing
using EasySIMP.Utils

@testset "EasySIMP.jl Tests" begin

  RUN_BEAM_fixed = false
  RUN_BEAM_slide = false
  RUN_BEAM_acc   = false
  RUN_CHAPADLO   = true

  if RUN_BEAM_fixed
    @testset "Cantilever Beam SIMP (fixed)" begin
        print_info("Running Cantilever Beam SIMP (fixed")
        
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

  if RUN_BEAM_slide
    @testset "Cantilever Beam SIMP (slide)" begin
        print_info("Running Cantilever Beam SIMP Optimization with Sliding Support")
        
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
        
        # Boundary conditions for sliding support test:
        # 1. YZ plane (x=0) - sliding constraint (fixed only in X direction)
        sliding_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
        
        # 2. Support at (60, 0, 2) in Y direction - fixed in Y
        support_nodes = select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [0.0, 1.0, 0.0], 0.5)
        if isempty(support_nodes)
            # Find closest node to support point
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
            support_nodes = Set([closest_node])
        end
        
        # 3. Force at (0, 20, 2) in direction (0, -1, 0)
        force_nodes = select_nodes_by_circle(grid, [0.0, 20.0, 2.0], [1.0, 0.0, 0.0], 1.0)
        if isempty(force_nodes)
            # Find closest node to force point
            target_point = [0.0, 20.0, 2.0]
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
        
        print_info("Found $(length(sliding_nodes)) sliding nodes")
        print_info("Found $(length(support_nodes)) support nodes") 
        print_info("Found $(length(force_nodes)) force nodes")
        
        # Apply boundary conditions
        assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, fill(0.4, getncells(grid)))
        
        # Sliding constraint: YZ plane fixed only in X direction (DOF 1)
        ch_sliding = apply_sliding_boundary!(K, f, dh, sliding_nodes, [1])  # Fix only X direction
        
        # Support constraint: fixed in Y direction (DOF 2)
        ch_support = apply_sliding_boundary!(K, f, dh, support_nodes, [2])  # Fix only Y direction
        
        # Apply force in negative Y direction
        apply_force!(f, dh, collect(force_nodes), [0.0, -1.0, 0.0])
        
        # Optimization parameters
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
            damping = 0.5
        )
                
        # Run optimization with both constraint handlers
        results = simp_optimize(
            grid, dh, cellvalues,
            [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], 
            [ch_sliding, ch_support], 
            opt_params
        )
        
        print_success("Optimization completed!")
        print_data("Final compliance: $(results.compliance)")
        print_data("Iterations: $(results.iterations)")
        
        # Export results
        results_data = create_results_data(grid, dh, results)
        export_results_vtu(results_data, "cantilever_beam_sliding_test")
        
        print_success("Sliding support test completed successfully!")
    end
  end

  if RUN_BEAM_acc
    @testset "Cantilever Beam SIMP (Sliding + Acceleration)" begin
        print_info("Running Cantilever Beam SIMP with Sliding Support and Y-Acceleration")
        
        # Import mesh
        grid = import_mesh("../data/cantilever_beam.vtu")
        print_success("Mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
        
        # Material properties - SNÍŽENÁ HUSTOTA pro lepší projev zrychlení
        E0 = 2.4e3
        ν = 0.35
        # ρ = 1040.0  # Sníženo z 7850 na 1000 kg/m³ (plast/kompozit)
        ρ = 1.04e-6     # kg/mm³
        λ, μ = create_material_model(E0, ν)
        material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)
        
        # Setup FEM
        dh, cellvalues, K, f = setup_problem(grid)
        print_success("FEM setup complete: $(ndofs(dh)) DOFs")
        
        # Boundary conditions (stejné jako sliding test)
        sliding_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
        support_nodes = select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [0.0, 1.0, 0.0], 0.5)
        force_nodes = select_nodes_by_circle(grid, [0.0, 20.0, 2.0], [1.0, 0.0, 0.0], 1.0)
        
        # Handle empty node sets (stejné jako předchozí test)
        # ... find closest nodes if empty ...
        
        # Assemble initial stiffness
        assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, fill(0.4, getncells(grid)))
        
        # Apply boundary conditions
        ch_sliding = apply_sliding_boundary!(K, f, dh, sliding_nodes, [1])  # Fix X
        ch_support = apply_sliding_boundary!(K, f, dh, support_nodes, [2])   # Fix Y
        
        # Apply point force (stejné jako předchozí)
        apply_force!(f, dh, collect(force_nodes), [0.0, -1000., 0.0])
        
        # NOVÉ: Apply acceleration in Y direction
        acceleration_vector = [0.0, 6000., 0.0]  # 15 m/s² dolů (silnější než gravitace)
        acceleration_data = (acceleration_vector, ρ)
        apply_acceleration!(f, dh, cellvalues, acceleration_vector, ρ)
        
        print_info("Applied Y-acceleration: $(acceleration_vector[2]) m/s² with density $(ρ) kg/m³")
        
        # Optimization parameters
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
            damping = 0.5
        )
        
        # Run optimization
        results = simp_optimize(
            grid, dh, cellvalues,
            [(dh, collect(force_nodes), [0.0, -1000., 0.0])], 
            [ch_sliding, ch_support], 
            opt_params, acceleration_data
        )
        
        print_success("Optimization with acceleration completed!")
        print_data("Final compliance: $(results.compliance)")
        print_data("Iterations: $(results.iterations)")
        
        # Export results
        results_data = create_results_data(grid, dh, results)
        export_results_vtu(results_data, "cantilever_beam_acceleration_test")
        
        print_success("Acceleration test completed successfully!")
    end
  end
  
  if RUN_CHAPADLO
      @testset "Chapadlo SIMP Optimization" begin
          print_info("Running Chapadlo SIMP topology optimization")
          
          # Import mesh
          grid = import_mesh("../data/stul15.vtu")
          print_success("Chapadlo mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
                    
          # Material properties for Chapadlo
          # E0 = 2.4e9  # Young's modulus 2400 MPa = 2.4e9 Pa
          E0 = 2.4e3      # MPa = N/mm²
          ν = 0.35    # Poisson's ratio
          # ρ = 1040.0  # Density 1040 kg/m³
          ρ = 1.04e-6     # kg/mm³
          λ, μ = create_material_model(E0, ν)
          material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)
          
          print_data("Material properties: E = $(E0/1e9) GPa, ν = $(ν), ρ = $(ρ) kg/m³")
          
          # Setup FEM
          dh, cellvalues, K, f = setup_problem(grid)
          
          # Boundary conditions - Vetknutí (Fixed support)
          # Omezená rovina XZ, y = 75, kruh r = 16.11, střed = [0, 75, 115]
          fixed_nodes = select_nodes_by_circle(grid, [0.0, 75.0, 115.0], [0.0, -1.0, 0.0], 16.11, 1e-3)
          
          # Symetrie - celá rovina YZ, x = 0, nulový posuv ve směru x
          symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
          
          # Load points
          # 1. Nožičky: rovina XY, z = -90, síla 2.5N
          nozicky_nodes = select_nodes_by_plane(grid, [0.0, 0.0, -90.0], [0.0, 0.0, 1.0], 1.0)
          
          # 2. Kamera: omezená rovina XY, z = 5, kruh r = 21.5, střed = [0, 0, 5], síla 1N
          kamera_nodes = select_nodes_by_circle(grid, [0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 21.5, 1e-3)

          # Export boundary conditions for visualization
          print_info("Exporting boundary conditions for ParaView inspection...")
          
          # Export all boundary conditions (fixed, symmetry vs all forces)
          all_force_nodes = union(nozicky_nodes, kamera_nodes)
          all_constraint_nodes = union(fixed_nodes, symmetry_nodes)
          export_boundary_conditions(grid, dh, all_constraint_nodes, all_force_nodes, "chapadlo_boundary_conditions_all")
          
          # exit()  # Uncomment to only export boundary conditions without running optimization
          
          # Handle empty node sets by finding closest nodes
          if isempty(fixed_nodes)
              print_warning("No fixed nodes found, finding closest to [0, 75, 115]")
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
              print_warning("No symmetry nodes found on YZ plane")
              # Find nodes closest to x=0 plane
              symmetry_nodes = Set{Int}()
              for node_id = 1:getnnodes(grid)
                  node_coord = grid.nodes[node_id].x
                  if abs(node_coord[1]) < 2.0  # Within 2mm of x=0 plane
                      push!(symmetry_nodes, node_id)
                  end
              end
              print_info("Found $(length(symmetry_nodes)) nodes near x=0 plane for symmetry")
          end
          
          if isempty(nozicky_nodes)
              print_warning("No nožičky nodes found, finding nodes near z = -90")
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
              print_warning("No kamera nodes found, finding nodes near [0, 0, 5]")
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
          assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, fill(0.3, getncells(grid)))
          
          # Fixed boundary condition (vetknutí) - all DOFs fixed
          ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
          
          # Symmetry boundary condition - fix only X direction (DOF 1)
          ch_symmetry = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
          
          print_info("Applied boundary conditions:")
          print_data("  Fixed support: $(length(fixed_nodes)) nodes (all DOFs)")
          print_data("  Symmetry: $(length(symmetry_nodes)) nodes (X direction only)")
          
          # Apply forces
          # Nožičky: 2.5N dolů (předpokládám směr [0, 0, -1])
          apply_force!(f, dh, collect(nozicky_nodes), [0.0, 0.0, -2500.])   # 2.5N = 2500 mN
          print_info("Applied 2.5N downward force to nožičky ($(length(nozicky_nodes)) nodes)")
          
          # Kamera: 1N dolů (předpokládám směr [0, 0, -1]) 
          apply_force!(f, dh, collect(kamera_nodes), [0.0, 0.0, -1000.0])   # 1N = 1000 mN
          print_info("Applied 1N downward force to kamera ($(length(kamera_nodes)) nodes)")
          
          # Acceleration data: 6 m/s² ve směru (0, 1, 0)
          acceleration_vector = [0.0, 6000.0, 0.0]  # 6 m/s² = 6000 mm/s²
          acceleration_data = (acceleration_vector, ρ)
          print_info("Vertical acceleration: $(acceleration_vector[2]) m/s² with density $(ρ) kg/m³")
          
          # Optimization parameters for Chapadlo
          opt_params = OptimizationParameters(
              E0 = E0,
              Emin = 1e-6,
              ν = ν,
              p = 3.0,
              volume_fraction = 0.4,     # 30% objemový poměr
              max_iterations = 15,       # Více iterací pro komplexnější geometrii
              tolerance = 0.005,
              filter_radius = 2.0,       # Větší filtr pro stabilitu
              move_limit = 0.2,          # Zadaný limitní krok
              damping = 0.5              # Zadané tlumení
          )
          
          print_info("Optimization parameters:")
          print_data("  Volume fraction: $(opt_params.volume_fraction)")
          print_data("  Move limit: $(opt_params.move_limit)")  
          print_data("  Damping: $(opt_params.damping)")
          print_data("  Filter radius: $(opt_params.filter_radius)")
          
          # Run optimization with multiple forces and both boundary conditions
          forces_list = [
              (dh, collect(nozicky_nodes), [0.0, 0.0, -2500.]),
              (dh, collect(kamera_nodes), [0.0, 0.0, -1000.])
          ]
          
          results = simp_optimize(
              grid, dh, cellvalues,
              forces_list,
              [ch_fixed, ch_symmetry],  # Both boundary conditions
              opt_params,
              acceleration_data
          )
          
          print_success("Chapadlo optimization completed!")
          print_data("Final compliance: $(results.compliance)")
          print_data("Final volume fraction: $(results.volume / calculate_volume(grid))")
          print_data("Iterations: $(results.iterations)")
          print_data("Converged: $(results.converged)")
          
          # Export results
          results_data = create_results_data(grid, dh, results)
          export_results_vtu(results_data, "chapadlo_optimization_velke")
          
          print_success("Chapadlo test completed successfully!")
          print_info("Results exported to: chapadlo_optimization.vtu")
      end
  end
  
end
