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
    RUN_BEAM_acc = false
    RUN_GRIPPER = true

    if RUN_BEAM_fixed
        @testset "Cantilever Beam SIMP (fixed)" begin
            print_info("Running Cantilever Beam SIMP (fixed")

            # Import mesh
            # grid = import_mesh("../data/cantilever_beam.vtu")
            # print_success(
            #     "Mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes",
            # )

            grid = generate_grid(
                Hexahedron,
                (60, 20, 4),
                Vec((0.0, 0.0, 0.0)),
                Vec((60.0, 20.0, 4.0)),
            )
            print_success("Generated mesh: $(getncells(grid)) elements")

            # Material properties
            E0 = 200.0
            ν = 0.3
            ρ = 7850.0
            λ, μ = create_material_model(E0, ν)
            material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

            # Setup FEM
            dh, cellvalues, K, f = setup_problem(grid)
            print_success("FEM setup complete: $(ndofs(dh)) DOFs")

            # Boundary conditions
            fixed_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
            force_nodes =
                select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [1.0, 0.0, 0.0], 1.0)

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
            assemble_stiffness_matrix_simp!(
                K,
                f,
                dh,
                cellvalues,
                material_model,
                fill(0.4, getncells(grid)),
            )
            ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            apply_force!(f, dh, collect(force_nodes), [0.0, -1.0, 0.0])

            # Optimization parameters
            opt_params = OptimizationParameters(
                E0 = E0,
                Emin = 1e-6,
                ν = ν,
                p = 3.0,
                volume_fraction = 0.4,
                max_iterations = 2000,
                tolerance = 0.080,
                filter_radius = 2.5,
                move_limit = 0.2,
                damping = 0.5,
            )

            # Run optimization
            results = simp_optimize(
                grid,
                dh,
                cellvalues,
                [(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
                [ch_fixed],
                opt_params,
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
            print_success(
                "Mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes",
            )

            # Material properties
            E0 = 200.0
            ν = 0.3
            ρ = 7850.0
            λ, μ = create_material_model(E0, ν)
            material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

            # Setup FEM
            dh, cellvalues, K, f = setup_problem(grid)
            print_success("FEM setup complete: $(ndofs(dh)) DOFs")

            # Boundary conditions for sliding support test:
            # 1. YZ plane (x=0) - sliding constraint (fixed only in X direction)
            sliding_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)

            # 2. Support at (60, 0, 2) in Y direction - fixed in Y
            support_nodes =
                select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [0.0, 1.0, 0.0], 0.5)
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
            force_nodes =
                select_nodes_by_circle(grid, [0.0, 20.0, 2.0], [1.0, 0.0, 0.0], 1.0)
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
            assemble_stiffness_matrix_simp!(
                K,
                f,
                dh,
                cellvalues,
                material_model,
                fill(0.4, getncells(grid)),
            )

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
                max_iterations = 400,
                tolerance = 0.01,
                filter_radius = 2.0,
                move_limit = 0.2,
                damping = 0.5,
            )

            # Run optimization with both constraint handlers
            results = @time simp_optimize(
                grid,
                dh,
                cellvalues,
                [(dh, collect(force_nodes), [0.0, -1.0, 0.0])],
                [ch_sliding, ch_support],
                opt_params,
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
            print_info(
                "Running Cantilever Beam SIMP with Sliding Support and Y-Acceleration",
            )

            # Import mesh
            grid = import_mesh("../data/cantilever_beam.vtu")
            print_success(
                "Mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes",
            )

            # Material properties
            E0 = 2.4e3
            ν = 0.35
            ρ = 1.04e-6     # kg/mm³
            λ, μ = create_material_model(E0, ν)
            material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

            # Setup FEM
            dh, cellvalues, K, f = setup_problem(grid)
            print_success("FEM setup complete: $(ndofs(dh)) DOFs")

            # Boundary conditions
            sliding_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
            support_nodes =
                select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [0.0, 1.0, 0.0], 0.5)
            force_nodes =
                select_nodes_by_circle(grid, [0.0, 20.0, 2.0], [1.0, 0.0, 0.0], 1.0)

            # Assemble initial stiffness
            assemble_stiffness_matrix_simp!(
                K,
                f,
                dh,
                cellvalues,
                material_model,
                fill(0.4, getncells(grid)),
            )

            # Apply boundary conditions
            ch_sliding = apply_sliding_boundary!(K, f, dh, sliding_nodes, [1])
            ch_support = apply_sliding_boundary!(K, f, dh, support_nodes, [2])

            # Apply point force
            apply_force!(f, dh, collect(force_nodes), [0.0, -1000.0, 0.0])

            # Apply acceleration in Y direction
            acceleration_vector = [0.0, 6000.0, 0.0]
            acceleration_data = (acceleration_vector, ρ)

            print_info(
                "Applied Y-acceleration: $(acceleration_vector[2]) m/s² with density $(ρ) kg/m³",
            )

            # Optimization parameters
            opt_params = OptimizationParameters(
                E0 = E0,
                Emin = 1e-6,
                ν = ν,
                p = 3.0,
                volume_fraction = 0.4,
                max_iterations = 400,
                tolerance = 0.01,
                filter_radius = 2.0,
                move_limit = 0.2,
                damping = 0.5,
                use_cache = true,
            )

            # Run optimization
            results = @time simp_optimize(
                grid,
                dh,
                cellvalues,
                [(dh, collect(force_nodes), [0.0, -1000.0, 0.0])],
                [ch_sliding, ch_support],
                opt_params,
                acceleration_data,
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

    if RUN_GRIPPER
        @testset "Gripper SIMP Optimization" begin
            print_info("Running Gripper SIMP topology optimization")

            # Import mesh
            grid = import_mesh("../data/stul14.vtu")
            print_success(
                "Gripper mesh imported: $(getncells(grid)) elements, $(getnnodes(grid)) nodes",
            )

            # Material properties for Gripper
            E0 = 2.4e3      # MPa = N/mm²
            ν = 0.35        # Poisson's ratio
            ρ = 1.04e-6     # kg/mm³
            λ, μ = create_material_model(E0, ν)
            material_model = create_simp_material_model(E0, ν, 1e-6, 3.0)

            print_data("Material properties: E = $(E0/1e9) GPa, ν = $(ν), ρ = $(ρ) kg/m³")

            # Setup FEM
            dh, cellvalues, K, f = setup_problem(grid)

            # Boundary conditions - Fixed support at circular region
            fixed_nodes = select_nodes_by_circle(
                grid,
                [0.0, 75.0, 115.0],
                [0.0, -1.0, 0.0],
                16.11,
                1e-3,
            )

            # Symmetry plane YZ at x = 0 (fix X displacement only)
            symmetry_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)

            # Load points
            # 1. Legs: plane XY at z = -90
            nozicky_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, -90.0], [0.0, 0.0, 1.0], 1.0)

            # 2. Camera: circular region on plane XY at z = 5, radius = 21.5
            kamera_nodes =
                select_nodes_by_circle(grid, [0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 21.5, 1e-3)

            # Export boundary conditions for visualization
            print_info("Exporting boundary conditions for ParaView inspection...")
            all_force_nodes = union(nozicky_nodes, kamera_nodes)
            all_constraint_nodes = union(fixed_nodes, symmetry_nodes)
            export_boundary_conditions(
                grid,
                dh,
                all_constraint_nodes,
                all_force_nodes,
                "gripper_boundary_conditions_all",
            )

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
                symmetry_nodes = Set{Int}()
                for node_id = 1:getnnodes(grid)
                    node_coord = grid.nodes[node_id].x
                    if abs(node_coord[1]) < 2.0  # Within 2mm of x=0 plane
                        push!(symmetry_nodes, node_id)
                    end
                end
                print_info(
                    "Found $(length(symmetry_nodes)) nodes near x=0 plane for symmetry",
                )
            end

            if isempty(nozicky_nodes)
                print_warning("No leg nodes found, finding nodes near z = -90")
                nozicky_nodes = Set{Int}()
                for node_id = 1:getnnodes(grid)
                    node_coord = grid.nodes[node_id].x
                    if length(node_coord) >= 3 && abs(node_coord[3] - (-90.0)) < 5.0
                        push!(nozicky_nodes, node_id)
                    end
                end
                print_info("Found $(length(nozicky_nodes)) nodes near z = -90 for legs")
            end

            if isempty(kamera_nodes)
                print_warning("No camera nodes found, finding nodes near [0, 0, 5]")
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
                print_info("Using closest node $(closest_node) for camera")
            end

            # Apply boundary conditions
            assemble_stiffness_matrix_simp!(
                K,
                f,
                dh,
                cellvalues,
                material_model,
                fill(0.3, getncells(grid)),
            )

            # Fixed support (all DOFs constrained)
            ch_fixed = apply_fixed_boundary!(K, f, dh, fixed_nodes)

            # Symmetry constraint (X direction only)
            ch_symmetry = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])

            print_info("Applied boundary conditions:")
            print_data("  Fixed support: $(length(fixed_nodes)) nodes (all DOFs)")
            print_data("  Symmetry: $(length(symmetry_nodes)) nodes (X direction only)")

            # Apply forces
            # Legs: F = π*(14²-7.5²)*3*0.00985 ≈ 13 N downward
            apply_force!(f, dh, collect(nozicky_nodes), [0.0, 0.0, -13000.0]) # mN
            print_info(
                "Applied 13N downward force to legs ($(length(nozicky_nodes)) nodes)",
            )

            # Camera: F = π*(21.5²-17²)*0.001852*0.5 ≈ 0.5 N downward
            apply_force!(f, dh, collect(kamera_nodes), [0.0, 0.0, -500.0]) # mN
            print_info(
                "Applied 0.5N downward force to camera ($(length(kamera_nodes)) nodes)",
            )

            # Acceleration: 6 m/s² in Y direction
            acceleration_vector = [0.0, 6000.0, 0.0]  # 6 m/s² = 6000 mm/s²
            acceleration_data = (acceleration_vector, ρ)
            print_info(
                "Vertical acceleration: $(acceleration_vector[2]) mm/s² with density $(ρ) kg/mm³",
            )

            # Optimization parameters
            opt_params = OptimizationParameters(
                E0 = E0,
                Emin = 1e-6,
                ν = ν,
                p = 3.0,
                volume_fraction = 0.3,
                max_iterations = 3000,
                tolerance = 0.10,
                filter_radius = 1.5,
                move_limit = 0.2,
                damping = 0.5,
            )

            print_info("Optimization parameters:")
            print_data("  Volume fraction: $(opt_params.volume_fraction)")
            print_data("  Move limit: $(opt_params.move_limit)")
            print_data("  Damping: $(opt_params.damping)")
            print_data("  Filter radius: $(opt_params.filter_radius)")

            # Run optimization with multiple forces
            forces_list = [
                (dh, collect(nozicky_nodes), [0.0, 0.0, -13000.0]), # mN
                (dh, collect(kamera_nodes), [0.0, 0.0, -500.0]), # mN
            ]

            results = simp_optimize(
                grid,
                dh,
                cellvalues,
                forces_list,
                [ch_fixed, ch_symmetry],
                opt_params,
                acceleration_data,
            )

            print_success("Gripper optimization completed!")
            print_data("Final compliance: $(results.compliance)")
            print_data("Final volume fraction: $(results.volume / calculate_volume(grid))")
            print_data("Iterations: $(results.iterations)")
            print_data("Converged: $(results.converged)")

            # Export results
            results_data = create_results_data(grid, dh, results)
            export_results_vtu(results_data, "gripper_TO_vfrac-$(0.3)")

            print_success("Gripper test completed successfully!")
            print_info("Results exported to: gripper_optimization.vtu")
        end
    end
end
