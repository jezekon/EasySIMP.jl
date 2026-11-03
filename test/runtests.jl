using Test
using Ferrite
using LinearAlgebra
using EasySIMP
using EasySIMP.MeshImport
using EasySIMP.FiniteElementAnalysis
using EasySIMP.Optimization
using EasySIMP.PostProcessing
using EasySIMP.Utils

# Unit tests of the finite element and sensitivity building blocks. The full
# optimizations (cantilever, gripper) are in test/Examples
# (01_basic_cantilever.jl, 04_gripper_complex.jl) and check their own results.

# Linear static solution: assemble K for the densities, add the loads, apply the
# boundary conditions, solve. Returns u and the compliance fᵀu.
function solve_static(dh, cellvalues, material_model, densities, add_loads!, ch)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities)
    add_loads!(f)
    apply!(K, f, ch)
    u = cholesky(Symmetric(K)) \ f
    return u, dot(f, u)
end

# Total force of a global force vector in direction d, summed through the node → dof map
total_force(f, node_to_dofs, d) = sum(f[dofs[d]] for dofs in values(node_to_dofs))

@testset "EasySIMP.jl Tests" begin
    # Cantilever mesh of 01_basic_cantilever.jl: 60 × 20 × 4 mm, 1 mm hexahedra
    grid = generate_grid(Hexahedron, (60, 20, 4), Vec((0.0, 0.0, 0.0)), Vec((60.0, 20.0, 4.0)))
    dh, cellvalues = setup_problem(grid)
    node_to_dofs = get_node_dofs(dh)
    material_model = create_simp_material_model(1.0, 0.3, 1e-9, 3.0)
    fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6)
    end_face_nodes = select_nodes_by_plane(grid, [60.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6)
    ch_fixed = redirect_stdout(() -> create_constraint_handler(dh, fixed_nodes), devnull)

    @testset "Surface traction: facets and resultant" begin
        # The local facet id goes to reinit! of the FacetValues, so it must follow
        # Ferrite's numbering: the quadrature points of every facet of the end face
        # x = L lie on that face, and a total force F gives the resultant F. A plane
        # inside the mesh (x = L/2) has no boundary facets.
        for (cell_type, n_facets) in ((Hexahedron, 4), (Tetrahedron, 4 * 2))
            g = generate_grid(cell_type, (6, 2, 2), Vec((0.0, 0.0, 0.0)), Vec((6.0, 2.0, 2.0)))
            d, _ = redirect_stdout(() -> setup_problem(g), devnull)
            face = select_nodes_by_plane(g, [6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6)
            inner = select_nodes_by_plane(g, [3.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6)

            facets = redirect_stdout(() -> get_boundary_facets(g, face), devnull)
            @test length(facets) == n_facets
            @test isempty(redirect_stdout(() -> get_boundary_facets(g, inner), devnull))

            x_quadrature = Float64[]
            traction_function = function (x, y, z)
                push!(x_quadrature, x)
                return [0.0, -1.0, 0.0]
            end
            apply_surface_traction!(zeros(ndofs(d)), d, g, facets, traction_function)
            @test all(x -> isapprox(x, 6.0; atol = 1e-12), x_quadrature)

            load = redirect_stdout(() -> SurfaceTractionLoad(d, g, face, [0.0, -1.0, 0.0]), devnull)
            f = zeros(ndofs(d))
            apply_load_condition!(f, load)
            @test isapprox(total_force(f, get_node_dofs(d), 2), -1.0; atol = 1e-12)
            @test isapprox(total_force(f, get_node_dofs(d), 1), 0.0; atol = 1e-12)
        end
    end

    @testset "Full block vs. Timoshenko beam" begin
        # Solid block (ρ = 1), E0 = 1 MPa, ν = 0.3, clamped at x = 0, 1 N in -y as a
        # uniform traction on the whole end face. Timoshenko beam: U = F w / 2 with
        # w = L³/(3EI) + L/(κGA), κ = 5/6, which gives 14.670 N·mm; the 3D block with
        # its clamped root is slightly stiffer: U = 14.4824 N·mm.
        load = redirect_stdout(devnull) do
            SurfaceTractionLoad(dh, grid, end_face_nodes, [0.0, -1.0, 0.0])
        end
        _, compliance = solve_static(
            dh,
            cellvalues,
            material_model,
            ones(getncells(grid)),
            f -> apply_load_condition!(f, load),
            ch_fixed,
        )
        U = compliance / 2

        b, h, L, E, ν = 4.0, 20.0, 60.0, 1.0, 0.3
        I = b * h^3 / 12
        G = E / (2 * (1 + ν))
        w = L^3 / (3 * E * I) + L / (5 / 6 * G * b * h)
        @test isapprox(U, 14.4824; atol = 1e-4)
        @test abs(U / (w / 2) - 1) < 0.02
    end

    @testset "VTU displacements through the node → dof map" begin
        # Ferrite numbers the dofs cell by cell, not as 3(n-1)+d by node id (3029 of
        # the 6405 nodes differ), so the export must take the displacement of each
        # node through get_node_dofs. Solid block, 1 N in -y on the edge x = 60,
        # y = 0: the tip (60, 0, 2) moves u_y = -30.3592 mm.
        edge_nodes = select_nodes(grid, x -> abs(x[1] - 60.0) < 1e-6 && abs(x[2]) < 1e-6)
        u, _ = solve_static(
            dh,
            cellvalues,
            material_model,
            ones(getncells(grid)),
            f -> apply_force!(f, dh, collect(edge_nodes), [0.0, -1.0, 0.0]),
            ch_fixed,
        )
        @test count(n -> node_to_dofs[n] != [3n - 2, 3n - 1, 3n], 1:getnnodes(grid)) > 0

        results_data = ResultsData(
            grid,
            dh,
            cellvalues,
            ones(getncells(grid)),
            u,
            zeros(getncells(grid)),
            Dict(),
            0.0,
            1.0,
            0,
            true,
            Float64[],
            Float64[],
        )
        nodal_displacements = EasySIMP.PostProcessing.extract_nodal_displacements(results_data)
        @test all(
            nodal_displacements[d, n] == u[node_to_dofs[n][d]] for
            n = 1:getnnodes(grid), d = 1:3
        )

        tip = only(select_nodes(grid, x -> norm(x - Vec((60.0, 0.0, 2.0))) < 1e-6))
        @test isapprox(nodal_displacements[2, tip], -30.3592; atol = 1e-4)
    end

    @testset "Body force on every cell" begin
        # The body force is ρ_e ρ_mat a on every cell with ρ_e > 0, so its total over
        # the block (4800 mm³) is ρ_e ρ_mat a V, with ρ_mat in kg/mm³ as in t/mm³. The
        # former cut-off ρ_e ρ_mat < 1e-6 dropped every cell below ρ_e = 0.96 in
        # kg/mm³ and all cells in t/mm³.
        acceleration = [0.0, 6000.0, 0.0]
        for ρ_mat in (1.04e-6, 1.04e-9), ρ_e in (1.0, 0.5)
            f = zeros(ndofs(dh))
            apply_variable_density_volume_force!(
                f,
                dh,
                cellvalues,
                acceleration,
                fill(ρ_e * ρ_mat, getncells(grid)),
            )
            expected = ρ_e * ρ_mat * 6000.0 * 4800.0
            @test isapprox(total_force(f, node_to_dofs, 2), expected; rtol = 1e-12)
        end
    end

    @testset "Sensitivities vs. finite differences" begin
        # Compliance C = fᵀu of a small clamped beam (6 × 2 × 2 elements, tip load on
        # the edge x = 6, y = 0) with smoothly varying densities 0.3–0.9. Analytic
        # sensitivities against central differences (step 1e-6):
        #   - without body force;
        #   - with a body force ρ_e ρ_mat a: the term 2 u_eᵀ g_e is needed, without it
        #     the error is 0.38 (12 of the 24 sensitivities are positive);
        #   - with the body force and rho_lin = 0.5 (8 of the 24 cells in the linear part).
        # The errors are about 6e-8.
        g = generate_grid(Hexahedron, (6, 2, 2), Vec((0.0, 0.0, 0.0)), Vec((6.0, 2.0, 2.0)))
        d, cv = redirect_stdout(() -> setup_problem(g), devnull)
        fixed = select_nodes_by_plane(g, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6)
        tip = select_nodes(g, x -> abs(x[1] - 6.0) < 1e-6 && abs(x[2]) < 1e-6)
        ch = redirect_stdout(() -> create_constraint_handler(d, fixed), devnull)

        E0, Emin, ν, p = 1.0, 1e-9, 0.3, 3.0
        acceleration, ρ_mat = [0.0, -50.0, 0.0], 1.0
        densities = [0.3 + 0.6 * (0.5 + 0.5 * sin(1.7 * i)) for i = 1:getncells(g)]

        for (body_force, rho_lin) in ((false, 0.0), (true, 0.0), (true, 0.5))
            model = create_simp_material_model(E0, ν, Emin, p; rho_lin = rho_lin)
            function add_loads!(f, ρ)
                if body_force
                    apply_variable_density_volume_force!(f, d, cv, acceleration, ρ .* ρ_mat)
                end
                apply_force!(f, d, collect(tip), [0.0, -1.0, 0.0])
            end
            compliance(ρ) = solve_static(d, cv, model, ρ, f -> add_loads!(f, ρ), ch)

            u, _ = compliance(densities)
            unit_body_forces =
                body_force ? calculate_unit_body_force_vectors(d, cv, acceleration .* ρ_mat) :
                nothing
            analytic = calculate_sensitivities(
                g,
                d,
                cv,
                densities,
                u,
                E0,
                Emin,
                ν,
                p;
                unit_body_forces = unit_body_forces,
                rho_lin = rho_lin,
            )

            step = 1e-6
            finite_difference = map(eachindex(densities)) do e
                ρ_plus = copy(densities)
                ρ_plus[e] += step
                ρ_minus = copy(densities)
                ρ_minus[e] -= step
                return (compliance(ρ_plus)[2] - compliance(ρ_minus)[2]) / (2 * step)
            end

            rel_error =
                maximum(abs.(analytic .- finite_difference)) / maximum(abs.(finite_difference))
            @test rel_error < 1e-6
        end
    end

    @testset "Filter radius from all cells" begin
        # The filter radius is filter_radius × the mean element size of ALL cells. On
        # the unstructured gripper mesh the first 10 cells give 2.775 mm, all cells
        # 3.9386 mm (median 3.981, range 2.10–5.70), so R = 1.5 is 5.908 mm.
        mesh_path = joinpath(@__DIR__, "..", "data", "stul14.vtu")
        gripper_grid = redirect_stdout(() -> import_mesh(mesh_path), devnull)
        element_sizes = EasySIMP.Optimization.calculate_element_sizes(gripper_grid)
        @test isapprox(estimate_element_size(gripper_grid), 3.9386; atol = 1e-4)
        @test estimate_element_size(gripper_grid) ≈ sum(element_sizes) / length(element_sizes)
    end
end
