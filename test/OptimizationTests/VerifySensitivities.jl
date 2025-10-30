"""
    verify_sensitivities(grid, dh, cellvalues, material_model, densities, u; 
                        perturbation=1e-6)

Verify analytical sensitivities using finite differences (for debugging).

# Arguments
- All parameters as in calculate_sensitivities
- `perturbation`: Size of finite difference perturbation

# Returns
- Comparison between analytical and finite difference sensitivities
"""
function verify_sensitivities(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    material_model,
    densities::Vector{Float64},
    u::Vector{Float64};
    perturbation::Float64 = 1e-6,
)
    n_cells = getncells(grid)

    # Calculate analytical sensitivities
    analytical = calculate_sensitivities(grid, dh, cellvalues, material_model, densities, u)

    # Calculate finite difference sensitivities
    finite_diff = zeros(n_cells)

    # Calculate baseline compliance
    c0 = calculate_compliance(grid, dh, cellvalues, material_model, densities, u)

    for i = 1:min(10, n_cells)  # Only check first 10 elements for efficiency
        # Perturb density
        densities_pert = copy(densities)
        densities_pert[i] += perturbation

        # Recalculate compliance
        K_pert = allocate_matrix(dh)
        f_pert = zeros(ndofs(dh))
        assemble_stiffness_matrix_simp!(
            K_pert,
            f_pert,
            dh,
            cellvalues,
            material_model,
            densities_pert,
        )
        u_pert = K_pert \ f_pert
        c_pert = calculate_compliance(
            grid,
            dh,
            cellvalues,
            material_model,
            densities_pert,
            u_pert,
        )

        # Finite difference approximation
        finite_diff[i] = (c_pert - c0) / perturbation
    end

    # Compare results
    println("Sensitivity verification (first 10 elements):")
    println("Element | Analytical | Finite Diff | Relative Error")
    for i = 1:min(10, n_cells)
        rel_error = abs(analytical[i] - finite_diff[i]) / (abs(analytical[i]) + 1e-12)
        @printf(
            "%7d | %10.4e | %11.4e | %13.4e\n",
            i,
            analytical[i],
            finite_diff[i],
            rel_error
        )
    end

    return analytical, finite_diff
end

"""
    calculate_compliance(grid, dh, cellvalues, material_model, densities, u)

Helper function to calculate total compliance.
"""
function calculate_compliance(
    grid::Grid,
    dh::DofHandler,
    cellvalues,
    material_model,
    densities::Vector{Float64},
    u::Vector{Float64},
)
    # Simple calculation: c = 0.5 * u^T * K * u
    # But we need to reassemble K for given densities

    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, densities)

    return 0.5 * dot(u, K * u)
end
