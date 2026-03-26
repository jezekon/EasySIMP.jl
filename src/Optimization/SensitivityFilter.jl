"""
SensitivityFilter.jl - Sigmund's sensitivity filter for SIMP topology optimization

Applies sensitivity filtering using pre-computed FilterCache from FilterCommon.jl.
"""

using Ferrite
using LinearAlgebra

export apply_sensitivity_filter_cached!

# =============================================================================
# CACHED FILTER APPLICATION (USE IN OPTIMIZATION LOOP)
# =============================================================================

"""
    apply_sensitivity_filter_cached!(filtered_sens, cache, densities, sensitivities)

Apply Sigmund's sensitivity filter using pre-computed neighbors.
Zero allocations - use in optimization loops.

Formula: filtered = Σ(H_ij ρ_j (∂J/∂ρ_j) / V_j) / (ρ_e/V_e Σ H_ij)

# Arguments
- `filtered_sens`: Output vector (modified in-place)
- `cache`: Pre-computed FilterCache (includes element_volumes)
- `densities`: Current density distribution
- `sensitivities`: Raw sensitivities to filter

# Returns
- `filtered_sens` (modified in-place)
"""
function apply_sensitivity_filter_cached!(
    filtered_sens::Vector{Float64},
    cache::FilterCache,
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
)
    n_cells = length(densities)
    cell_centers = cache.cell_centers
    filter_radius = cache.filter_radius
    volumes = cache.element_volumes

    @inbounds for i = 1:n_cells
        numerator = 0.0
        denominator = 0.0
        center_i = cell_centers[i]

        for j in cache.neighbor_lists[i]
            distance = norm(cell_centers[j] - center_i)
            weight = max(0.0, filter_radius - distance)

            if weight > 0.0
                numerator += weight * densities[j] * sensitivities[j] / volumes[j]
                denominator += weight
            end
        end

        # Guard against division by near-zero density — Sigmund (2007), below eq. 16
        rho_safe = max(1e-3, densities[i])
        filtered_sens[i] =
            denominator > 1e-12 ? numerator / (rho_safe / volumes[i] * denominator) :
            sensitivities[i]
    end

    return filtered_sens
end

# =============================================================================
# FILTER INFO
# =============================================================================

"""
    print_filter_info(grid, filter_radius_ratio, filter_type="auto")

Print filter settings information.
"""
function print_filter_info(
    grid::Grid,
    filter_radius_ratio::Float64,
    filter_type::String = "auto",
)
    char_size = estimate_element_size(grid)
    element_sizes = calculate_element_sizes(grid)
    size_variation = maximum(element_sizes) / minimum(element_sizes)

    cell = getcells(grid, 1)
    cell_type = cell isa Ferrite.Tetrahedron ? "Tetrahedron" : "Hexahedron"

    println("Sensitivity filter information:")
    println("  Element type: $cell_type")
    println("  Characteristic element size: $(round(char_size, digits=4))")
    println("  Element size variation: $(round(size_variation, digits=2))")
    println("  Filter radius ratio: $filter_radius_ratio")
    println("  Actual filter radius: $(round(filter_radius_ratio * char_size, digits=4))")

    actual_type =
        filter_type == "auto" ? (size_variation > 1.5 ? "adaptive" : "uniform") :
        filter_type
    println("  Filter type: $actual_type")
end
