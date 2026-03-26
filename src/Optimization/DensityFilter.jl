"""
DensityFilter.jl - Density filtering for SIMP topology optimization

Implements the density filter and its chain rule transformation for
converting sensitivities between physical and design variable spaces.
"""

using LinearAlgebra

export apply_density_filter_cached!, apply_density_filter_chain_rule_cached!

# =============================================================================
# DENSITY FILTER (CACHED)
# =============================================================================

"""
    apply_density_filter_cached!(filtered_densities, cache, densities)

Apply density filter using pre-computed neighbors. Replaces each element density
with a volume-weighted average over neighbors within the filter radius.

Formula: ρ̃_e = Σ H_ei V_i ρ_i / Σ H_ei V_i
where H_ei = max(0, R - ||x_e - x_i||)

# Arguments
- `filtered_densities`: Output vector (modified in-place)
- `cache`: Pre-computed FilterCache
- `densities`: Current density distribution (design variables)
"""
function apply_density_filter_cached!(
    filtered_densities::Vector{Float64},
    cache::FilterCache,
    densities::Vector{Float64},
)
    n_cells = length(densities)
    cell_centers = cache.cell_centers
    filter_radius = cache.filter_radius
    element_volumes = cache.element_volumes

    @inbounds for i = 1:n_cells
        numerator = 0.0
        denominator = 0.0
        center_i = cell_centers[i]

        for j in cache.neighbor_lists[i]
            distance = norm(cell_centers[j] - center_i)
            weight = max(0.0, filter_radius - distance)

            if weight > 0.0
                wv = weight * element_volumes[j]
                numerator += wv * densities[j]
                denominator += wv
            end
        end

        filtered_densities[i] = denominator > 1e-12 ? numerator / denominator : densities[i]
    end

    return filtered_densities
end

"""
    apply_density_filter_chain_rule_cached!(filtered_sens, cache, sensitivities)

Apply chain rule for density filter to transform sensitivities from
filtered (physical) space back to design variable space.

Formula: ∂f/∂ρ_e = Σ_{i∈N_e} (H_ie V_e / Σ_j H_ij V_j) * ∂f/∂ρ̃_i

Uses the transpose of the density filter operator.

# Arguments
- `filtered_sens`: Output vector (modified in-place)
- `cache`: Pre-computed FilterCache
- `sensitivities`: Sensitivities w.r.t. filtered densities (∂f/∂ρ̃)
"""
function apply_density_filter_chain_rule_cached!(
    filtered_sens::Vector{Float64},
    cache::FilterCache,
    sensitivities::Vector{Float64},
)
    n_cells = length(sensitivities)
    cell_centers = cache.cell_centers
    filter_radius = cache.filter_radius
    element_volumes = cache.element_volumes

    fill!(filtered_sens, 0.0)

    # Transpose operation: for each element i, distribute its sensitivity
    # contribution to all its neighbors e
    @inbounds for i = 1:n_cells
        center_i = cell_centers[i]

        # Compute denominator for element i: Σ_j H_ij V_j
        denominator_i = 0.0
        for j in cache.neighbor_lists[i]
            distance = norm(cell_centers[j] - center_i)
            weight = max(0.0, filter_radius - distance)
            if weight > 0.0
                denominator_i += weight * element_volumes[j]
            end
        end

        if denominator_i > 1e-12
            for e in cache.neighbor_lists[i]
                distance = norm(cell_centers[e] - center_i)
                weight = max(0.0, filter_radius - distance)
                if weight > 0.0
                    filtered_sens[e] +=
                        (weight * element_volumes[e] / denominator_i) * sensitivities[i]
                end
            end
        end
    end

    return filtered_sens
end
