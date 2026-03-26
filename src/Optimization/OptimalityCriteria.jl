"""
OptimalityCriteria.jl

Implementation of Optimality Criteria method for SIMP topology optimization.
Based on Sigmund (2001) 99-line topology optimization code.
"""

using Statistics: median

export optimality_criteria_update, check_sensitivity_health

"""
    check_sensitivity_health(sensitivities)

Check if sensitivities are in a reasonable range for OC algorithm.
Uses relative thresholds (median-based) instead of absolute values.
Warns user only if problems are detected.
"""
function check_sensitivity_health(sensitivities::Vector{Float64})
    if count(s -> s < 0, sensitivities) < length(sensitivities) * 0.5
        @warn "Less than 50% of sensitivities are negative. Check if energy sensitivities are computed correctly."
        return false
    end

    abs_sens = abs.(sensitivities)
    med = median(abs_sens)

    if med < eps(Float64)
        @warn "Sensitivities are effectively zero (median: $med)."
        return false
    end

    range_ratio = maximum(abs_sens) / max(med, eps(Float64))
    if range_ratio > 1e8
        @warn "Sensitivity range too large (max/median: $(range_ratio)). Check problem scaling."
        return false
    end

    return true
end

"""
    optimality_criteria_update(densities, sensitivities, volume_sensitivities,
                              target_volume_fraction, total_volume, element_volumes,
                              move_limit, damping)

Update design variables using the Optimality Criteria method.

# Arguments
- `densities`: Current density distribution
- `sensitivities`: Sensitivity of objective function w.r.t. design densities
- `volume_sensitivities`: Sensitivity of volume constraint w.r.t. design densities
  (chain-rule transformed for density filter, raw V_i/V_total for sensitivity filter)
- `target_volume_fraction`: Target volume fraction
- `total_volume`: Total volume of the design domain
- `element_volumes`: Volume of each element
- `move_limit`: Maximum change in density per iteration (default: 0.2)
- `damping`: Damping coefficient (default: 0.5)

# Returns
- `(new_densities, lagrange_multiplier)`: Updated densities and final λ

# Method
Uses Sigmund's OC formula:
x_new = max(x_min, max(x - move, min(1, min(x + move, x * sqrt(-dc/λ)))))

where λ is found by bisection to satisfy the volume constraint.
"""
function optimality_criteria_update(
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    volume_sensitivities::Vector{Float64},
    target_volume_fraction::Float64,
    total_volume::Float64,
    element_volumes::Vector{Float64},
    move_limit::Float64 = 0.2,
    damping::Float64 = 0.5;
    filter_cache::Union{FilterCache, Nothing} = nothing,
    use_density_filter::Bool = false,
)
    # Check sensitivity health (warns only if problems detected)
    check_sensitivity_health(sensitivities)

    n_elements = length(densities)
    target_volume = target_volume_fraction * total_volume
    x_min = 1e-3

    # Bisection algorithm to find Lagrange multiplier
    λ_low = 1e-9
    λ_high = 1e9
    tolerance = 1e-6
    max_iter = 200

    new_densities = copy(densities)
    filtered_new_densities = use_density_filter ? zeros(n_elements) : new_densities

    λ_mid = NaN
    for iter = 1:max_iter
        λ_mid = 0.5 * (λ_low + λ_high)

        # Update densities using Sigmund's OC formula
        for i = 1:n_elements
            # Optimality ratio: Be = |dc/dx| / (λ * dV/dx)
            Be = abs(sensitivities[i]) / (λ_mid * volume_sensitivities[i])

            # Update with damping
            optimality_ratio = densities[i] * (Be^damping)

            # Apply move limits and bounds
            new_densities[i] = max(
                x_min,
                max(
                    densities[i] - move_limit,
                    min(1.0, min(densities[i] + move_limit, optimality_ratio)),
                ),
            )
        end

        # Apply density filter before volume check (physical densities for constraint)
        if use_density_filter
            apply_density_filter_cached!(filtered_new_densities, filter_cache, new_densities)
        end

        # Check volume constraint on physical densities
        current_volume = dot(filtered_new_densities, element_volumes)
        volume_error = current_volume - target_volume

        if abs(volume_error) < tolerance
            break
        end

        # Update λ bounds for bisection
        if volume_error > 0
            λ_low = λ_mid  # Too much material, increase λ
        else
            λ_high = λ_mid  # Too little material, decrease λ
        end

        # Check for convergence failure
        if iter == max_iter
            @warn "OC bisection did not converge within $max_iter iterations. Volume error: $volume_error"
        end
    end

    return new_densities, λ_mid
end

