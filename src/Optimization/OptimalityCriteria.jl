"""
OptimalityCriteria.jl

Implementation of Optimality Criteria method for SIMP topology optimization.
Based on Sigmund (2001) 99-line topology optimization code.
"""

export optimality_criteria_update, check_sensitivity_health

"""
    check_sensitivity_health(sensitivities)

Check if sensitivities are in a reasonable range for OC algorithm.
Warns user only if problems are detected.
"""
function check_sensitivity_health(sensitivities::Vector{Float64})
    max_sens = maximum(abs.(sensitivities))
    min_sens = minimum(abs.(sensitivities[sensitivities .< 0]))  # Only negative sensitivities
    
    if max_sens < 1e-8
        @warn "Sensitivities too small (max: $max_sens). Consider increasing force magnitude or reducing material stiffness."
        return false
    elseif max_sens > 1e2
        @warn "Sensitivities too large (max: $max_sens). Consider reducing force magnitude or increasing material stiffness."
        return false
    elseif count(s -> s < 0, sensitivities) < length(sensitivities) * 0.5
        @warn "Less than 50% of sensitivities are negative. Check if compliance sensitivities are computed correctly."
        return false
    end
    
    return true
end

"""
    optimality_criteria_update(densities, sensitivities, target_volume_fraction, 
                              total_volume, element_volumes, move_limit, damping)

Update design variables using the Optimality Criteria method.

# Arguments
- `densities`: Current density distribution
- `sensitivities`: Sensitivity of objective function w.r.t. densities  
- `target_volume_fraction`: Target volume fraction
- `total_volume`: Total volume of the design domain
- `element_volumes`: Volume of each element
- `move_limit`: Maximum change in density per iteration (default: 0.2)
- `damping`: Damping coefficient (default: 0.5)

# Returns
- Updated density distribution

# Method
Uses Sigmund's OC formula:
x_new = max(x_min, max(x - move, min(1, min(x + move, x * sqrt(-dc/λ)))))

where λ is found by bisection to satisfy the volume constraint.
"""
function optimality_criteria_update(
    densities::Vector{Float64},
    sensitivities::Vector{Float64}, 
    target_volume_fraction::Float64,
    total_volume::Float64,
    element_volumes::Vector{Float64},
    move_limit::Float64 = 0.2,
    damping::Float64 = 0.5
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
    
    for iter = 1:max_iter
        λ_mid = 0.5 * (λ_low + λ_high)
        
        # Update densities using Sigmund's OC formula
        for i = 1:n_elements
            if sensitivities[i] < 0
                # Volume sensitivity (normalized)
                volume_sensitivity = element_volumes[i] / total_volume
                
                # Optimality ratio: Be = (-dc/dx) / (λ * dV/dx)
                Be = (-sensitivities[i]) / (λ_mid * volume_sensitivity)
                
                # Update with damping
                optimality_ratio = densities[i] * (Be^damping)
                
                # Apply move limits and bounds
                new_densities[i] = max(
                    x_min,
                    max(
                        densities[i] - move_limit,
                        min(
                            1.0,
                            min(
                                densities[i] + move_limit,
                                optimality_ratio
                            )
                        )
                    )
                )
            end
        end
        
        # Check volume constraint
        current_volume = dot(new_densities, element_volumes)
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
    
    return new_densities
end

"""
    check_volume_constraint(densities, target_volume_fraction, total_volume, element_volumes)

Check volume constraint satisfaction.
Returns volume error and current volume fraction.
"""
function check_volume_constraint(
    densities::Vector{Float64},
    target_volume_fraction::Float64,
    total_volume::Float64,
    element_volumes::Vector{Float64}
)
    current_volume = dot(densities, element_volumes)
    target_volume = target_volume_fraction * total_volume
    current_volume_fraction = current_volume / total_volume
    
    volume_error = current_volume - target_volume
    
    return volume_error, current_volume_fraction
end
