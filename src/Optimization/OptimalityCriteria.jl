"""
OptimalityCriteria.jl

Corrected implementation of Optimality Criteria method for SIMP topology optimization.
Based on Sigmund (2001) 99-line topology optimization code.
"""

export optimality_criteria_update

"""
    optimality_criteria_update(densities, sensitivities, volume_fraction, 
                              grid, move_limit, damping)

Update design variables using the corrected Optimality Criteria method.

# Arguments
- `densities`: Current density distribution
- `sensitivities`: Sensitivity of objective function w.r.t. densities  
- `volume_fraction`: Target volume fraction
- `grid`: Ferrite Grid object for proper volume calculation
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
    volume_fraction::Float64,
    grid::Grid,
    move_limit::Float64 = 0.2,
    damping::Float64 = 0.5
)
    n_elements = length(densities)
    
    # Calculate proper total volume using the grid geometry
    total_volume = calculate_volume(grid)
    target_volume = volume_fraction * total_volume
    
    # Minimum density to avoid singularity
    x_min = 1e-3
    
    # Initialize Lagrange multiplier bounds (as in Sigmund 2001)
    l1 = 0.0
    l2 = 100000.0
    
    # Bisection algorithm parameters
    max_bisection_iter = 50
    tolerance = 1e-4  # Sigmund uses 1e-4
    
    new_densities = copy(densities)
    
    # Bisection algorithm to find Lagrange multiplier
    for iter = 1:max_bisection_iter
        lmid = 0.5 * (l1 + l2)
        
        # Update densities using Sigmund's OC formula
        for i = 1:n_elements
            x = densities[i]
            dc = sensitivities[i]
            
            # Sigmund's exact formula: x * sqrt(-dc/λ)
            if dc < 0  # Compliance sensitivities should be negative
                optimality_ratio = x * sqrt(-dc / lmid)  # ← Tady je ta oprava!
            else
                optimality_ratio = x  # No change if sensitivity is not negative
            end
            
            # Apply move limits (Sigmund's nested min/max)
            new_densities[i] = max(
                x_min,
                max(
                    x - move_limit,
                    min(
                        1.0,
                        min(
                            x + move_limit,
                            optimality_ratio  # ← Bez dodatečného damping exponentu!
                        )
                    )
                )
            )
        end
                
        # Calculate resulting volume using proper volume calculation
        current_volume = calculate_volume(grid, new_densities)
        
        # Volume constraint error
        volume_error = current_volume - target_volume
        
        # Check for convergence
        if abs(l2 - l1) < tolerance
            print_data("OC converged after $iter bisection iterations")
            break
        end
        
        # Update Lagrange multiplier bounds
        if volume_error > 0
            l1 = lmid  # Too much material, increase λ
        else
            l2 = lmid  # Too little material, decrease λ
        end
        
        # Check for convergence failure
        if iter == max_bisection_iter
            print_warning("OC bisection did not converge within $max_bisection_iter iterations")
            print_data("Final volume error: $volume_error")
            print_data("Current volume: $current_volume, Target: $target_volume")
        end
    end
    
    return new_densities
end

"""
    check_volume_constraint_correct(densities, target_volume_fraction, grid)

Check volume constraint using proper volume calculation.
"""
function check_volume_constraint_correct(
    densities::Vector{Float64},
    target_volume_fraction::Float64,
    grid::Grid
)
    total_volume = calculate_volume(grid)
    current_volume = calculate_volume(grid, densities)
    target_volume = target_volume_fraction * total_volume
    
    volume_error = current_volume - target_volume
    current_volume_fraction = current_volume / total_volume
    
    return volume_error, current_volume_fraction
end

"""
    verify_oc_implementation(densities, sensitivities, volume_fraction, grid)

Verify OC implementation by checking volume constraint satisfaction.
"""
function verify_oc_implementation(
    densities::Vector{Float64},
    sensitivities::Vector{Float64},
    volume_fraction::Float64,
    grid::Grid
)
    print_info("Verifying OC implementation...")
    
    # Test with different move limits and damping
    test_move_limits = [0.1, 0.2, 0.3]
    test_damping = [0.3, 0.5, 0.7]
    
    for move in test_move_limits
        for damp in test_damping
            new_densities = optimality_criteria_update(
                densities, sensitivities, volume_fraction, grid, move, damp
            )
            
            volume_error, vol_frac = check_volume_constraint_correct(
                new_densities, volume_fraction, grid
            )
            
            print_data("Move: $move, Damping: $damp")
            print_data("  Volume fraction: $vol_frac (target: $volume_fraction)")
            print_data("  Volume error: $volume_error")
            println()
        end
    end
end

"""
    apply_move_limits(old_density, new_density, move_limit, x_min, x_max)

Apply move limits to density update.
"""
function apply_move_limits(
    old_density::Float64,
    new_density::Float64, 
    move_limit::Float64,
    x_min::Float64 = 1e-3,
    x_max::Float64 = 1.0
)
    # Apply move limits
    bounded_density = max(
        old_density - move_limit,
        min(
            old_density + move_limit,
            new_density
        )
    )
    
    # Apply physical bounds
    return max(x_min, min(x_max, bounded_density))
end
