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
    target_volume_fraction::Float64,
    total_volume::Float64,
    element_volumes::Vector{Float64},
    move_limit::Float64 = 0.2,
    damping::Float64 = 0.5
)
    # Nastavit testovací volume_fraction
    # target_volume_fraction = 0.1  # Testovací hodnota
    target_volume = target_volume_fraction * total_volume
    
    println("=== OC DEBUG INFO ===")
    println("Target volume fraction: $target_volume_fraction")
    println("Target volume: $target_volume")
    println("Total volume: $total_volume")
    println("Number of elements: $(length(densities))")
    println("Current volume: $(dot(densities, element_volumes))")
    println("Current volume fraction: $(dot(densities, element_volumes) / total_volume)")
    
    # Kontrola vstupních dat
    println("Density range: [$(minimum(densities)), $(maximum(densities))]")
    println("Sensitivity range: [$(minimum(sensitivities)), $(maximum(sensitivities))]")
    println("Element volume range: [$(minimum(element_volumes)), $(maximum(element_volumes))]")
    
    # Zkusit velmi jednoduchou implementaci podle Sigmund 2001
    n_elements = length(densities)
    x_min = 1e-3
    
    # Bisection pro λ
    λ_low = 1e-9
    λ_high = 1e9
    tolerance = 1e-6
    
    new_densities = copy(densities)
    
    for iter = 1:50
        λ_mid = 0.5 * (λ_low + λ_high)
        
        # Sigmund (2001) formula: x_new = x * (-dc/dx / λ)^0.5
        for i = 1:n_elements
            if sensitivities[i] < 0
                # Normalized: dV/dx = 1 for unit volumes
                # For actual volumes: dV/dx = element_volumes[i] / total_volume
                volume_sensitivity = element_volumes[i] / total_volume
                
                # Be = (-dc/dx) / (λ * dV/dx)
                Be = (-sensitivities[i]) / (λ_mid * volume_sensitivity)
                
                # Update podle Sigmund
                optimality_ratio = densities[i] * sqrt(Be)  # damping = 0.5
                
                # Apply bounds and move limits
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
        
        # Kontrola objemu
        current_volume = dot(new_densities, element_volumes)
        volume_error = current_volume - target_volume
        
        if iter <= 20
            println("Iter $iter: λ=$λ_mid, volume=$current_volume, error=$volume_error")
        end
        
        if abs(volume_error) < tolerance
            println("Converged in $iter iterations!")
            break
        end
        
        # Update bounds
        if volume_error > 0
            λ_low = λ_mid
        else
            λ_high = λ_mid
        end
    end
    
    # Final check
    final_volume = dot(new_densities, element_volumes)
    final_fraction = final_volume / total_volume
    println("=== FINAL RESULTS ===")
    println("Achieved volume fraction: $final_fraction")
    println("Target volume fraction: $target_volume_fraction")
    println("Error: $(abs(final_fraction - target_volume_fraction))")
    
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
# function verify_oc_implementation(
#     densities::Vector{Float64},
#     sensitivities::Vector{Float64},
#     volume_fraction::Float64,
#     grid::Grid
# )
#     print_info("Verifying OC implementation...")
#
#     # Test with different move limits and damping
#     test_move_limits = [0.1, 0.2, 0.3]
#     test_damping = [0.3, 0.5, 0.7]
#
#     for move in test_move_limits
#         for damp in test_damping
#             new_densities = optimality_criteria_update(
#                 densities, sensitivities, volume_fraction, grid, move, damp
#             )
#
#             volume_error, vol_frac = check_volume_constraint_correct(
#                 new_densities, volume_fraction, grid
#             )
#
#             print_data("Move: $move, Damping: $damp")
#             print_data("  Volume fraction: $vol_frac (target: $volume_fraction)")
#             print_data("  Volume error: $volume_error")
#             println()
#         end
#     end
# end

