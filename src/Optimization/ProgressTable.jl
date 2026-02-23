# src/Optimization/ProgressTable.jl
# Add this to src/Optimization/Optimization.jl module includes section
"""
ProgressTable.jl

Clean tabular progress display for SIMP optimization iterations.
"""

using Printf

export OptimizationProgress, print_header, print_iteration, print_final

"""
    OptimizationProgress

Structure to track optimization progress data for clean display.
"""
mutable struct OptimizationProgress
    iteration::Int
    volume_fraction::Float64
    energy::Float64
    force_magnitude::Float64
    sensitivity_warning::Bool
    change::Float64
end

"""
    print_header()

Print the table header for optimization progress.
"""
function print_header()
    println()
    println("=" ^ 80)
    println("SIMP TOPOLOGY OPTIMIZATION PROGRESS")
    println("=" ^ 80)
    println(
        @sprintf(
            "%-4s │ %-12s │ %-12s │ %-12s │ %-8s │ %-s",
            "Iter",
            "Vol. Frac.",
            "Energy",
            "Force Mag.",
            "Change",
            "Warnings"
        )
    )
    println("─" ^ 80)
end

"""
    print_iteration(progress::OptimizationProgress)

Print a single iteration row in the progress table.
"""
function print_iteration(progress::OptimizationProgress)
    # Format volume fraction
    vol_str = @sprintf("%.6f", progress.volume_fraction)

    # Format energy (use scientific notation if very large/small)
    if progress.energy > 1e6 || progress.energy < 1e-3
        comp_str = @sprintf("%.3e", progress.energy)
    else
        comp_str = @sprintf("%.3f", progress.energy)
    end

    # Format force magnitude
    if progress.force_magnitude > 1e6 || progress.force_magnitude < 1e-3
        force_str = @sprintf("%.3e", progress.force_magnitude)
    else
        force_str = @sprintf("%.3f", progress.force_magnitude)
    end

    # Format change
    change_str = @sprintf("%.6f", progress.change)

    # Warning indicators
    warning_str = progress.sensitivity_warning ? "SENS!" : ""

    println(
        @sprintf(
            "%-4d │ %-12s │ %-12s │ %-12s │ %-8s │ %-s",
            progress.iteration,
            vol_str,
            comp_str,
            force_str,
            change_str,
            warning_str
        )
    )
end

"""
    print_final(final_progress::OptimizationProgress, converged::Bool)

Print final optimization results summary.
"""
function print_final(final_progress::OptimizationProgress, converged::Bool)
    println("─" ^ 80)

    if converged
        println("✅ OPTIMIZATION CONVERGED")
    else
        println("⚠️  OPTIMIZATION STOPPED (Max iterations reached)")
    end

    println()
    println("FINAL RESULTS:")
    println("  Iterations:     $(final_progress.iteration)")
    println("  Volume Frac.:   $(final_progress.volume_fraction)")
    println("  Energy:     $(final_progress.energy)")
    println("  Force Mag.:     $(final_progress.force_magnitude)")
    println("=" ^ 80)
    println()
end

"""
    check_sensitivity_health_quiet(sensitivities)

Quiet version of sensitivity health check that returns warning flag.
"""
function check_sensitivity_health_quiet(sensitivities::Vector{Float64})
    max_sens = maximum(abs.(sensitivities))
    min_sens_count = count(s -> s < 0, sensitivities)
    min_sens_ratio = min_sens_count / length(sensitivities)

    # Check for problematic sensitivities
    if max_sens < 1e-8
        return true  # Too small
    elseif max_sens > 1e2
        return true  # Too large  
    elseif min_sens_ratio < 0.5
        return true  # Too few negative sensitivities
    end

    return false
end
