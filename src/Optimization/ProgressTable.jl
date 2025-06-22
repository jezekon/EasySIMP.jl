# src/Optimization/ProgressTable.jl
"""
ProgressTable.jl

Enhanced tabular progress display for SIMP optimization iterations with solver performance metrics.
"""

using Printf

export OptimizationProgress, print_header, print_iteration, print_final, print_solver_performance_summary

"""
    OptimizationProgress

Structure to track optimization progress data for clean display with solver information.
"""
mutable struct OptimizationProgress
    iteration::Int
    volume_fraction::Float64
    compliance::Float64
    force_magnitude::Float64
    sensitivity_warning::Bool
    change::Float64
    solver_strategy::Union{Symbol, Nothing}
    cg_iterations::Int
    density_change::Float64
end

# Simple constructor for backward compatibility
function OptimizationProgress(iteration, volume_fraction, compliance, force_magnitude, sensitivity_warning, change)
    return OptimizationProgress(iteration, volume_fraction, compliance, force_magnitude, 
                               sensitivity_warning, change, nothing, 0, 0.0)
end

"""
    print_header()

Print the enhanced table header for optimization progress including solver information.
"""
function print_header()
    println()
    println("=" ^ 95)
    println("SIMP TOPOLOGY OPTIMIZATION PROGRESS (Enhanced with Solver Metrics)")
    println("=" ^ 95)
    println(@sprintf("%-4s │ %-10s │ %-11s │ %-10s │ %-8s │ %-12s │ %-8s │ %-s", 
                     "Iter", "Vol.Frac.", "Compliance", "Force Mag.", "Change", "Solver", "CG Its", "Notes"))
    println("─" ^ 95)
end

"""
    print_iteration(progress::OptimizationProgress)

Print a single iteration row in the enhanced progress table.
"""
function print_iteration(progress::OptimizationProgress)
    # Format volume fraction
    vol_str = @sprintf("%.6f", progress.volume_fraction)
    
    # Format compliance (use scientific notation if very large/small)
    if progress.compliance > 1e6 || progress.compliance < 1e-3
        comp_str = @sprintf("%.3e", progress.compliance)
    else
        comp_str = @sprintf("%.3f", progress.compliance)
    end
    
    # Format force magnitude
    if progress.force_magnitude > 1e6 || progress.force_magnitude < 1e-3
        force_str = @sprintf("%.3e", progress.force_magnitude)
    else
        force_str = @sprintf("%.3f", progress.force_magnitude)
    end
    
    # Format change
    change_str = @sprintf("%.6f", progress.change)
    
    # Format solver strategy
    solver_str = if progress.solver_strategy === nothing
        "Direct"
    else
        strategy_map = Dict(
            :direct_solve => "Direct",
            :warm_start_strict => "WS-Strict",
            :warm_start_relaxed => "WS-Relax"
        )
        get(strategy_map, progress.solver_strategy, string(progress.solver_strategy))
    end
    
    # Format CG iterations
    cg_str = if progress.cg_iterations > 0
        @sprintf("%d", progress.cg_iterations)
    elseif progress.cg_iterations == -1
        "Fallback"
    else
        "-"
    end
    
    # Warning indicators and notes
    notes = String[]
    if progress.sensitivity_warning
        push!(notes, "SENS!")
    end
    if progress.density_change > 0.0 && progress.density_change < 0.005
        push!(notes, "SmΔρ")  # Small density change
    end
    if progress.cg_iterations > 500
        push!(notes, "SlowCG")
    end
    
    warning_str = join(notes, ",")
    
    println(@sprintf("%-4d │ %-10s │ %-11s │ %-10s │ %-8s │ %-12s │ %-8s │ %-s", 
                     progress.iteration, vol_str, comp_str, force_str, change_str, solver_str, cg_str, warning_str))
end

"""
    print_final(final_progress::OptimizationProgress, converged::Bool, 
                total_cg_iterations::Int, avg_cg_per_iteration::Float64)

Print final optimization results summary with solver performance.
"""
function print_final(final_progress::OptimizationProgress, converged::Bool, 
                    total_cg_iterations::Int = 0, avg_cg_per_iteration::Float64 = 0.0)
    println("─" ^ 95)
    
    if converged
        println("✅ OPTIMIZATION CONVERGED")
    else
        println("⚠️  OPTIMIZATION STOPPED (Max iterations reached)")
    end
    
    println()
    println("FINAL RESULTS:")
    println("  Iterations:      $(final_progress.iteration)")
    println("  Volume Frac.:    $(final_progress.volume_fraction)")
    println("  Compliance:      $(final_progress.compliance)")
    println("  Force Mag.:      $(final_progress.force_magnitude)")
    
    if total_cg_iterations > 0
        println()
        println("SOLVER PERFORMANCE:")
        println("  Total CG Iterations:    $total_cg_iterations")
        println("  Avg CG per Iteration:   $(round(avg_cg_per_iteration, digits=1))")
        
        # Estimate performance improvement
        if avg_cg_per_iteration < 50
            println("  Performance:            🚀 Excellent (warm-start working well)")
        elseif avg_cg_per_iteration < 150
            println("  Performance:            ⚡ Good (moderate CG iterations)")
        else
            println("  Performance:            ⚠️  Poor (consider direct solver)")
        end
    end
    
    println("=" ^ 95)
    println()
end

"""
    print_solver_performance_summary(solver_strategies::Vector{Symbol}, 
                                   total_cg_iterations::Int, total_iterations::Int)

Print a summary of solver strategy usage and performance.
"""
function print_solver_performance_summary(solver_strategies::Vector{Symbol}, 
                                        total_cg_iterations::Int, total_iterations::Int)
    if isempty(solver_strategies)
        return
    end
    
    println()
    println("SOLVER STRATEGY SUMMARY:")
    println("─" ^ 40)
    
    # Count strategy usage
    strategy_counts = Dict{Symbol, Int}()
    for strategy in solver_strategies
        strategy_counts[strategy] = get(strategy_counts, strategy, 0) + 1
    end
    
    # Print strategy distribution
    for (strategy, count) in sort(collect(strategy_counts))
        percentage = round(100 * count / length(solver_strategies), digits=1)
        strategy_name = Dict(
            :direct_solve => "Direct Solve",
            :warm_start_strict => "Warm-Start (Strict)",
            :warm_start_relaxed => "Warm-Start (Relaxed)"
        )[strategy]
        
        println("  $strategy_name: $count iterations ($(percentage)%)")
    end
    
    # Performance analysis
    if total_cg_iterations > 0
        avg_cg = round(total_cg_iterations / total_iterations, digits=1)
        theoretical_direct = total_iterations  # Assume ~1 "iteration" per direct solve
        
        if avg_cg < theoretical_direct * 0.8
            efficiency = "🎯 Excellent"
        elseif avg_cg < theoretical_direct * 1.2
            efficiency = "✅ Good"
        else
            efficiency = "⚠️  Suboptimal"
        end
        
        println()
        println("  Avg CG iterations/solve: $avg_cg")
        println("  Overall efficiency: $efficiency")
        
        # Recommendations
        warm_start_fraction = get(strategy_counts, :warm_start_strict, 0) + get(strategy_counts, :warm_start_relaxed, 0)
        warm_start_fraction /= length(solver_strategies)
        
        if warm_start_fraction < 0.5
            println("  💡 Tip: Consider enabling warm-start earlier (reduce early_iteration_threshold)")
        elseif avg_cg > 200
            println("  💡 Tip: Consider updating preconditioner more frequently")
        end
    end
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

"""
    analyze_convergence_rate(compliance_history::Vector{Float64})

Analyze convergence behavior and provide recommendations.
"""
function analyze_convergence_rate(compliance_history::Vector{Float64})
    if length(compliance_history) < 5
        return "Insufficient data"
    end
    
    # Calculate recent improvement rate
    recent_start = max(1, length(compliance_history) - 5)
    recent_improvement = (compliance_history[recent_start] - compliance_history[end]) / compliance_history[recent_start]
    
    # Overall improvement
    total_improvement = (compliance_history[1] - compliance_history[end]) / compliance_history[1]
    
    if recent_improvement < 0.001  # Less than 0.1% improvement in last 5 iterations
        return "🐌 Slow convergence (consider stopping or adjusting parameters)"
    elseif total_improvement > 0.1  # More than 10% total improvement
        return "🚀 Good convergence"
    else
        return "⚡ Moderate convergence"
    end
end

"""
    print_iteration_with_solver_info(iteration, volume_fraction, compliance, force_magnitude,
                                   change, solver_strategy, cg_iterations, density_change,
                                   sensitivity_warning=false)

Convenience function to print iteration with solver information.
"""
function print_iteration_with_solver_info(iteration, volume_fraction, compliance, force_magnitude,
                                        change, solver_strategy, cg_iterations, density_change,
                                        sensitivity_warning=false)
    progress = OptimizationProgress(
        iteration, volume_fraction, compliance, force_magnitude,
        sensitivity_warning, change, solver_strategy, cg_iterations, density_change
    )
    print_iteration(progress)
end
