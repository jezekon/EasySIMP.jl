# Performance Test Example for Optimized SIMP
# This example demonstrates the performance improvements from warm-start solvers

using EasySIMP
using BenchmarkTools
using Printf

"""
    run_performance_comparison()

Compare performance between standard and optimized SIMP implementations.
"""
function run_performance_comparison()
    println("🚀 PERFORMANCE COMPARISON: Standard vs Optimized SIMP")
    println("="^60)
    
    # Import mesh
    grid = import_mesh("../data/cantilever_beam.vtu")
    println("Mesh: $(getncells(grid)) elements, $(getnnodes(grid)) nodes")
    
    # Material properties
    E0 = 200.0
    ν = 0.3
    
    # Setup FEM
    dh, cellvalues, K, f = setup_problem(grid)
    
    # Boundary conditions
    fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
    force_nodes = select_nodes_by_circle(grid, [60.0, 0.0, 2.0], [1.0, 0.0, 0.0], 1.0)
    
    if isempty(force_nodes)
        target_point = [60.0, 0.0, 2.0]
        min_dist = Inf
        closest_node = 1
        for node_id = 1:getnnodes(grid)
            node_coord = grid.nodes[node_id].x
            dist = norm(node_coord - target_point)
            if dist < min_dist
                min_dist = dist
                closest_node = node_id
            end
        end
        force_nodes = Set([closest_node])
    end
    
    ch_fixed = apply_fixed_boundary!(copy(K), copy(f), dh, fixed_nodes)
    
    # Test 1: Standard optimization (direct solver)
    println("\n📊 Test 1: Standard Optimization (Direct Solver)")
    
    standard_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-6,
        ν = ν,
        p = 3.0,
        volume_fraction = 0.4,
        max_iterations = 15,
        tolerance = 0.01,
        filter_radius = 2.5,
        move_limit = 0.1,
        damping = 0.5,
        solver_options = SolverOptions(use_warm_start = false)  # Disable optimizations
    )
    
    time_standard = @elapsed begin
        results_standard = simp_optimize(
            grid, dh, cellvalues,
            [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], 
            [ch_fixed], 
            standard_params
        )
    end
    
    # Test 2: Optimized with warm-start
    println("\n🚀 Test 2: Optimized with Warm-Start Solver")
    
    optimized_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-6,
        ν = ν,
        p = 3.0,
        volume_fraction = 0.4,
        max_iterations = 15,
        tolerance = 0.01,
        filter_radius = 2.5,
        move_limit = 0.1,
        damping = 0.5,
        solver_options = create_optimized_solver_options(conservative=false)
    )
    
    time_optimized = @elapsed begin
        results_optimized = simp_optimize(
            grid, dh, cellvalues,
            [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], 
            [ch_fixed], 
            optimized_params
        )
    end
    
    # Test 3: Conservative optimized settings
    println("\n⚡ Test 3: Conservative Optimized Settings")
    
    conservative_params = OptimizationParameters(
        E0 = E0,
        Emin = 1e-6,
        ν = ν,
        p = 3.0,
        volume_fraction = 0.4,
        max_iterations = 15,
        tolerance = 0.01,
        filter_radius = 2.5,
        move_limit = 0.1,
        damping = 0.5,
        solver_options = create_optimized_solver_options(conservative=true)
    )
    
    time_conservative = @elapsed begin
        results_conservative = simp_optimize(
            grid, dh, cellvalues,
            [(dh, collect(force_nodes), [0.0, -1.0, 0.0])], 
            [ch_fixed], 
            conservative_params
        )
    end
    
    # Performance comparison
    print_performance_summary(
        time_standard, results_standard,
        time_optimized, results_optimized,
        time_conservative, results_conservative
    )
    
    return results_standard, results_optimized, results_conservative
end

"""
    print_performance_summary(time_std, results_std, time_opt, results_opt, time_cons, results_cons)

Print detailed performance comparison.
"""
function print_performance_summary(time_std, results_std, time_opt, results_opt, time_cons, results_cons)
    println("\n" * "="^80)
    println("PERFORMANCE SUMMARY")
    println("="^80)
    
    # Timing comparison
    println("⏱️  EXECUTION TIME:")
    @printf("  Standard (Direct):      %.2f seconds\n", time_std)
    @printf("  Optimized (Aggressive): %.2f seconds (%.1fx speedup)\n", time_opt, time_std/time_opt)
    @printf("  Optimized (Conservative): %.2f seconds (%.1fx speedup)\n", time_cons, time_std/time_cons)
    
    # Solver statistics comparison
    println("\n🔧 SOLVER STATISTICS:")
    
    println("  Standard (Direct Solver Only):")
    @printf("    Iterations: %d\n", results_std.iterations)
    @printf("    Final Compliance: %.6f\n", results_std.compliance)
    @printf("    Converged: %s\n", results_std.converged ? "Yes" : "No")
    
    println("\n  Optimized (Aggressive):")
    @printf("    Iterations: %d\n", results_opt.iterations)
    @printf("    Final Compliance: %.6f\n", results_opt.compliance)
    @printf("    Total CG Iterations: %d\n", results_opt.total_cg_iterations)
    @printf("    Avg CG per Iteration: %.1f\n", results_opt.average_cg_per_iteration)
    @printf("    Converged: %s\n", results_opt.converged ? "Yes" : "No")
    
    println("\n  Optimized (Conservative):")
    @printf("    Iterations: %d\n", results_cons.iterations)
    @printf("    Final Compliance: %.6f\n", results_cons.compliance)
    @printf("    Total CG Iterations: %d\n", results_cons.total_cg_iterations)
    @printf("    Avg CG per Iteration: %.1f\n", results_cons.average_cg_per_iteration)
    @printf("    Converged: %s\n", results_cons.converged ? "Yes" : "No")
    
    # Solution quality comparison
    println("\n📊 SOLUTION QUALITY:")
    compliance_diff_aggressive = abs(results_opt.compliance - results_std.compliance) / results_std.compliance * 100
    compliance_diff_conservative = abs(results_cons.compliance - results_std.compliance) / results_std.compliance * 100
    
    @printf("  Compliance difference (Aggressive): %.3f%%\n", compliance_diff_aggressive)
    @printf("  Compliance difference (Conservative): %.3f%%\n", compliance_diff_conservative)
    
    if compliance_diff_aggressive < 1.0 && compliance_diff_conservative < 1.0
        println("  ✅ Excellent agreement between methods (<1% difference)")
    elseif compliance_diff_aggressive < 5.0 && compliance_diff_conservative < 5.0
        println("  ⚡ Good agreement between methods (<5% difference)")
    else
        println("  ⚠️  Significant differences detected - check solver settings")
    end
    
    # Strategy usage analysis
    println("\n🎯 SOLVER STRATEGY USAGE:")
    
    analyze_strategy_usage("Aggressive", results_opt.solver_strategy_history)
    analyze_strategy_usage("Conservative", results_cons.solver_strategy_history)
    
    # Recommendations
    println("\n💡 RECOMMENDATIONS:")
    
    if time_opt < time_std * 0.7
        println("  🚀 Aggressive settings provide significant speedup with good accuracy")
    end
    
    if results_opt.average_cg_per_iteration < 100
        println("  ✅ Warm-start solver is working efficiently")
    elseif results_opt.average_cg_per_iteration > 300
        println("  ⚠️  Consider more frequent preconditioner updates or conservative settings")
    end
    
    if compliance_diff_aggressive > 2.0
        println("  🎯 Use conservative settings for critical applications")
    else
        println("  ⚡ Aggressive settings recommended for most applications")
    end
    
    println("="^80)
end

"""
    analyze_strategy_usage(name, strategy_history)

Analyze and report solver strategy usage.
"""
function analyze_strategy_usage(name::String, strategy_history::Vector{Symbol})
    if isempty(strategy_history)
        return
    end
    
    strategy_counts = Dict{Symbol, Int}()
    for strategy in strategy_history
        strategy_counts[strategy] = get(strategy_counts, strategy, 0) + 1
    end
    
    println("  $name Strategy Distribution:")
    total = length(strategy_history)
    
    for (strategy, count) in sort(collect(strategy_counts))
        percentage = round(100 * count / total, digits=1)
        strategy_name = Dict(
            :direct_solve => "Direct",
            :warm_start_strict => "WS-Strict", 
            :warm_start_relaxed => "WS-Relaxed"
        )[strategy]
        @printf("    %s: %d/%d (%.1f%%)\n", strategy_name, count, total, percentage)
    end
end

"""
    run_scalability_test()

Test performance scaling with different mesh sizes.
"""
function run_scalability_test()
    println("\n🔬 SCALABILITY TEST")
    println("="^50)
    
    # This would require different mesh sizes
    # For now, simulate with different iteration counts
    
    test_cases = [
        (iterations=5, name="Quick Test"),
        (iterations=15, name="Standard"),
        (iterations=30, name="Extended")
    ]
    
    for (iterations, name) in test_cases
        println("\n📈 $name ($iterations iterations):")
        
        # Simulate performance characteristics
        # In reality, you would run actual optimizations
        estimated_direct_time = iterations * 0.8  # Estimated time per iteration
        estimated_optimized_time = iterations * 0.3  # With warm-start
        
        @printf("  Estimated Direct Time: %.1f seconds\n", estimated_direct_time)
        @printf("  Estimated Optimized Time: %.1f seconds\n", estimated_optimized_time)
        @printf("  Expected Speedup: %.1fx\n", estimated_direct_time / estimated_optimized_time)
    end
end

"""
    benchmark_individual_operations()

Benchmark individual operations to identify bottlenecks.
"""
function benchmark_individual_operations()
    println("\n🔍 DETAILED OPERATION BENCHMARKS")
    println("="^50)
    
    # This would benchmark individual operations
    println("Placeholder for detailed benchmarks:")
    println("  - Matrix assembly time")
    println("  - Solver time (direct vs iterative)")
    println("  - Sensitivity calculation time")
    println("  - Density filtering time")
    println("  - Memory allocation patterns")
    
    # Implementation would require instrumented code
end

# Example usage
println("EasySIMP Performance Testing Suite")
println("Run with: include(\"performance_test_example.jl\"); run_performance_comparison()")

# Uncomment to run automatically:
# run_performance_comparison()
# run_scalability_test()
# benchmark_individual_operations()
