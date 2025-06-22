"""
WarmStartSolver.jl

Optimized solvers for SIMP topology optimization utilizing:
1. Warm-start iterative solvers with previous solution as initial guess
2. Adaptive tolerance based on density changes
3. Preconditioner reuse when matrix changes are small
4. Sparsity pattern reuse
"""

using Ferrite
using LinearAlgebra
using SparseArrays
using IterativeSolvers  # For CG solver
using Printf

export WarmStartSolver, SolverOptions, solve_with_warmstart!, 
       adaptive_solve_strategy, update_preconditioner!

"""
    SolverOptions

Configuration for optimized solver strategies.
"""
struct SolverOptions
    use_warm_start::Bool                    # Enable warm-start solvers
    preconditioner_update_freq::Int         # Update preconditioner every N iterations
    iterative_tolerance_adaptive::Bool      # Adaptive tolerance based on density changes
    incremental_assembly_threshold::Float64 # Threshold for incremental assembly
    relaxed_tolerance::Float64              # Tolerance for small changes
    strict_tolerance::Float64               # Tolerance for large changes
    max_cg_iterations::Int                  # Maximum CG iterations
    
    function SolverOptions(;
        use_warm_start = true,
        preconditioner_update_freq = 5,
        iterative_tolerance_adaptive = true,
        incremental_assembly_threshold = 0.005,
        relaxed_tolerance = 1e-3,
        strict_tolerance = 1e-6,
        max_cg_iterations = 1000
    )
        new(use_warm_start, preconditioner_update_freq, iterative_tolerance_adaptive,
            incremental_assembly_threshold, relaxed_tolerance, strict_tolerance, max_cg_iterations)
    end
end

"""
    WarmStartSolver

State container for warm-start iterative solvers.
"""
mutable struct WarmStartSolver
    last_solution::Vector{Float64}
    last_densities::Vector{Float64}
    preconditioner::Union{Nothing, Any}
    options::SolverOptions
    iteration_count::Int
    total_cg_iterations::Int
    preconditioner_updates::Int
    
    function WarmStartSolver(n_dofs::Int, n_elements::Int, options::SolverOptions = SolverOptions())
        new(
            zeros(n_dofs),           # last_solution
            ones(n_elements) * 0.5,  # last_densities (start with 50%)
            nothing,                 # preconditioner
            options,                 # options
            0,                       # iteration_count
            0,                       # total_cg_iterations
            0                        # preconditioner_updates
        )
    end
end

"""
    adaptive_solve_strategy(iteration, density_change, solver_options)

Determine the best solving strategy based on optimization state.
"""
function adaptive_solve_strategy(iteration::Int, density_change::Float64, solver_options::SolverOptions)
    if iteration < 3
        return :direct_solve  # Early iterations: use direct solver for stability
    elseif density_change < solver_options.incremental_assembly_threshold && iteration > 10
        return :warm_start_relaxed  # Late iterations with small changes
    elseif solver_options.use_warm_start
        return :warm_start_strict   # Middle iterations with warm start
    else
        return :direct_solve        # Fallback to direct solve
    end
end

"""
    update_preconditioner!(solver, K, force_update=false)

Update or reuse preconditioner based on matrix changes.
"""
function update_preconditioner!(solver::WarmStartSolver, K, force_update::Bool = false)
    if solver.preconditioner === nothing || force_update || 
       (solver.iteration_count % solver.options.preconditioner_update_freq == 0)
        
        # Simple diagonal preconditioner (fast to compute)
        # For better performance, could use ILU(0) or AMG
        solver.preconditioner = Diagonal(diag(K))
        solver.preconditioner_updates += 1
        
        # Ensure diagonal elements are not too small
        for i in 1:length(solver.preconditioner.diag)
            if abs(solver.preconditioner.diag[i]) < 1e-12
                solver.preconditioner.diag[i] = 1.0
            end
        end
        
        return true  # Preconditioner was updated
    end
    return false  # Preconditioner was reused
end

"""
    solve_with_warmstart!(solver, K, f, constraints, strategy=:warm_start_strict)

Solve linear system using warm-start strategy with adaptive tolerance.
"""
function solve_with_warmstart!(
    solver::WarmStartSolver, 
    K, 
    f, 
    constraints, 
    strategy::Symbol = :warm_start_strict
)
    solver.iteration_count += 1
    
    # Apply boundary conditions
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    if strategy == :direct_solve
        # Direct solve for early iterations or when requested
        u = K \ f
        solver.last_solution = copy(u)
        return u, 0  # Return solution and CG iterations (0 for direct)
    end
    
    # Determine tolerance based on strategy
    tolerance = if strategy == :warm_start_relaxed
        solver.options.relaxed_tolerance
    else
        solver.options.strict_tolerance
    end
    
    # Update preconditioner if needed
    prec_updated = update_preconditioner!(solver, K)
    
    # Initialize with previous solution (warm start)
    u = copy(solver.last_solution)
    
    # Ensure the initial guess satisfies boundary conditions
    for ch in constraints
        apply_zero!(u, ch)
    end
    
    # Solve using Conjugate Gradient with warm start
    try
        u_result, cg_info = cg!(
            u, K, f, 
            Pl = solver.options.use_warm_start ? solver.preconditioner : nothing,
            tol = tolerance,
            maxiter = solver.options.max_cg_iterations,
            log = true
        )
        
        cg_iterations = cg_info.iters
        solver.total_cg_iterations += cg_iterations
        
        # Check convergence
        if !cg_info.isconverged
            @warn "CG did not converge in $cg_iterations iterations (tolerance: $tolerance)"
            # Fallback to direct solve if CG fails
            println("  Falling back to direct solve...")
            u_result = K \ f
            cg_iterations = -1  # Indicate fallback
        end
        
        # Store solution for next warm start
        solver.last_solution = copy(u_result)
        
        return u_result, cg_iterations
        
    catch e
        @warn "CG solver failed: $e. Falling back to direct solve."
        u_direct = K \ f
        solver.last_solution = copy(u_direct)
        return u_direct, -1  # Indicate fallback
    end
end

"""
    solve_system_optimized!(solver, K, f, dh, cellvalues, material_model, 
                           old_densities, new_densities, constraints...)

Main optimized solve function that decides strategy and handles assembly optimization.
"""
function solve_system_optimized!(
    solver::WarmStartSolver,
    K, f,  # Pre-allocated matrices (will be modified)
    dh, cellvalues, material_model,
    old_densities::Vector{Float64},
    new_densities::Vector{Float64},
    constraints...
)
    # Calculate density change for strategy selection
    density_change = maximum(abs.(new_densities - old_densities))
    
    # Determine solving strategy
    strategy = adaptive_solve_strategy(solver.iteration_count + 1, density_change, solver.options)
    
    # Perform assembly (could be optimized for incremental assembly)
    # For now, use full assembly but reuse sparsity pattern
    fill!(K.nzval, 0.0)  # Zero out values but keep structure
    fill!(f, 0.0)
    
    # Assembly with new densities
    assembler = start_assemble(K, f)
    
    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        density = new_densities[cell_id]
        
        # Get material parameters for this density
        λ, μ = material_model(density)
        
        # Reinitialize cell values
        reinit!(cellvalues, cell)
        
        # Compute element stiffness matrix
        n_basefuncs = getnbasefunctions(cellvalues)
        ke = zeros(n_basefuncs, n_basefuncs)
        fe = zeros(n_basefuncs)
        
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                ∇Ni = shape_gradient(cellvalues, q_point, i)
                
                for j in 1:n_basefuncs
                    ∇Nj = shape_gradient(cellvalues, q_point, j)
                    εi = symmetric(∇Ni)
                    εj = symmetric(∇Nj)
                    
                    # Constitutive relation
                    σ = λ * tr(εj) * one(εj) + 2μ * εj
                    ke[i, j] += (εi ⊡ σ) * dΩ
                end
            end
        end
        
        # Assemble element contributions
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    
    # Solve the system with chosen strategy
    u, cg_iterations = solve_with_warmstart!(solver, K, f, constraints, strategy)
    
    # Update stored densities for next iteration
    solver.last_densities = copy(new_densities)
    
    # Print solver information
    if solver.iteration_count % 5 == 1 || cg_iterations > 0
        strategy_str = string(strategy)
        if cg_iterations > 0
            @printf("  Solver: %s (CG: %d iterations, tolerance: %.1e)\n", 
                   strategy_str, cg_iterations, 
                   strategy == :warm_start_relaxed ? solver.options.relaxed_tolerance : solver.options.strict_tolerance)
        elseif cg_iterations == -1
            @printf("  Solver: %s -> direct (CG fallback)\n", strategy_str)
        else
            @printf("  Solver: %s\n", strategy_str)
        end
    end
    
    return u, density_change, strategy
end

"""
    print_solver_statistics(solver)

Print accumulated solver performance statistics.
"""
function print_solver_statistics(solver::WarmStartSolver)
    avg_cg_per_iteration = solver.total_cg_iterations / max(1, solver.iteration_count)
    
    println("\n" * "="^60)
    println("SOLVER PERFORMANCE STATISTICS")
    println("="^60)
    println("Total iterations: $(solver.iteration_count)")
    println("Total CG iterations: $(solver.total_cg_iterations)")
    println("Average CG iterations per solve: $(round(avg_cg_per_iteration, digits=1))")
    println("Preconditioner updates: $(solver.preconditioner_updates)")
    println("Warm-start enabled: $(solver.options.use_warm_start)")
    println("Adaptive tolerance: $(solver.options.iterative_tolerance_adaptive)")
    println("="^60)
end

"""
    incremental_assembly_analysis(old_densities, new_densities, threshold=0.005)

Analyze which elements have significant density changes for potential incremental assembly.
"""
function incremental_assembly_analysis(old_densities::Vector{Float64}, new_densities::Vector{Float64}, threshold::Float64 = 0.005)
    changes = abs.(new_densities - old_densities)
    n_changed = count(x -> x > threshold, changes)
    max_change = maximum(changes)
    
    return (
        n_changed = n_changed,
        total_elements = length(old_densities),
        fraction_changed = n_changed / length(old_densities),
        max_change = max_change,
        assembly_strategy = n_changed < 0.1 * length(old_densities) ? :incremental : :full
    )
end
