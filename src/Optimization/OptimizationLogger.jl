# src/Optimization/OptimizationLogger.jl

using Printf
using Dates

export OptimizationLogger, log_iteration!, write_summary, close_logger

"""
    OptimizationLogger

Handles logging of optimization progress to CSV and summary files.
"""
mutable struct OptimizationLogger
    csv_file::IOStream
    task_name::String
    start_time::Float64
    export_path::String
    iteration_count::Int

    function OptimizationLogger(export_path::String, task_name::String)
        mkpath(export_path)

        # Create CSV file with header
        csv_path = joinpath(export_path, "optimization_progress.csv")
        csv_file = open(csv_path, "w")
        println(csv_file, "Iteration,Energy,VolumeFraction,MaxDensityChange")

        new(csv_file, task_name, time(), export_path, 0)
    end
end

"""
    log_iteration!(logger, iteration, energy, volume_fraction, max_change)

Log a single iteration to the CSV file.
"""
function log_iteration!(
    logger::OptimizationLogger,
    iteration::Int,
    energy::Float64,
    volume_fraction::Float64,
    max_change::Float64,
)
    @printf(
        logger.csv_file,
        "%d,%.6e,%.6f,%.6e\n",
        iteration,
        energy,
        volume_fraction,
        max_change
    )
    flush(logger.csv_file)
    logger.iteration_count = iteration
end

"""
    write_summary(logger, final_energy, final_volume, converged)

Write final summary file after optimization completes.
"""
function write_summary(
    logger::OptimizationLogger,
    final_energy::Float64,
    final_volume::Float64,
    converged::Bool,
)
    total_time = time() - logger.start_time

    summary_path = joinpath(logger.export_path, "optimization_summary.txt")
    open(summary_path, "w") do io
        println(io, "=" ^ 50)
        println(io, "SIMP TOPOLOGY OPTIMIZATION SUMMARY")
        println(io, "=" ^ 50)
        println(io)
        println(io, "Task name:           $(logger.task_name)")
        println(io, "Iterations:          $(logger.iteration_count)")
        println(io, "Total time:          $(round(total_time, digits=2)) s")
        println(io, "Converged:           $(converged ? "Yes" : "No")")
        println(io)
        println(io, "Final energy:    $(final_energy)")
        println(io, "Final volume:        $(final_volume)")
        println(io)
        println(io, "Generated:           $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "=" ^ 50)
    end

    println("Summary saved to: $summary_path")
end

"""
    close_logger(logger)

Close the CSV file stream.
"""
function close_logger(logger::OptimizationLogger)
    close(logger.csv_file)
end
