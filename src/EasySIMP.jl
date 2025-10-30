module EasySIMP

const VERSION = v"0.1.0"

# Core dependencies
using Ferrite
using WriteVTK
using LinearAlgebra
using SparseArrays

# Include core modules
include("Utils/Utils.jl")
using .Utils

include("MeshImport/MeshImport.jl")
using .MeshImport

include("FiniteElementAnalysis/FiniteElementAnalysis.jl")
using .FiniteElementAnalysis

include("Optimization/Optimization.jl")
using .Optimization

include("PostProcessing/PostProcessing.jl")
using .PostProcessing

# Export core functionality
export import_mesh
export setup_problem,
    create_material_model,
    apply_fixed_boundary!,
    apply_sliding_boundary!,
    apply_force!,
    solve_system
export OptimizationParameters, simp_optimize
export export_results_vtu, create_results_data
export calculate_volume, print_info, print_success, print_warning, print_error, print_data

end # module EasySIMP
