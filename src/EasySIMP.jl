# src/EasySIMP.jl

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

include("PostProcessing/PostProcessing.jl")
using .PostProcessing

include("Optimization/Optimization.jl")
using .Optimization


# Export core functionality
export import_mesh

export setup_problem,
    create_material_model,
    create_simp_material_model,
    assemble_stiffness_matrix_simp!,
    apply_fixed_boundary!,
    apply_sliding_boundary!,
    apply_force!,
    solve_system,
    select_nodes_by_plane,
    select_nodes_by_circle

export OptimizationParameters, simp_optimize

export export_results_vtu, create_results_data, export_boundary_conditions

export calculate_volume, print_info, print_success, print_warning, print_error, print_data

export select_nodes_by_cylinder, select_nodes_by_arc, apply_nodal_traction!

end # module EasySIMP
