# src/EasySIMP.jl
"""
EasySIMP.jl - SIMP Topology Optimization Package

A Julia package for Solid Isotropic Material with Penalization (SIMP)
topology optimization based on Ferrite.jl finite element framework.

Main features:
- SIMP material interpolation with OC method
- Density filtering for numerical stability
- Support for point loads, nodal traction, and surface traction
- VTU export for ParaView visualization
"""
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

# =============================================================================
# EXPORTS
# =============================================================================

# Mesh import
export import_mesh

# FEM setup and material
export setup_problem,
    create_material_model, create_simp_material_model, assemble_stiffness_matrix_simp!

# Boundary conditions
export apply_fixed_boundary!, apply_sliding_boundary!

# Force application
export apply_force!, apply_surface_traction!, get_boundary_facets

# Node selection
export select_nodes_by_plane,
    select_nodes_by_circle, select_nodes_by_cylinder, select_nodes_by_arc

# Solver
export solve_system

# Optimization
export OptimizationParameters, simp_optimize

# Load condition types
export AbstractLoadCondition, PointLoad, SurfaceTractionLoad

# Post-processing
export export_results_vtu, create_results_data, export_boundary_conditions

# Utilities
export calculate_volume, print_info, print_success, print_warning, print_error, print_data

end # module EasySIMP
