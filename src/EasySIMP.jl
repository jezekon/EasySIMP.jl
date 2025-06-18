module EasySIMP

# Package version
const VERSION = v"0.1.0"

# Core dependencies
using Ferrite
using WriteVTK
using LinearAlgebra
using SparseArrays

# Include all submodules
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

# Export main interface functions
export simp_topology_optimization, SIMPProblem
export cantilever_beam, bridge_structure, mbb_beam

# Export key functions from submodules for advanced users
export import_mesh                    # MeshImport
export setup_problem, create_material_model, apply_fixed_boundary!, 
       apply_sliding_boundary!, apply_force!, solve_system    # FiniteElementAnalysis
export OptimizationParameters, simp_optimize               # Optimization
export export_results_vtu, create_results_data           # PostProcessing
export calculate_volume, print_info, print_success, 
       print_warning, print_error, print_data              # Utils

# Include main interface
include("Interface.jl")

# Module information
function __init__()
    println("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                EasySIMP.jl                                    ║
    ║                   Simple SIMP Topology Optimization for Julia                 ║
    ║                                                                               ║
    ║  Built on Ferrite.jl for robust finite element analysis                      ║
    ║  Version: $(VERSION)                                                            ║
    ║                                                                               ║
    ║  Quick start:                                                                 ║
    ║    problem = cantilever_beam("mesh.vtu", 0.4)                                ║
    ║    results = simp_topology_optimization(problem)                             ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
end

"""
    EasySIMP

Simple SIMP (Solid Isotropic Material with Penalization) topology optimization 
package for Julia, built on Ferrite.jl.

## Features
- Import VTU meshes from ParaView/GMSH
- 3D SIMP topology optimization with OC solver
- Density filtering for mesh independence
- Point loads and body forces (gravity, acceleration)
- Fixed and sliding boundary conditions
- VTU output for ParaView visualization

## Basic Usage
```julia
using EasySIMP

# Define problem
problem = cantilever_beam("cantilever.vtu", 0.4)

# Run optimization  
results = simp_topology_optimization(problem)
```

## Advanced Usage
```julia
# Custom problem definition
problem = SIMPProblem(
    "structure.vtu",
    volume_fraction = 0.5,
    E0 = 200e9,  # Steel properties
    ν = 0.3,
    point_loads = [("load_nodes", [0, 0, -1000])],
    fixed_supports = ["fixed_nodes"],
    filter_radius = 1.5,
    max_iterations = 200
)

results = simp_topology_optimization(problem)
```

See documentation for full API reference.
"""
EasySIMP

end # module EasySIMP
