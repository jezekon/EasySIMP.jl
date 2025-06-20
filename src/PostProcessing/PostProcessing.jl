module PostProcessing

using Ferrite
using WriteVTK
using ..FiniteElementAnalysis
using ..Utils

export export_results_vtu, ResultsData, create_results_data, export_boundary_conditions

include("ExportBoundaryConditions.jl")

"""
    ResultsData

Container for optimization results to be exported.
"""
struct ResultsData
    # Mesh and DOF information
    grid::Grid
    dh::DofHandler
    
    # Primary results
    densities::Vector{Float64}
    displacements::Vector{Float64}
    
    # Stress results  
    von_mises_stress::Vector{Float64}
    stress_tensors::Dict  # Full stress tensors per element
    
    # Optimization information
    compliance::Float64
    volume_fraction::Float64
    iterations::Int
    converged::Bool
    
    # History data
    compliance_history::Vector{Float64}
    volume_history::Vector{Float64}
end

"""
    create_results_data(grid, dh, optimization_result)

Create ResultsData from optimization results.
"""
function create_results_data(
    grid::Grid,
    dh::DofHandler, 
    opt_result  # OptimizationResult from Optimization module
)
    # Calculate von Mises stresses from stress tensors
    von_mises = calculate_von_mises_stresses(opt_result.stresses)
    
    return ResultsData(
        grid,
        dh,
        opt_result.densities,
        opt_result.displacements,
        von_mises,
        opt_result.stresses,
        opt_result.compliance,
        opt_result.volume / calculate_volume(grid),
        opt_result.iterations,
        opt_result.converged,
        opt_result.compliance_history,
        opt_result.volume_history
    )
end

"""
    export_results_vtu(results_data, filename_base; include_history=true)

Export optimization results to VTU files for ParaView visualization.

# Arguments
- `results_data`: ResultsData struct with all results
- `filename_base`: Base filename (without extension)
- `include_history`: Whether to export convergence history plots

# Output Files
- `{filename_base}_results.vtu`: Main results with density, displacement, stress
- `{filename_base}_history.vtu`: Convergence history (if requested)
"""
function export_results_vtu(
    results_data::ResultsData,
    filename_base::String;
    include_history::Bool = true
)
    print_info("Exporting results to VTU format...")
    
    # Export main results
    export_main_results(results_data, filename_base * "_results")
    
    # Export convergence history if requested
    if include_history && length(results_data.compliance_history) > 1
        export_convergence_history(results_data, filename_base * "_history")
    end
    
    print_success("VTU export completed")
    print_data("Main results: $(filename_base)_results.vtu")
    if include_history
        print_data("Convergence history: $(filename_base)_history.vtu")  
    end
end

"""
    export_main_results(results_data, filename)

Export main optimization results to VTU file.
"""
function export_main_results(results_data::ResultsData, filename::String)
    grid = results_data.grid
    dh = results_data.dh
    
    # Prepare points (node coordinates)
    n_nodes = getnnodes(grid)
    points = zeros(3, n_nodes)
    for (i, node) in enumerate(getnodes(grid))
        coord = node.x
        points[1, i] = coord[1]
        points[2, i] = length(coord) >= 2 ? coord[2] : 0.0
        points[3, i] = length(coord) >= 3 ? coord[3] : 0.0
    end
    
    # Prepare cells
    cells = create_vtk_cells(grid)
    
    # Create VTU file
    vtk_grid(filename, points, cells) do vtk
        # Cell data (element-based)
        vtk["density"] = results_data.densities
        vtk["von_mises_stress"] = results_data.von_mises_stress
        
        # Add element-wise compliance
        element_compliance = calculate_element_compliance(results_data)
        vtk["element_compliance"] = element_compliance
        
        # Point data (node-based)  
        nodal_displacements = extract_nodal_displacements(results_data)
        vtk["displacement"] = nodal_displacements
        
        # Add displacement magnitude
        displacement_magnitude = [norm(nodal_displacements[:, i]) for i in 1:size(nodal_displacements, 2)]
        vtk["displacement_magnitude"] = displacement_magnitude
        
        # Metadata
        vtk["compliance", VTKFieldData()] = results_data.compliance
        vtk["volume_fraction", VTKFieldData()] = results_data.volume_fraction
        vtk["iterations", VTKFieldData()] = results_data.iterations
        vtk["converged", VTKFieldData()] = results_data.converged ? 1 : 0
    end
end

"""
    export_convergence_history(results_data, filename)

Export convergence history as a simple line plot in VTU format.
"""
function export_convergence_history(results_data::ResultsData, filename::String)
    n_iter = length(results_data.compliance_history)
    
    if n_iter <= 1
        print_warning("Insufficient history data for convergence plot")
        return
    end
    
    # Create coordinate vectors for line plot
    x_coords = Float64[i for i in 1:n_iter]
    y_coords = zeros(Float64, n_iter)
    z_coords = zeros(Float64, n_iter)
    
    # Create line cells connecting consecutive points
    cells = WriteVTK.MeshCell[]
    for i = 1:(n_iter-1)
        push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, [i, i+1]))
    end
    
    # Create VTU file
    vtk_grid(filename, x_coords, y_coords, z_coords, cells) do vtk
        # Point data
        vtk["iteration"] = collect(1:n_iter)
        vtk["compliance"] = results_data.compliance_history
        vtk["volume_fraction"] = results_data.volume_history ./ calculate_volume(results_data.grid)
        
        # Normalized values for easier visualization
        if maximum(results_data.compliance_history) > 0
            vtk["compliance_normalized"] = results_data.compliance_history ./ maximum(results_data.compliance_history)
        end
    end
end


"""
    create_vtk_cells(grid)

Convert Ferrite grid cells to VTK format.
"""
function create_vtk_cells(grid::Grid)
    cells = WriteVTK.MeshCell[]  # Properly typed vector
    
    for cell in getcells(grid)
        if cell isa Ferrite.Tetrahedron
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TETRA, cell.nodes))
        elseif cell isa Ferrite.Hexahedron
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_HEXAHEDRON, cell.nodes))
        elseif cell isa Ferrite.Triangle
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, cell.nodes))
        elseif cell isa Ferrite.Quadrilateral
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, cell.nodes))
        else
            @warn "Unsupported cell type: $(typeof(cell))"
        end
    end
    
    return cells
end


"""
    extract_nodal_displacements(results_data)

Extract nodal displacements in proper format for VTU export.
"""
function extract_nodal_displacements(results_data::ResultsData)
    dh = results_data.dh
    u = results_data.displacements
    n_nodes = getnnodes(results_data.grid)
    
    # Assume 3D problem with 3 DOFs per node
    dofs_per_node = 3
    nodal_displacements = zeros(3, n_nodes)
    
    # Extract displacement components for each node
    for node = 1:n_nodes
        for dim = 1:dofs_per_node
            dof = (node - 1) * dofs_per_node + dim
            if dof <= length(u)
                nodal_displacements[dim, node] = u[dof]
            end
        end
    end
    
    return nodal_displacements
end

"""
    calculate_element_compliance(results_data)

Calculate compliance contribution from each element.
"""
function calculate_element_compliance(results_data::ResultsData)
    n_cells = getncells(results_data.grid)
    element_compliance = zeros(n_cells)
    
    # This would require element-wise compliance calculation
    # For now, return normalized density as proxy
    total_compliance = results_data.compliance
    total_volume = sum(results_data.densities)
    
    for i = 1:n_cells
        # Approximate element compliance based on density and stress
        element_compliance[i] = results_data.densities[i] * results_data.von_mises_stress[i] / total_volume
    end
    
    return element_compliance
end

"""
    calculate_von_mises_stresses(stress_tensors)

Calculate von Mises stress from stress tensor dictionary.
"""
function calculate_von_mises_stresses(stress_tensors::Dict)
    n_elements = length(stress_tensors)
    von_mises = zeros(n_elements)
    
    for (cell_id, stress_data) in stress_tensors
        if haskey(stress_tensors, cell_id) && !isempty(stress_data)
            # Take first quadrature point stress or average if multiple
            if stress_data isa Vector && !isempty(stress_data)
                σ = stress_data[1]  # First quadrature point
            else
                σ = stress_data
            end
            
            # Calculate von Mises stress: √(3/2 * (dev(σ) ⊡ dev(σ)))
            dev_stress = dev(σ)  # Deviatoric stress
            von_mises[cell_id] = sqrt(3/2 * (dev_stress ⊡ dev_stress))
        end
    end
    
    return von_mises
end

"""
    create_summary_report(results_data, filename)

Create a text summary report of optimization results.
"""
function create_summary_report(results_data::ResultsData, filename::String)
    open(filename * ".txt", "w") do io
        println(io, "=" ^ 60)
        println(io, "SIMP TOPOLOGY OPTIMIZATION RESULTS SUMMARY")
        println(io, "=" ^ 60)
        println(io)
        
        println(io, "OPTIMIZATION RESULTS:")
        println(io, "-" ^ 30)
        println(io, "Final Compliance: ", results_data.compliance)
        println(io, "Volume Fraction: ", results_data.volume_fraction)
        println(io, "Iterations: ", results_data.iterations)
        println(io, "Converged: ", results_data.converged ? "Yes" : "No")
        println(io)
        
        println(io, "MESH INFORMATION:")
        println(io, "-" ^ 30)
        println(io, "Number of Elements: ", getncells(results_data.grid))
        println(io, "Number of Nodes: ", getnnodes(results_data.grid))
        println(io, "Number of DOFs: ", ndofs(results_data.dh))
        println(io)
        
        println(io, "STRESS ANALYSIS:")
        println(io, "-" ^ 30)
        max_vm = maximum(results_data.von_mises_stress)
        min_vm = minimum(results_data.von_mises_stress)
        avg_vm = sum(results_data.von_mises_stress) / length(results_data.von_mises_stress)
        println(io, "Max von Mises Stress: ", max_vm)
        println(io, "Min von Mises Stress: ", min_vm)
        println(io, "Avg von Mises Stress: ", avg_vm)
        println(io)
        
        if length(results_data.compliance_history) > 1
            println(io, "CONVERGENCE HISTORY:")
            println(io, "-" ^ 30)
            println(io, "Initial Compliance: ", results_data.compliance_history[1])
            println(io, "Final Compliance: ", results_data.compliance_history[end])
            improvement = (results_data.compliance_history[1] - results_data.compliance_history[end]) / results_data.compliance_history[1] * 100
            println(io, "Improvement: ", round(improvement, digits=2), "%")
        end
    end
    
    print_success("Summary report created: $(filename).txt")
end

end # module
