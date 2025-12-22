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
    grid::Grid
    dh::DofHandler
    densities::Vector{Float64}
    displacements::Vector{Float64}
    von_mises_stress::Vector{Float64}
    stress_tensors::Dict
    compliance::Float64
    volume_fraction::Float64
    iterations::Int
    converged::Bool
    compliance_history::Vector{Float64}
    volume_history::Vector{Float64}
end

"""
    create_results_data(grid, dh, optimization_result)

Create ResultsData from optimization results.
"""
function create_results_data(grid::Grid, dh::DofHandler, opt_result)
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
        opt_result.volume_history,
    )
end

"""
    export_results_vtu(results_data, filename_base; include_history=false)

Export optimization results to VTU file for ParaView visualization.
History export is disabled by default.
"""
function export_results_vtu(
    results_data::ResultsData,
    filename_base::String;
    include_history::Bool = false,
)
    print_info("Exporting results to VTU format...")
    export_main_results(results_data, filename_base * "_results")
    print_success("VTU export completed: $(filename_base)_results.vtu")
end

"""
    export_main_results(results_data, filename)

Export main optimization results to VTU file.
"""
function export_main_results(results_data::ResultsData, filename::String)
    grid = results_data.grid
    dh = results_data.dh

    n_nodes = getnnodes(grid)
    points = zeros(3, n_nodes)
    for (i, node) in enumerate(getnodes(grid))
        coord = node.x
        points[1, i] = coord[1]
        points[2, i] = length(coord) >= 2 ? coord[2] : 0.0
        points[3, i] = length(coord) >= 3 ? coord[3] : 0.0
    end

    cells = create_vtk_cells(grid)

    vtk_grid(filename, points, cells) do vtk
        vtk["density"] = results_data.densities
        vtk["von_mises_stress"] = results_data.von_mises_stress

        element_compliance = calculate_element_compliance(results_data)
        vtk["element_compliance"] = element_compliance

        nodal_displacements = extract_nodal_displacements(results_data)
        vtk["displacement"] = nodal_displacements

        displacement_magnitude =
            [norm(nodal_displacements[:, i]) for i = 1:size(nodal_displacements, 2)]
        vtk["displacement_magnitude"] = displacement_magnitude

        vtk["compliance", VTKFieldData()] = results_data.compliance
        vtk["volume_fraction", VTKFieldData()] = results_data.volume_fraction
        vtk["iterations", VTKFieldData()] = results_data.iterations
        vtk["converged", VTKFieldData()] = results_data.converged ? 1 : 0
    end
end

"""
    create_vtk_cells(grid)

Convert Ferrite grid cells to VTK format.
"""
function create_vtk_cells(grid::Grid)
    cells = WriteVTK.MeshCell[]

    for cell in getcells(grid)
        if cell isa Ferrite.Tetrahedron
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TETRA, cell.nodes))
        elseif cell isa Ferrite.Hexahedron
            push!(
                cells,
                WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_HEXAHEDRON, cell.nodes),
            )
        elseif cell isa Ferrite.Triangle
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, cell.nodes))
        elseif cell isa Ferrite.Quadrilateral
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, cell.nodes))
        end
    end

    return cells
end

"""
    extract_nodal_displacements(results_data)

Extract nodal displacements for VTU export.
"""
function extract_nodal_displacements(results_data::ResultsData)
    dh = results_data.dh
    u = results_data.displacements
    n_nodes = getnnodes(results_data.grid)

    dofs_per_node = 3
    nodal_displacements = zeros(3, n_nodes)

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

    total_volume = sum(results_data.densities)

    for i = 1:n_cells
        element_compliance[i] =
            results_data.densities[i] * results_data.von_mises_stress[i] / total_volume
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
            if stress_data isa Vector && !isempty(stress_data)
                σ = stress_data[1]
            else
                σ = stress_data
            end

            dev_stress = dev(σ)
            von_mises[cell_id] = sqrt(3/2 * (dev_stress ⊡ dev_stress))
        end
    end

    return von_mises
end

end # module
