using ReadVTK
using WriteVTK

"""
    extract_mesh_from_vtu(input_file::String, output_file::String)

Načte VTU soubor, extrahuje z něj pouze síť a uloží ji jako čistý VTU soubor.

Parameters:
- `input_file`: Cesta k vstupnímu VTU souboru
- `output_file`: Název výstupního souboru (bez přípony .vtu)

Returns:
- `Bool`: true pokud úspěšné, jinak false
"""
function extract_mesh_from_vtu(input_file::String, output_file::String)
    try
        println("Reading VTU file: $input_file")

        # Read the VTU file using ReadVTK
        vtk_file = ReadVTK.VTKFile(input_file)

        # Extract coordinates from points
        points = ReadVTK.get_points(vtk_file)

        # Extract cells
        vtk_cells = ReadVTK.get_cells(vtk_file)
        connectivity = vtk_cells.connectivity
        offsets = vtk_cells.offsets
        types = vtk_cells.types

        println("Found $(size(points, 2)) points and $(length(types)) cells")

        # Create start offsets for cell connectivity indexing
        start_indices = vcat(1, offsets[1:(end-1)] .+ 1)

        # Convert VTK cells to WriteVTK format
        cells = WriteVTK.MeshCell[]

        for i = 1:length(types)
            # Get connectivity indices for this cell
            conn_indices = start_indices[i]:offsets[i]
            cell_conn = connectivity[conn_indices]
            vtk_type = types[i]

            # Convert VTK type to WriteVTK MeshCell
            if vtk_type == 10  # VTK_TETRA
                push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TETRA, cell_conn))
            elseif vtk_type == 12  # VTK_HEXAHEDRON
                push!(
                    cells,
                    WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_HEXAHEDRON, cell_conn),
                )
            elseif vtk_type == 5  # VTK_TRIANGLE
                push!(
                    cells,
                    WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, cell_conn),
                )
            elseif vtk_type == 9  # VTK_QUAD
                push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, cell_conn))
            elseif vtk_type == 3  # VTK_LINE
                push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, cell_conn))
            else
                println("Warning: Unsupported cell type $vtk_type, skipping")
            end
        end

        println("Converted $(length(cells)) cells")

        # Write clean VTU file with only mesh geometry
        vtk_grid(output_file, points, cells) do vtk
            # Only mesh geometry - no additional data
        end

        println("Clean mesh saved to $(output_file).vtu")
        return true

    catch e
        println("Error: $e")
        return false
    end
end

extract_mesh_from_vtu("cantilever_beam_opt.vtu", "clean_mesh")
