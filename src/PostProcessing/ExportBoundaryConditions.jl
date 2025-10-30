"""
    export_boundary_conditions(grid::Grid, dh::DofHandler, fixed_nodes::Set{Int}, 
                              force_nodes::Set{Int}, output_file::String)

Exports a mesh with marked boundary conditions to a VTU file.

Parameters:
- `grid`: Computational mesh
- `dh`: DofHandler from the problem setup
- `fixed_nodes`: Set of node IDs with fixed boundary conditions
- `force_nodes`: Set of node IDs with applied forces
- `output_file`: Path to the output file (without extension)
"""

function export_boundary_conditions(
    grid::Ferrite.Grid,
    dh::Ferrite.DofHandler,
    fixed_nodes::Set{Int},
    force_nodes::Set{Int},
    output_file::String,
)
    println("Exporting mesh with boundary conditions to $output_file...")

    # Initialize array for node data (0 = no BC, 1 = fixed, 2 = force)
    n_nodes = getnnodes(grid)
    bc_data = zeros(Int, n_nodes)

    # Mark nodes with boundary conditions
    for node_id in fixed_nodes
        bc_data[node_id] = 1  # For fixed boundary conditions
    end

    for node_id in force_nodes
        bc_data[node_id] = 2  # For forces
    end

    # Identify boundary faces
    boundary_faces = Tuple{Int,Vector{Int}}[]  # (BC_type, face_nodes)

    # Extract node coordinates in format compatible with WriteVTK
    # Convert Node objects to coordinates
    coords = zeros(3, n_nodes)
    for i = 1:n_nodes
        c = grid.nodes[i].x
        if length(c) == 3
            coords[:, i] = c
        else
            # Handle 2D case
            coords[1:length(c), i] = c
        end
    end

    # For each cell, identify faces where boundary conditions are applied
    for cell_idx = 1:getncells(grid)
        cell = getcells(grid, cell_idx)

        # Get all faces of the cell
        cell_faces = get_faces(cell)

        for face in cell_faces
            # Check if all nodes on the face have the same boundary condition type
            # and this type is not 0 (no boundary condition)
            face_bc_types = [bc_data[node_id] for node_id in face]

            # Skip if not all nodes have the same type, or if type is 0
            if length(unique(face_bc_types)) != 1 || face_bc_types[1] == 0
                continue
            end

            # Add the face and its boundary condition type to the list
            push!(boundary_faces, (face_bc_types[1], face))
        end
    end

    # Create output VTK file for faces with boundary conditions
    vtk_cells = WriteVTK.MeshCell[]
    bc_types = Int[]

    for (bc_type, face) in boundary_faces
        # Determine VTK cell type based on the number of nodes in the face
        if length(face) == 3
            cell_type = WriteVTK.VTKCellTypes.VTK_TRIANGLE
        elseif length(face) == 4
            cell_type = WriteVTK.VTKCellTypes.VTK_QUAD
        else
            # For other face types, additional options would need to be added
            continue
        end

        push!(vtk_cells, WriteVTK.MeshCell(cell_type, face))
        push!(bc_types, bc_type)
    end

    # Create VTK file with boundary elements
    vtk_grid(output_file, coords, vtk_cells) do vtk
        vtk["boundary_type"] = bc_types
    end

    println("Boundary conditions successfully exported to $(output_file).vtu")
end


# Helper function to get the faces of a cell
function get_faces(cell::Ferrite.Tetrahedron)
    return [
        [cell.nodes[1], cell.nodes[2], cell.nodes[3]], # face 1
        [cell.nodes[1], cell.nodes[2], cell.nodes[4]], # face 2
        [cell.nodes[2], cell.nodes[3], cell.nodes[4]], # face 3
        [cell.nodes[1], cell.nodes[3], cell.nodes[4]],  # face 4
    ]
end

function get_faces(cell::Ferrite.Hexahedron)
    return [
        [cell.nodes[1], cell.nodes[2], cell.nodes[3], cell.nodes[4]], # bottom face
        [cell.nodes[5], cell.nodes[6], cell.nodes[7], cell.nodes[8]], # top face
        [cell.nodes[1], cell.nodes[2], cell.nodes[6], cell.nodes[5]], # front face
        [cell.nodes[2], cell.nodes[3], cell.nodes[7], cell.nodes[6]], # right face
        [cell.nodes[3], cell.nodes[4], cell.nodes[8], cell.nodes[7]], # back face
        [cell.nodes[4], cell.nodes[1], cell.nodes[5], cell.nodes[8]],  # left face
    ]
end
