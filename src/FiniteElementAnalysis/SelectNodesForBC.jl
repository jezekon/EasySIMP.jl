"""
    select_nodes_by_plane(grid::Grid, 
                          point::Vector{Float64}, 
                          normal::Vector{Float64}, 
                          tolerance::Float64=1e-6)

Selects nodes that lie on a plane defined by a point and normal vector.

Parameters:
- `grid`: Computational mesh
- `point`: A point on the plane [x, y, z]
- `normal`: Normal vector to the plane [nx, ny, nz]
- `tolerance`: Distance tolerance for node selection

Returns:
- Set of node IDs that lie on the plane
"""
function select_nodes_by_plane(
    grid::Grid,
    point::Vector{Float64},
    normal::Vector{Float64},
    tolerance::Float64 = 1e-4,
)
    # Normalize the normal vector
    unit_normal = normal / norm(normal)

    # Extract number of nodes
    num_nodes = getnnodes(grid)
    selected_nodes = Set{Int}()

    # Check each node
    for node_id = 1:num_nodes
        coord = grid.nodes[node_id].x

        # Calculate distance from point to plane: d = (p - p0) · n
        dist = abs(dot(coord - point, unit_normal))

        # If distance is within tolerance, node is on plane
        if dist < tolerance
            push!(selected_nodes, node_id)
        end
    end

    # println("Selected $(length(selected_nodes)) nodes on the specified plane")
    return selected_nodes
end

"""
    select_nodes_by_circle(grid::Grid, 
                           center::Vector{Float64}, 
                           normal::Vector{Float64}, 
                           radius::Float64, 
                           tolerance::Float64=1e-6)

Selects nodes that lie on a circular region defined by center, normal and radius.

Parameters:
- `grid`: Computational mesh
- `center`: Center of the circle [x, y, z]
- `normal`: Normal vector to the plane containing the circle [nx, ny, nz]
- `radius`: Radius of the circle
- `tolerance`: Distance tolerance for node selection

Returns:
- Set of node IDs that lie on the circular region
"""
function select_nodes_by_circle(
    grid::Grid,
    center::Vector{Float64},
    normal::Vector{Float64},
    radius::Float64,
    tolerance::Float64 = 1e-6,
)
    # First, get nodes on the plane
    nodes_on_plane = select_nodes_by_plane(grid, center, normal, tolerance)

    # Normalize the normal vector
    unit_normal = normal / norm(normal)

    # Initialize set for nodes in circle
    nodes_in_circle = Set{Int}()

    # Check which nodes are within the circle radius
    for node_id in nodes_on_plane
        coord = grid.nodes[node_id].x

        # Project the vector from center to node onto the plane
        v = coord - center
        projection = v - dot(v, unit_normal) * unit_normal

        # Calculate distance from center in the plane
        dist = norm(projection)

        # If distance is less than radius, node is in the circle
        if dist <= radius + tolerance
            push!(nodes_in_circle, node_id)
        end
    end

    println("Selected $(length(nodes_in_circle)) nodes in the circular region")
    return nodes_in_circle
end

"""
    select_nodes_by_cylinder(grid::Grid, 
                             axis_point::Vector{Float64}, 
                             axis_direction::Vector{Float64}, 
                             radius::Float64, 
                             tolerance::Float64=1e-4)

Selects nodes that lie on a cylindrical surface.

Parameters:
- `grid`: Computational mesh
- `axis_point`: A point on the cylinder axis [x, y, z]
- `axis_direction`: Direction vector of the axis [dx, dy, dz]
- `radius`: Radius of the cylinder
- `tolerance`: Distance tolerance for node selection

Returns:
- Set of node IDs that lie on the cylindrical surface
"""
function select_nodes_by_cylinder(
    grid::Grid,
    axis_point::Vector{Float64},
    axis_direction::Vector{Float64},
    radius::Float64,
    tolerance::Float64 = 1e-4,
)
    unit_axis = axis_direction / norm(axis_direction)
    selected_nodes = Set{Int}()

    for node_id = 1:getnnodes(grid)
        coord = grid.nodes[node_id].x

        # Vector from axis point to node
        v = coord - axis_point

        # Project onto axis to find closest point on axis
        proj_length = dot(v, unit_axis)
        proj_point = axis_point + proj_length * unit_axis

        # Radial distance from axis
        radial_dist = norm(coord - proj_point)

        if abs(radial_dist - radius) < tolerance
            push!(selected_nodes, node_id)
        end
    end

    println("Selected $(length(selected_nodes)) nodes on cylinder (r = $radius)")
    return selected_nodes
end

"""
    select_nodes_by_arc(grid::Grid, 
                        center::Vector{Float64}, 
                        axis::Vector{Float64}, 
                        radius::Float64, 
                        angle_start::Float64, 
                        angle_end::Float64,
                        tolerance::Float64=1e-4)

Selects nodes on a cylindrical arc defined by axis, radius and angle range.
Angles are in DEGREES, measured counter-clockwise from reference direction.
"""
function select_nodes_by_arc(
    grid::Grid,
    center::Vector{Float64},
    axis::Vector{Float64},
    radius::Float64,
    angle_start::Float64,
    angle_end::Float64,
    tolerance::Float64 = 1e-4,
)
    unit_axis = axis / norm(axis)

    # Reference directions perpendicular to axis
    if abs(unit_axis[3]) > 0.9
        ref_x = [1.0, 0.0, 0.0] - dot([1.0, 0.0, 0.0], unit_axis) * unit_axis
    else
        ref_x = cross([0.0, 0.0, 1.0], unit_axis)
    end
    ref_x = ref_x / norm(ref_x)
    ref_y = cross(unit_axis, ref_x)

    selected_nodes = Set{Int}()

    for node_id = 1:getnnodes(grid)
        coord = grid.nodes[node_id].x
        v = coord - center

        # Radial vector (perpendicular to axis)
        radial_vec = v - dot(v, unit_axis) * unit_axis
        radial_dist = norm(radial_vec)

        # Check if on cylinder surface
        if abs(radial_dist - radius) > tolerance
            continue
        end

        # Calculate angle
        v_normalized = radial_vec / radial_dist
        angle_deg = rad2deg(atan(dot(v_normalized, ref_y), dot(v_normalized, ref_x)))
        if angle_deg < 0
            angle_deg += 360.0
        end

        # Check angle range
        in_range = if angle_start <= angle_end
            angle_start <= angle_deg <= angle_end
        else
            angle_deg >= angle_start || angle_deg <= angle_end
        end

        if in_range
            push!(selected_nodes, node_id)
        end
    end

    println(
        "Selected $(length(selected_nodes)) nodes on arc ($(angle_start)°-$(angle_end)°)",
    )
    return selected_nodes
end
