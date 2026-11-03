"""
    select_nodes(grid::Grid, predicate::Function)

Selects nodes whose coordinates satisfy a condition. The `select_nodes_by_*`
functions below are special cases.

Parameters:
- `grid`: Computational mesh
- `predicate`: Function x -> Bool of the node coordinates x = [x, y, z]

Returns:
- Set of node IDs that satisfy the predicate

Example:
    # Strip y <= 1 of the end face x = 60
    nodes = select_nodes(grid, x -> abs(x[1] - 60.0) < 1e-6 && x[2] <= 1.0 + 1e-6)
"""
function select_nodes(grid::Grid, predicate::Function)
    selected_nodes = Set{Int}()

    for node_id = 1:getnnodes(grid)
        if predicate(grid.nodes[node_id].x)
            push!(selected_nodes, node_id)
        end
    end

    return selected_nodes
end

"""
    select_nodes_by_plane(grid::Grid, 
                          point::Vector{Float64}, 
                          normal::Vector{Float64}, 
                          tolerance::Float64=1e-4)

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

    # Distance from point to plane: d = (p - p0) · n
    return select_nodes(grid, coord -> abs(dot(coord - point, unit_normal)) < tolerance)
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
    # Normalize the normal vector
    unit_normal = normal / norm(normal)

    function in_circle(coord)
        v = coord - center

        # The node must lie on the plane of the circle
        if abs(dot(v, unit_normal)) >= tolerance
            return false
        end

        # Distance from center in the plane (projection of v onto the plane)
        projection = v - dot(v, unit_normal) * unit_normal
        return norm(projection) <= radius + tolerance
    end

    nodes_in_circle = select_nodes(grid, in_circle)

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

    function on_cylinder(coord)
        # Vector from axis point to node
        v = coord - axis_point

        # Project onto axis to find closest point on axis
        proj_length = dot(v, unit_axis)
        proj_point = axis_point + proj_length * unit_axis

        # Radial distance from axis
        radial_dist = norm(coord - proj_point)

        return abs(radial_dist - radius) < tolerance
    end

    selected_nodes = select_nodes(grid, on_cylinder)

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

    function on_arc(coord)
        v = coord - center

        # Radial vector (perpendicular to axis)
        radial_vec = v - dot(v, unit_axis) * unit_axis
        radial_dist = norm(radial_vec)

        # Check if on cylinder surface
        if abs(radial_dist - radius) > tolerance
            return false
        end

        # Calculate angle
        v_normalized = radial_vec / radial_dist
        angle_deg = rad2deg(atan(dot(v_normalized, ref_y), dot(v_normalized, ref_x)))
        if angle_deg < 0
            angle_deg += 360.0
        end

        # Check angle range
        if angle_start <= angle_end
            return angle_start <= angle_deg <= angle_end
        else
            return angle_deg >= angle_start || angle_deg <= angle_end
        end
    end

    selected_nodes = select_nodes(grid, on_arc)

    println(
        "Selected $(length(selected_nodes)) nodes on arc ($(angle_start)°-$(angle_end)°)",
    )
    return selected_nodes
end
