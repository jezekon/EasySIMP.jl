using Gmsh
gmsh.initialize()
gmsh.open("Wheel_3d_coarse.geo")
gmsh.model.mesh.generate(3)
gmsh.write("Wheel_3d_coarse.msh")
gmsh.finalize()
