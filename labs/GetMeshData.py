from bpy import context as ctx
from bpy import data as data

dpgraph = ctx.evaluated_depsgraph_get()
print("=============")
for ob in data.objects:
    print(ob.name)
    obj_eval = ob.evaluated_get(dpgraph)
    try:
        mesh = obj_eval.to_mesh()
        verts = mesh.vertices
        print(verts.values()[0].co)
        poly = mesh.polygons[0]
        print(list(poly.vertices))
        print(poly.normal)
        obj_eval.to_mesh_clear()
    except:
        print("No mesh data")
    print()