# import sys
# import os
import bpy

# sys.path.insert(1, "./bin")
from .bin import PythonAndC as pac


class ObjectMoveX(bpy.types.Operator):
    """My Object Moving Script"""
    bl_idname = "object.move_x"
    bl_label = "Move X by One"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        for obj in scene.objects:
            obj.location.x += 1.0

        print(pac.add(6, 8))

        return {'FINISHED'}


def add_move_btn(self, _):
    self.layout.operator(ObjectMoveX.bl_idname, text="Move X by One", icon='PLUGIN')


def register():
    print("Registering addon")
    bpy.utils.register_class(ObjectMoveX)
    bpy.types.VIEW3D_MT_transform_object.append(add_move_btn)


def unregister():
    bpy.utils.unregister_class(ObjectMoveX)
    bpy.types.VIEW3D_MT_transform_object.remove(add_move_btn)


if __name__ == "__main__":
    register()
