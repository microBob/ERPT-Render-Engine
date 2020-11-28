import bpy


bl_info = {
	"name": "Move X Axis",
	"version": (1,0),
	"blender": (2, 80, 0),
	"category": "Object",
}


class ObjectMoveX(bpy.types.Operator):
	"""My Object Moving Script"""
	bl_idname = "object.move_x"
	bl_label = "Move X by One"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		scene = context.scene
		for obj in scene.objects:
			obj.location.x += 1.0

		return {'FINISHED'}


def add_move_btn(self, context):
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