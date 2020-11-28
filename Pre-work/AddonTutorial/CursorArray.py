import bpy

bl_info = {
	"name": "Cursor Array",
	"blender": (2, 80, 0),
	"category": "Object",
}


class ObjectCursorArray(bpy.types.Operator):
	"""Object Cursor Array"""
	bl_idname = "object.cursor_array"
	bl_label = "Cursor Array"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, ctx):
		scene = ctx.scene
		cursor_location = scene.cursor.location
		active_obj = ctx.active_object

		total = 10

		for i in range(total):
			obj_new = active_obj.copy()
			scene.collection.objects.link(obj_new)

			factor = i / total
			obj_new.location = (active_obj.location * factor) + (cursor_location * (1.0 - factor))

		return {'FINISHED'}


def add_cursor_array_btn(self, _):
	self.layout.operator(ObjectCursorArray.bl_idname, text="Cursor Array", icon='CURSOR')


def register():
	bpy.utils.register_class(ObjectCursorArray)
	bpy.types.VIEW3D_MT_mesh_add.append(add_cursor_array_btn)


def unregister():
	bpy.utils.unregister_class(ObjectCursorArray)
	bpy.types.VIEW3D_MT_mesh_add.remove(add_cursor_array_btn)


if __name__ == "__main__":
	register()
