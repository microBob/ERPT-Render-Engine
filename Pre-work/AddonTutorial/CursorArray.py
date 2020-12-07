import bpy

bl_info = {
	"name": "Cursor Array",
	"blender": (2, 80, 0),
	"category": "Object",
}

addon_keymaps = []


class ObjectCursorArray(bpy.types.Operator):
	"""Object Cursor Array"""
	bl_idname = "object.cursor_array"
	bl_label = "Cursor Array"
	bl_options = {'REGISTER', 'UNDO'}

	# make "total" a Int property
	total: bpy.props.IntProperty(name="Count", default=2, min=1, max=100)

	def execute(self, ctx):
		scene = ctx.scene
		cursor_location = scene.cursor.location
		active_obj = ctx.active_object

		for i in range(self.total):
			obj_new = active_obj.copy()
			scene.collection.objects.link(obj_new)

			factor = i / self.total
			obj_new.location = (active_obj.location * factor) + (cursor_location * (1.0 - factor))

		return {'FINISHED'}


def add_cursor_array_btn(self, _):
	self.layout.operator(ObjectCursorArray.bl_idname, text="Cursor Array", icon='CURSOR')


def register():
	bpy.utils.register_class(ObjectCursorArray)
	bpy.types.VIEW3D_MT_mesh_add.append(add_cursor_array_btn)

	# Handle keymap
	wm = bpy.context.window_manager
	km = wm.keyconfigs.addon.keymaps.new(name="Object Mode", space_type='EMPTY')

	kmi = km.keymap_items.new(ObjectCursorArray.bl_idname, 'T', 'PRESS', ctrl=True, shift=True)
	kmi.properties.total = 4

	addon_keymaps.append((km, kmi))


def unregister():
	bpy.utils.unregister_class(ObjectCursorArray)
	bpy.types.VIEW3D_MT_mesh_add.remove(add_cursor_array_btn)

	for km, kmi in addon_keymaps:
		km.keymap_items.remove(kmi)
	addon_keymaps.clear()


if __name__ == "__main__":
	register()
