bl_info = {
    "name": "POC Engine",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "category": "Render Engine",
}

moduleNames = ["engine"]

import sys
import importlib
import bpy
from bpy.types import AddonPreferences
from bpy.props import StringProperty

# Import Modules
moduleFullNames = {}
for currentModuleName in moduleNames:
    moduleFullNames[currentModuleName] = ('{}.{}'.format(__name__, currentModuleName))

for currentModuleFullName in moduleFullNames.values():
    if currentModuleFullName in sys.modules:
        importlib.reload(sys.modules[currentModuleFullName])
    else:
        globals()[currentModuleFullName] = importlib.import_module(currentModuleFullName)
        setattr(globals()[currentModuleFullName], 'modulesNames', moduleFullNames)


# Run register and unregister functions
class EngineAddonPreferences(AddonPreferences):
    bl_idname = __package__

    engineExecutablePath: StringProperty(
        name="Engine Executable",
        description="Path to ERPT Engine executable. Downloaded from its GitHub",
        subtype="FILE_PATH"
    )

    # noinspection PyUnresolvedReferences
    def draw(self, _):
        layout = self.layout
        layout.label(text="ERPT Engine Preferences")
        layout.prop(self, "engineExecutablePath")


def register():
    for curModuleName in moduleFullNames.values():
        if curModuleName in sys.modules:
            if hasattr(sys.modules[curModuleName], 'register'):
                sys.modules[curModuleName].register()
    bpy.utils.register_class(EngineAddonPreferences)


def unregister():
    for curModuleName in moduleFullNames.values():
        if curModuleName in sys.modules:
            if hasattr(sys.modules[curModuleName], 'unregister'):
                sys.modules[curModuleName].unregister()
    bpy.utils.unregister_class(EngineAddonPreferences)


if __name__ == "__main__":
    register()
