bl_info = {
    "name": "POC Engine",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "category": "Render Engine",
}

moduleNames = ["engine"]

import sys
import importlib

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
def register():
    for curModuleName in moduleFullNames.values():
        if curModuleName in sys.modules:
            if hasattr(sys.modules[curModuleName], 'register'):
                sys.modules[curModuleName].register()


def unregister():
    for curModuleName in moduleFullNames.values():
        if curModuleName in sys.modules:
            if hasattr(sys.modules[curModuleName], 'unregister'):
                sys.modules[curModuleName].unregister()


if __name__ == "__main__":
    register()
