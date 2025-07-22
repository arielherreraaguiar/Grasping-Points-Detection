import bpy
import random
import os

# Delete all existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create a plane to represent the towel
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
towel = bpy.context.active_object
towel.name = "Towel"

# Subdivide the plane to allow realistic folding
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=20)
bpy.ops.object.mode_set(mode='OBJECT')

# Add a displacement modifier to simulate folds
mod = towel.modifiers.new(name='Displace', type='DISPLACE')
mod.strength = 0.05
texture = bpy.data.textures.new('TowelTexture', type='CLOUDS')
mod.texture = texture

# Randomly rotate the towel
towel.rotation_euler[2] = random.uniform(0, 3.14)

# Add a camera and set it as the active camera
bpy.ops.object.camera_add(location=(1.5, -1.5, 1.5), rotation=(1.1, 0, 0.8))
bpy.context.scene.camera = bpy.context.active_object

# Add a sunlight source
bpy.ops.object.light_add(type='SUN', location=(5, -5, 5))

# Render the scene and save the image in the current directory
current_dir = os.path.dirname(bpy.data.filepath)
if not current_dir:
    current_dir = os.getcwd()
bpy.context.scene.render.filepath = os.path.join(current_dir, "towel_render.png")
bpy.ops.render.render(write_still=True)

# Blender will remain open after execution if launched without --background

