import bpy
import os
import random
import numpy as np
from mathutils import *
from random import sample

# Keep track of used textures
used_textures = set()

# Paths
table_texture_dir = "textures_table"
cloth_texture_dir = "textures"
output_dir = "wrinkle-training"
os.makedirs(output_dir, exist_ok=True)

# Load texture files
all_table_textures = [os.path.join(table_texture_dir, f) for f in os.listdir(table_texture_dir) if f.endswith(('.jpg', '.png'))]
all_textures = [f for f in os.listdir(cloth_texture_dir) if f.endswith(('.jpg', '.png'))]

# Clear the scene
def clear_scene():
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in collection:
            if block.users == 0:
                collection.remove(block)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# Create the table plane and apply texture
def make_table(texture_queue):
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    table = bpy.context.object
    mat = bpy.data.materials.new(name="TableTexture")
    mat.use_nodes = True
    tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_file = texture_queue.pop(0)
    tex_image.image = bpy.data.images.load(tex_file)
    mat.node_tree.links.new(tex_image.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    table.data.materials.append(mat)
    bpy.ops.object.modifier_add(type='COLLISION')
    return table

# Create the cloth plane and apply modifiers
def make_cloth():
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=25)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.modifier_add(type='CLOTH')
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels = 3
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness = 0.1
    bpy.context.object.modifiers["Cloth"].collision_settings.use_self_collision = True
    return bpy.context.object

# Random cloth location and pin some vertices
def generate_cloth_state(cloth):
    dx, dy = [np.random.uniform(0,0.7)*random.choice([-1,1]) for _ in range(2)]
    dz = np.random.uniform(0.4, 0.8)
    cloth.location = (dx, dy, dz)
    cloth.rotation_euler = (0, 0, random.uniform(0, np.pi))
    if 'Pinned' in cloth.vertex_groups:
        cloth.vertex_groups.remove(cloth.vertex_groups['Pinned'])
    group = cloth.vertex_groups.new(name='Pinned')
    pins = sample(range(len(cloth.data.vertices)), random.randint(1, 3))
    group.add(pins, 1.0, 'ADD')
    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'
    bpy.context.scene.frame_start = 0 
    bpy.context.scene.frame_end = 30
    return cloth

# Reset cloth location
def reset_cloth(cloth):
    cloth.modifiers["Cloth"].settings.vertex_group_mass = ''
    cloth.location = (0,0,0)
    bpy.context.scene.frame_set(0)

# Apply random texture and optional color tint
def pattern(obj, texture_filename, apply_tint=False):
    mat = bpy.data.materials.new(name="ImageTexture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_filename)
    if apply_tint:
        mix = nodes.new(type='ShaderNodeMixRGB')
        mix.blend_type = 'MULTIPLY'
        mix.inputs['Fac'].default_value = 1.0
        mix.inputs['Color2'].default_value = (random.random(), random.random(), random.random(), 1)
        links.new(tex_image.outputs['Color'], mix.inputs['Color1'])
        links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    obj.data.materials.append(mat)

# Add camera and light
def add_camera_light():
    bpy.ops.object.light_add(type='POINT', location=(random.uniform(-3,3), random.uniform(-3,3), random.uniform(4,6)))
    bpy.context.object.data.energy = random.uniform(100.0, 300.0)
    bpy.ops.object.camera_add(location=(0, 0, 8), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object

# Render each frame but save only the 10th image per episode
def render(filename, episode, cloth, cam):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.exposure = 1.3

    total_frames = scene.frame_end - scene.frame_start
    save_index = 9  # Save only the 10th image
    counter = 0

    for frame in range(scene.frame_start, scene.frame_end):
        scene.frame_set(frame)
        if frame % 3 == 0:
            if counter == save_index:
                index = ((total_frames) * episode + frame) // 3
                img_path = filename % index
                scene.render.filepath = img_path
                bpy.ops.render.render(write_still=True)
            counter += 1

# Run simulation episodes
def render_dataset(num_episodes, filename):
    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100

    global table_texture_queue, texture_queue
    table_texture_queue = []
    texture_queue = []

    for episode in range(num_episodes):
        clear_scene()
        add_camera_light()
        cam = bpy.context.object

        if not table_texture_queue:
            table_texture_queue = random.sample(all_table_textures, len(all_table_textures))
        make_table(table_texture_queue)

        cloth = make_cloth()

        if not texture_queue:
            texture_queue = random.sample(all_textures, len(all_textures))
        tex_file = texture_queue.pop(0)
        tex_path = os.path.join(cloth_texture_dir, tex_file)
        apply_tint = tex_file in used_textures
        pattern(cloth, tex_path, apply_tint=apply_tint)
        used_textures.add(tex_file)

        reset_cloth(cloth)
        cloth = generate_cloth_state(cloth)
        render(os.path.join(output_dir, "%06d_rgb.png"), episode, cloth, cam)

    print("Rendering completed. Images saved in wrinkle-training/")

# Main function
if __name__ == '__main__':
    render_dataset(200, "wrinkle-training/%06d_rgb.png")
