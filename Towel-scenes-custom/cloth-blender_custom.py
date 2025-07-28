import bpy
import os
import random
import numpy as np
from random import sample
from mathutils import *

# Keep track of cloth textures already used
used_textures = set()

# Define the directory where table textures are located
table_texture_dir = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/Towel-scenes-custom/textures_table"
all_table_textures = [os.path.join(table_texture_dir, f) for f in os.listdir(table_texture_dir) if f.endswith(('.jpg', '.png'))]
table_texture_queue = []

def clear_scene():
    """Remove all objects and unused data blocks from the scene"""
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def make_table(texture_queue):
    """Create a table surface with a texture selected randomly from a non-repeating queue"""
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0,0,0))
    table = bpy.context.object
    mat = bpy.data.materials.new(name="TableTexture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    tex_image = nodes.new("ShaderNodeTexImage")

    tex_file = texture_queue.pop(0)
    tex_image.image = bpy.data.images.load(tex_file)

    links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
    table.data.materials.append(mat)
    bpy.ops.object.modifier_add(type='COLLISION')
    return table

def make_cloth():
    """Create the cloth mesh and apply necessary physics modifiers"""
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

def generate_cloth_state(cloth):
    """Randomly move the cloth and pin a few vertices to simulate variation"""
    dx = np.random.uniform(0,0.7)*random.choice((-1,1))
    dy = np.random.uniform(0,0.7)*random.choice((-1,1))
    dz = np.random.uniform(0.4,0.8)
    cloth.location = (dx, dy, dz)
    cloth.rotation_euler = (0, 0, random.uniform(0, np.pi))
    if 'Pinned' in cloth.vertex_groups:
        cloth.vertex_groups.remove(cloth.vertex_groups['Pinned'])
    pinned_group = cloth.vertex_groups.new(name='Pinned')
    n = random.choice(range(1,4))
    subsample = sample(range(len(cloth.data.vertices)), n)
    pinned_group.add(subsample, 1.0, 'ADD')
    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'
    bpy.context.scene.frame_start = 0 
    bpy.context.scene.frame_end = 30
    return cloth

def reset_cloth(cloth):
    """Reset the cloth location and unpin any pinned vertices"""
    cloth.modifiers["Cloth"].settings.vertex_group_mass = ''
    cloth.location = (0,0,0)
    bpy.context.scene.frame_set(0)

def pattern(obj, texture_filename, apply_tint=False):
    """Apply a texture to the cloth with optional color tint if reused"""
    mat = bpy.data.materials.new(name="ImageTexture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes["Principled BSDF"]
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_filename)

    if apply_tint:
        mix_rgb = nodes.new(type='ShaderNodeMixRGB')
        mix_rgb.blend_type = 'MULTIPLY'
        mix_rgb.inputs['Fac'].default_value = 1.0
        mix_rgb.inputs['Color2'].default_value = (random.random(), random.random(), random.random(), 1)
        links.new(tex_image.outputs['Color'], mix_rgb.inputs['Color1'])
        links.new(mix_rgb.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])

    obj.data.materials.append(mat)

def add_camera_light():
    """Add a POINT light source with random position and intensity, and add a top-down camera"""
    bpy.ops.object.light_add(type='POINT', location=(
        random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(4, 6)))
    light = bpy.context.object
    light.data.energy = random.uniform(100.0, 300.0)

    bpy.ops.object.camera_add(location=(0, 0, 8), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object

def render(filename, episode):
    """Render 10 images per episode (sampled every 3 frames from 30-frame sim)"""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.filepath = "./images/{}".format(filename)
    scene.view_settings.exposure = 1.3
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'
    for frame in range(0, scene.frame_end):
        if frame % 3 == 0:
            index = ((scene.frame_end - scene.frame_start) * episode + frame) // 3
            scene.render.filepath = filename % index
            bpy.ops.render.render(write_still=True)
        scene.frame_set(frame)

def render_dataset(num_episodes, filename, render_width=640, render_height=480):
    """Main loop that orchestrates generation of cloth and table textures for each episode"""
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height

    all_textures = [f for f in os.listdir('textures') if f.endswith(('.jpg', '.png'))]
    texture_queue = []

    global table_texture_queue
    table_texture_queue = []

    for episode in range(num_episodes):
        global iteration
        iteration = episode
        clear_scene()
        add_camera_light()

        if not table_texture_queue:
            table_texture_queue = random.sample(all_table_textures, len(all_table_textures))
        make_table(table_texture_queue)

        cloth = make_cloth()

        if not texture_queue:
            texture_queue = random.sample(all_textures, len(all_textures))
        tex_file = texture_queue.pop(0)
        full_path = os.path.join('textures', tex_file)
        apply_tint = tex_file in used_textures
        pattern(cloth, full_path, apply_tint=apply_tint)
        used_textures.add(tex_file)

        reset_cloth(cloth)
        cloth = generate_cloth_state(cloth)
        render(filename, episode)

if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs('./images')
    else:
        os.system('rm -r ./images')
        os.makedirs('./images')
    filename = "images/%06d_rgb.png"
    render_dataset(500, filename) # 500 episodes × 10 images = 5000 total
    print("Rendering complete. Check the 'images' directory for output.")
