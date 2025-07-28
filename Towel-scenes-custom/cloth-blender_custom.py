import bpy
import os
import random
import numpy as np
from mathutils import *
from random import sample
import json
import bpy_extras.object_utils

used_textures = set()
corner_indices = [0, 1, 2, 3]  # Initial 4 corners (flat cloth)

# Paths
table_texture_dir = "textures_table"
cloth_texture_dir = "textures"
image_output_dir = "images"
os.makedirs(image_output_dir, exist_ok=True)

# Texture lists
all_table_textures = [os.path.join(table_texture_dir, f) for f in os.listdir(table_texture_dir) if f.endswith(('.jpg', '.png'))]
all_textures = [f for f in os.listdir(cloth_texture_dir) if f.endswith(('.jpg', '.png'))]

table_texture_queue = []
texture_queue = []
dataset = {"images": []}

def clear_scene():
    """Delete all objects and unused data"""
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in collection:
            if block.users == 0:
                collection.remove(block)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def make_table(texture_queue):
    """Create a plane for table with random texture"""
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
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
    """Create the cloth plane and apply simulation modifiers"""
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
    """Randomize cloth position and pin some vertices"""
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

def reset_cloth(cloth):
    """Reset cloth to initial frame and origin"""
    cloth.modifiers["Cloth"].settings.vertex_group_mass = ''
    cloth.location = (0,0,0)
    bpy.context.scene.frame_set(0)

def pattern(obj, texture_filename, apply_tint=False):
    """Apply texture to cloth object, with optional tinting"""
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

def add_camera_light():
    """Add camera and point light looking top-down"""
    bpy.ops.object.light_add(type='POINT', location=(random.uniform(-3,3), random.uniform(-3,3), random.uniform(4,6)))
    light = bpy.context.object
    light.data.energy = random.uniform(100.0, 300.0)
    bpy.ops.object.camera_add(location=(0, 0, 8), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object

def track_corners(cloth, cam):
    """Track current positions of 4 original corners and project to 2D"""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    # Get deformed cloth at current frame
    cloth_eval = cloth.evaluated_get(depsgraph)
    mesh_eval = cloth_eval.to_mesh()

    coords_2d = []
    for idx in corner_indices:
        world = cloth.matrix_world @ mesh_eval.vertices[idx].co
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, world)
        x = int(co_2d.x * res_x)
        y = res_y - int(co_2d.y * res_y)  # flip Y axis
        coords_2d.append([x, y])

    cloth_eval.to_mesh_clear()
    return coords_2d

def render(filename, episode, cloth, cam):
    """Render frames and track 4 corners on each image"""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.exposure = 1.3
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'
    for frame in range(scene.frame_start, scene.frame_end):
        scene.frame_set(frame)
        if frame % 3 == 0:
            index = ((scene.frame_end - scene.frame_start) * episode + frame) // 3
            img_path = filename % index
            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)

            keypoints = track_corners(cloth, cam)
            dataset["images"].append({
                "file_name": os.path.basename(img_path),
                "keypoints": keypoints
            })

def render_dataset(num_episodes, filename):
    """Run multiple cloth simulations and record image keypoints"""
    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
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
        render(os.path.join(image_output_dir, "%06d_rgb.png"), episode, cloth, cam)

    with open(os.path.join(image_output_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)

if __name__ == '__main__':
    render_dataset(5, "images/%06d_rgb.png") # Adjust number of episodes as needed
    print("Rendering and annotation completed.")
