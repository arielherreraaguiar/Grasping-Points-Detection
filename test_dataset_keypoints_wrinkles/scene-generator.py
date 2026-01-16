import bpy
import os
import random
import numpy as np
import json
import bpy_extras.object_utils
from random import sample

# ================= CONFIGURATION =================
# Adjust this based on how many scenes you want
NUM_SCENES = 3           
# Folder configuration
BASE_OUTPUT_DIR = "images"
TABLE_TEXTURE_DIR = "textures_table"
CLOTH_TEXTURE_DIR = "textures"
# =================================================

# Global variables
used_textures = set()
corner_indices = [0, 1, 2, 3]

# Load texture paths
all_table_textures = [os.path.join(TABLE_TEXTURE_DIR, f) for f in os.listdir(TABLE_TEXTURE_DIR) if f.endswith(('.jpg', '.png'))]
all_textures = [f for f in os.listdir(CLOTH_TEXTURE_DIR) if f.endswith(('.jpg', '.png'))]

table_texture_queue = []
texture_queue = []

def clear_scene():
    """Clears the entire scene."""
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in collection:
            if block.users == 0:
                collection.remove(block)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def make_table(texture_queue):
    """Creates the table with texture (Original Code)."""
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

def make_cloth():
    """
    Creates the cloth with ORIGINAL physics from the uploaded file.
    Subdiv: 25, Levels: 3, Thickness: 0.1
    """
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=25)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.modifier_add(type='CLOTH')
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels = 3  # Original value
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness = 0.1 # Original value
    bpy.context.object.modifiers["Cloth"].collision_settings.use_self_collision = True
    return bpy.context.object

def generate_cloth_state(cloth):
    """
    Random state with ORIGINAL parameters.
    Pins: 1-3, Range: 0.7
    """
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
    cloth.modifiers["Cloth"].settings.vertex_group_mass = ''
    cloth.location = (0,0,0)
    bpy.context.scene.frame_set(0)

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

def add_camera_light():
    bpy.ops.object.light_add(type='POINT', location=(random.uniform(-3,3), random.uniform(-3,3), random.uniform(4,6)))
    bpy.context.object.data.energy = random.uniform(100.0, 300.0)
    bpy.ops.object.camera_add(location=(0, 0, 8), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object

def track_visible_corners(cloth, cam):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    cloth_eval = cloth.evaluated_get(depsgraph)
    mesh_eval = cloth_eval.to_mesh()

    keypoints = []
    # Blender corner order: 0, 1, 2, 3
    for idx in corner_indices:
        world = cloth.matrix_world @ mesh_eval.vertices[idx].co
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, world)
        x = int(co_2d.x * res_x)
        y = res_y - int(co_2d.y * res_y)
        
        # Basic visibility (inside frame) [x, y, v]
        # v=2 visible, v=0 not visible (simplified COCO style)
        visible = 2 if (0 <= co_2d.x <= 1 and 0 <= co_2d.y <= 1 and co_2d.z >= 0) else 0
        keypoints.append([x, y, visible])

    cloth_eval.to_mesh_clear()
    return keypoints

def render_scene_frames(scene_idx, cloth, cam):
    """
    Renders frames for ONE simulation and saves them in the corresponding Scene folder.
    """
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.exposure = 1.3
    
    # Prepare scene folders
    scene_dir = os.path.join(BASE_OUTPUT_DIR, f"scene{scene_idx}")
    raw_images_dir = os.path.join(scene_dir, "raw_images")
    os.makedirs(raw_images_dir, exist_ok=True)
    
    scene_gt = {}
    local_img_counter = 0

    # Iterate frames (same logic as original: frame % 3)
    for frame in range(scene.frame_start, scene.frame_end):
        scene.frame_set(frame)
        
        if frame % 3 == 0:
            # Filename: 000000_raw.png, etc.
            filename = f"{local_img_counter:06d}_raw.png"
            filepath = os.path.join(raw_images_dir, filename)
            
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
            
            # Save annotations for this frame
            kps = track_visible_corners(cloth, cam)
            scene_gt[filename] = kps
            
            local_img_counter += 1

    # Save Scene JSON
    json_path = os.path.join(scene_dir, "keypoints_gt.json")
    with open(json_path, "w") as f:
        json.dump(scene_gt, f, indent=4)
        
    print(f"Scene {scene_idx} finished. {local_img_counter} images saved.")

def run_pipeline():
    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    
    global table_texture_queue, texture_queue

    # Each iteration here is a complete SCENE (Simulation 1 -> Scene 1)
    for i in range(1, NUM_SCENES + 1):
        print(f"--- Generating Scene {i} ---")
        
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
        tex_path = os.path.join(CLOTH_TEXTURE_DIR, tex_file)
        apply_tint = tex_file in used_textures
        pattern(cloth, tex_path, apply_tint=apply_tint)
        used_textures.add(tex_file)

        reset_cloth(cloth)
        cloth = generate_cloth_state(cloth)
        
        # Render and save in sceneX folder
        render_scene_frames(i, cloth, cam)

    print("All scenes generated successfully.")

if __name__ == '__main__':
    run_pipeline()