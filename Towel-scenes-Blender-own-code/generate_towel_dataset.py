import bpy
import random
import os
import math
from mathutils import Vector

# Output directory
output_dir = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/Towel-scenes-Blender-own-code/towel_output"
os.makedirs(output_dir, exist_ok=True)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge(do_recursive=True)

def random_color():
    return (random.random(), random.random(), random.random(), 1)

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])) ** 0.5

def add_displaced_towel(name, is_messy, size=(2.0, 1.0), location=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    towel = bpy.context.active_object
    towel.name = name
    towel.scale = (size[0] / 2.0, size[1] / 2.0, 1)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=100)
    bpy.ops.object.mode_set(mode='OBJECT')

    if is_messy:
        subsurf = towel.modifiers.new(name="Subsurf", type='SUBSURF')
        subsurf.levels = 2
        subsurf.render_levels = 2

    texture = bpy.data.textures.new(f"{name}_Tex", type='CLOUDS')
    texture.noise_scale = random.uniform(0.3, 0.6) if is_messy else random.uniform(0.8, 1.5)

    disp_mod = towel.modifiers.new(name="Displace", type='DISPLACE')
    disp_mod.texture = texture
    disp_mod.direction = 'Z'
    disp_mod.strength = random.uniform(0.15, 0.25) if is_messy else random.uniform(0.02, 0.03)

    return towel

def create_half_folded_towel():
    size = (2.0, 1.0)
    offset_z = 0.025
    shift_y = 0.06

    base = add_displaced_towel("TowelBase", is_messy=True, size=size, location=(0, 0, 0))
    folded = add_displaced_towel("TowelFolded", is_messy=True, size=size, location=(0, shift_y, offset_z))

    towel_color = random_color()
    mat = bpy.data.materials.new(name="TowelMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = towel_color
    base.data.materials.append(mat)
    folded.data.materials.append(mat)

    # Curved connector (aligned along Y axis)
    mid_y = shift_y / 2
    mid_z = offset_z / 2
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, mid_y, mid_z))
    curve = bpy.context.active_object
    curve.name = "TowelBridge"
    curve.scale = (0.05, size[0] / 2.0, 1)  # thin in X, long in Y
    curve.rotation_euler = (0, 0, math.radians(90))  # rotate to align along Y

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=20)
    bpy.ops.object.mode_set(mode='OBJECT')

    bend = curve.modifiers.new(name="Bend", type='SIMPLE_DEFORM')
    bend.deform_method = 'BEND'
    bend.deform_axis = 'Y'
    bend.angle = math.radians(160)

    curve.data.materials.append(mat)

    # Join all into one object
    bpy.ops.object.select_all(action='DESELECT')
    base.select_set(True)
    folded.select_set(True)
    curve.select_set(True)
    bpy.context.view_layer.objects.active = base
    bpy.ops.object.join()

    return base, towel_color

def create_towel(fold_type):
    if fold_type == "half_folded":
        return create_half_folded_towel()

    is_messy = fold_type == "messy"
    towel = add_displaced_towel("Towel", is_messy=is_messy, size=(1.0, 1.0), location=(0, 0, 0))

    towel_color = random_color()
    mat = bpy.data.materials.new(name="TowelMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = towel_color
    towel.data.materials.append(mat)

    return towel, towel_color

def get_min_z_with_modifiers(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    min_z = min((obj_eval.matrix_world @ v.co).z for v in mesh.vertices)
    obj_eval.to_mesh_clear()
    return min_z

def create_table(min_z, fold_type, towel_color):
    offset = 0.02 if fold_type == "flat" else 0.08
    table_z = min_z - offset

    bpy.ops.mesh.primitive_plane_add(size=2.5, location=(0, 0, table_z))
    table = bpy.context.active_object
    table.name = "Table"

    attempts = 0
    while True:
        table_color = random_color()
        if color_distance(towel_color, table_color) > 0.4 or attempts > 10:
            break
        attempts += 1

    mat = bpy.data.materials.new(name="TableMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = table_color
    table.data.materials.append(mat)

def add_light():
    bpy.ops.object.light_add(type='POINT', location=(
        random.uniform(-1.5, 1.5),
        random.uniform(-1.5, 1.5),
        random.uniform(2.0, 4.0)
    ))
    light = bpy.context.active_object
    light.data.energy = random.uniform(300, 800)

def add_camera(target):
    distance = random.uniform(1.8, 3.0)
    angle = random.uniform(0, 2 * math.pi)
    height = random.uniform(1.2, 2.2)

    cam_x = distance * math.cos(angle)
    cam_y = distance * math.sin(angle)
    cam_z = height

    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam

    direction = target.location - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# Fold types
fold_types = ["flat", "messy", "half_folded"]
random.shuffle(fold_types)
while len(fold_types) < 10:
    fold_types.append(random.choice(fold_types))

# Render towel scenes
for i in range(10):
    clear_scene()
    fold = fold_types[i]
    towel, towel_color = create_towel(fold)

    min_z = get_min_z_with_modifiers(towel)
    create_table(min_z, fold, towel_color)
    add_light()
    add_camera(towel)

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 768
    bpy.context.scene.render.filepath = os.path.join(output_dir, f"towel_scene_{i+1}.png")
    bpy.ops.render.render(write_still=True)
