import bpy
import json
import mathutils
import os

# === 1. Путь к JSON ===
json_path = "C:/learning/LG/HPE/data/source/animation/frames_3d.json"

with open(json_path, "r") as f:
    frames_3d = json.load(f)

# === 2. Название арматуры ===
armature_name = "Armature"  # Замени при необходимости
armature = bpy.data.objects.get(armature_name)

if armature is None:
    raise Exception(f"Арматура с именем '{armature_name}' не найдена.")

bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

# === 3. Карта: кость -> индекс точки ===
# Подстрой под свою модель
bone_mapping = {
    "mixamorig:Head": 0,
    "mixamorig:LeftShoulder": 5,
    "mixamorig:RightShoulder": 6,
    "mixamorig:LeftForeArm": 7,
    "mixamorig:RightForeArm": 8,
    "mixamorig:LeftHand": 9,
    "mixamorig:RightHand": 10,
    "mixamorig:Hips": 11,
    "mixamorig:LeftUpLeg": 12,
    "mixamorig:RightUpLeg": 13,
    "mixamorig:LeftLeg": 14,
    "mixamorig:RightLeg": 15,
}

for i in range(len(frames_3d)):
    for j in range(len(frames_3d[i])):
        frames_3d[i][j][0] *= 1
        frames_3d[i][j][1] *= 1
        frames_3d[i][j][2] *= 1

        

# === 4. Проход по кадрам ===
for frame_idx, keypoints in enumerate(frames_3d, start=1):
    bpy.context.scene.frame_set(frame_idx)

    for bone_name, kp_index in bone_mapping.items():
        if bone_name not in armature.pose.bones:
            print(f"Кость '{bone_name}' не найдена в арматуре.")
            continue

        bone = armature.pose.bones[bone_name]
        pos = mathutils.Vector(keypoints[kp_index])

        # Расчёт направления относительно головы кости
        direction = (pos - bone.head).normalized()

        if direction.length > 0.0001:
            quat = direction.to_track_quat('-Y', 'Z')  # Можно менять ориентацию
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = quat
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
