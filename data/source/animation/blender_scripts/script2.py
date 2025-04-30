import bpy
import mathutils
import json

# === Параметры ===
armature_name = "Armature"  # Название объекта арматуры
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

print("Run script!!!")

# === Загрузка ключевых точек ===
json_path = "C:/learning/LG/HPE/data/source/animation/frames_3d.json"
with open(json_path, "r") as f:
    frames_3d = json.load(f)

# Преобразуем в [[(x, y, z)], ...]
keypoints = [
    [tuple(coord) for coord in frame] for frame in frames_3d
]

# Масштабируем при необходимости
v = 50
for i in range(len(keypoints)):
    keypoints[i] = [(x * v, y * v, z * v) for (x, y, z) in keypoints[i]]

# === Основной процесс ===
armature = bpy.data.objects[armature_name]
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

for frame_idx, frame_keypoints in enumerate(keypoints, start=1):
    bpy.context.scene.frame_set(frame_idx)

    for bone_name, kp_index in bone_mapping.items():
        if kp_index >= len(frame_keypoints):
            print(f"⚠️ Пропущена точка {kp_index} в кадре {frame_idx}")
            continue
        if bone_name not in armature.pose.bones:
            print(f"Кость {bone_name} не найдена")
            continue
        bone = armature.pose.bones[bone_name]
        target_pos = frame_keypoints[kp_index]

        # Преобразуем координаты target_pos в объект Vector
        target_vec = mathutils.Vector(target_pos)

        # Вычисляем направление от родительской кости к целевой точке
        if bone.parent:
            parent_head = bone.parent.head
        else:
            parent_head = mathutils.Vector((0, 0, 0))

        # Вычисляем разницу между позицией кости и целевой точкой
        direction = target_vec - parent_head
        direction.normalize()  # Нормализуем вектор для правильной ориентации

        # Ось кости в rest-позе (обычно вдоль Y)
        rest_axis = mathutils.Vector((0, 1, 0))

        # Рассчитываем вращение, которое повернет rest_axis в direction
        rotation = rest_axis.rotation_difference(direction)

        # Применяем вращение
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = rotation
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

bpy.ops.object.mode_set(mode='OBJECT')
