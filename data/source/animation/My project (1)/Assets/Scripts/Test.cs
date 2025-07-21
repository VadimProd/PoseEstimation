using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;


public class Pos3D : MonoBehaviour {
	public String pos_filename;
	public int start_frame;
	public String end_frame;

	private List<Vector3[]> pos;
	private Transform hips;
	private Vector3 headUpVector;
	private JointPoint[] jointPoints;
    [SerializeField] private Transform characterPlacement;
	
	Animator anim;
	int s_frame;
	int e_frame;
	float play_time;
	int bone_num = 21;
	
	void InitHumanoidPose() {
		jointPoints = new JointPoint[bone_num];
		for (var i = 0; i < jointPoints.Length; i++) {
            jointPoints[i] = new JointPoint();
            //jointPoints[i].LastPoses = new Vector3[lowPassFilterChannels];
        }

		// Right Arm
		jointPoints[(int) BodyPoints.RightShoulder].Transform = anim.GetBoneTransform(HumanBodyBones.RightUpperArm); 
		jointPoints[(int) BodyPoints.RightElbow].Transform = anim.GetBoneTransform(HumanBodyBones.RightLowerArm);
		jointPoints[(int) BodyPoints.RightWrist].Transform = anim.GetBoneTransform(HumanBodyBones.RightHand);

		// Left Arm
        jointPoints[(int) BodyPoints.LeftShoulder].Transform = anim.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        jointPoints[(int) BodyPoints.LeftElbow].Transform = anim.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        jointPoints[(int) BodyPoints.LeftWrist].Transform = anim.GetBoneTransform(HumanBodyBones.LeftHand);

        // Right Leg
        jointPoints[(int) BodyPoints.RightHip].Transform = anim.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        jointPoints[(int) BodyPoints.RightKnee].Transform = anim.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        jointPoints[(int) BodyPoints.RightAnkle].Transform = anim.GetBoneTransform(HumanBodyBones.RightFoot);

        // Left Leg
        jointPoints[(int) BodyPoints.LeftHip].Transform = anim.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        jointPoints[(int) BodyPoints.LeftKnee].Transform = anim.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        jointPoints[(int) BodyPoints.LeftAnkle].Transform = anim.GetBoneTransform(HumanBodyBones.LeftFoot);

		// etc
        jointPoints[(int) BodyPoints.Hips].Transform = anim.GetBoneTransform(HumanBodyBones.Hips);
        jointPoints[(int) BodyPoints.Head].Transform = anim.GetBoneTransform(HumanBodyBones.Head);
        jointPoints[(int) BodyPoints.Neck].Transform = anim.GetBoneTransform(HumanBodyBones.Neck);
        jointPoints[(int) BodyPoints.Spine].Transform = anim.GetBoneTransform(HumanBodyBones.Spine);

		// Child Settings
        // Right Arm
        jointPoints[(int) BodyPoints.RightShoulder].Child = jointPoints[(int) BodyPoints.RightElbow];
        jointPoints[(int) BodyPoints.RightElbow].Child = jointPoints[(int) BodyPoints.RightWrist];
        jointPoints[(int) BodyPoints.RightElbow].Parent = jointPoints[(int) BodyPoints.RightShoulder];

		// Left Arm
        jointPoints[(int) BodyPoints.LeftShoulder].Child = jointPoints[(int) BodyPoints.LeftElbow];
        jointPoints[(int) BodyPoints.LeftElbow].Child = jointPoints[(int) BodyPoints.LeftWrist];
        jointPoints[(int) BodyPoints.LeftElbow].Parent = jointPoints[(int) BodyPoints.LeftShoulder];

        // Right Leg
        jointPoints[(int) BodyPoints.RightHip].Child = jointPoints[(int) BodyPoints.RightKnee];
        jointPoints[(int) BodyPoints.RightKnee].Child = jointPoints[(int) BodyPoints.RightAnkle];
        jointPoints[(int) BodyPoints.RightKnee].Parent = jointPoints[(int) BodyPoints.RightHip];

        // Left Leg
        jointPoints[(int) BodyPoints.LeftHip].Child = jointPoints[(int) BodyPoints.LeftKnee];
        jointPoints[(int) BodyPoints.LeftKnee].Child = jointPoints[(int) BodyPoints.LeftAnkle];
        jointPoints[(int) BodyPoints.LeftKnee].Parent = jointPoints[(int) BodyPoints.LeftHip];

		for (int i = 0; i < jointPoints.Length; i++) {
            if (jointPoints[i].Child == null || jointPoints[i].Child.Transform == null)
                continue;

            jointPoints[i].DistanceFromChild = Vector3.Distance(
                jointPoints[i].Child.Transform.position,
                jointPoints[i].Transform.position
            );
        }

		// Set Inverse
        Vector3 a = jointPoints[(int) BodyPoints.LeftHip].Transform.position;
        Vector3 b = jointPoints[(int) BodyPoints.Spine].Transform.position;
        Vector3 c = jointPoints[(int) BodyPoints.RightHip].Transform.position;
        Vector3 forward = TriangleNormal(a, b, c);

		foreach (var jointPoint in jointPoints) {
            if (jointPoint.Transform != null) {
                jointPoint.InitRotation = jointPoint.Transform.rotation;
            }
            if (jointPoint.Child != null) {
				var dir = jointPoint.Transform.position - jointPoint.Child.Transform.position;
                jointPoint.Inverse = Quaternion.Inverse(Quaternion.LookRotation(dir, forward));
                jointPoint.InverseRotation = jointPoint.Inverse * jointPoint.InitRotation;
            }
        }

		// Hip
        var hip = jointPoints[(int) BodyPoints.Hips];
        var spine = jointPoints[(int) BodyPoints.Spine];
        hip.Inverse = Quaternion.Inverse(Quaternion.LookRotation(
			forward,
			spine.Transform.position-hip.Transform.position)
		);
        hip.InverseRotation = hip.Inverse * hip.InitRotation;
        
        // Spine
        var right_shoulder = jointPoints[(int) BodyPoints.RightShoulder].Transform.position;
        var left_shoulder = jointPoints[(int) BodyPoints.LeftShoulder].Transform.position;
        
        spine.Inverse = Quaternion.Inverse(Quaternion.LookRotation(
            TriangleNormal(right_shoulder, spine.Transform.position, left_shoulder),
            jointPoints[(int) BodyPoints.Neck].Transform.position - spine.Transform.position)
        );
        spine.InverseRotation = spine.Inverse * spine.InitRotation;

		// For Head Rotation
        var head = jointPoints[(int) BodyPoints.Head];
		var gaze = head.Transform.up;

        head.InitRotation = jointPoints[(int) BodyPoints.Head].Transform.rotation;
        head.Inverse = Quaternion.Inverse(Quaternion.LookRotation(gaze));
		// head.InverseRotation = head.Inverse * head.InitRotation; // TODO: check why?
       
        head.InverseRotation = head.InitRotation;
        headUpVector = head.Transform.up;

		// feet setup
        
        var r_feet = jointPoints[(int) BodyPoints.RightAnkle];
        var l_feet = jointPoints[(int) BodyPoints.LeftAnkle];
        
        r_feet.Inverse = Quaternion.Inverse(Quaternion.LookRotation(
			r_feet.Transform.position - Vector3.zero, 
			jointPoints[(int) BodyPoints.RightKnee].Transform.position - r_feet.Transform.position)
		);
        r_feet.InverseRotation = r_feet.Inverse * r_feet.InitRotation;
        
        l_feet.Inverse = Quaternion.Inverse(Quaternion.LookRotation(
            l_feet.Transform.position - Vector3.zero, 
            jointPoints[(int) BodyPoints.LeftKnee].Transform.position - l_feet.Transform.position)
        );
        l_feet.InverseRotation = l_feet.Inverse * l_feet.InitRotation;

		for (int i = 0; i < jointPoints.Length; i++) {
            if(jointPoints[i].Transform != null) {
                jointPoints[i].LandmarkPose = jointPoints[i].Transform.position;
			}
        }
    
        // character.transform.rotation = characterPlacement.rotation;
        jointPoints[(int) BodyPoints.Hips].Transform.position = characterPlacement.position;
	}

	void Start() {
		anim = GetComponent<Animator>();
        anim.enabled = false;

		play_time = 0;
		if (System.IO.File.Exists (pos_filename) == false) {
			Debug.Log(
				"<color=blue>Error! Pos file not found(" + pos_filename + "). Check Pos_filename in Inspector.</color>"
			);
		}
		pos = ReadPosData(pos_filename);
		InitHumanoidPose();
		if (pos != null) {
			if (start_frame >= 0 && start_frame < pos.Count) 
				s_frame = start_frame;
			else 
				s_frame = 0;

			int ef;

			if (int.TryParse(end_frame, out ef)) {
				if (ef >= s_frame && ef < pos.Count) 
					e_frame = ef;
				else 
					e_frame = pos.Count - 1;
			} 
			else e_frame = pos.Count - 1;
		}
	}

	void Update() {
		if (pos == null) 
			return;

		play_time += Time.deltaTime;
		int frame = s_frame + (int)(play_time * 30.0f);

		if (frame > e_frame) {
			play_time = 0;
			frame = s_frame;
		}
        
        int prevFrameIndex = Mathf.Max(frame - 1, s_frame);
        float t = (play_time * 30.0f) - Mathf.Floor(play_time * 30.0f); // Доля между кадрами

        for (int i = 0; i < pos[frame].Length; i++) {
            jointPoints[i].LandmarkPose = Vector3.Lerp(pos[prevFrameIndex][i], pos[frame][i], t);
        }
        

        // Setting position of each bone
        jointPoints[(int) BodyPoints.Hips].Transform.position = pos[frame][(int) BodyPoints.Hips];

        for (int i = 0; i < jointPoints.Length && i < pos[frame].Length; i++) {
            JointPoint bone = jointPoints[i];

            if (bone.Transform != null) {
                bone.WorldPos = bone.Transform.position;
            }
        }

        for (int i = 0; i < jointPoints.Length && i < pos[frame].Length; i++) {
            JointPoint bone = jointPoints[i];
            
            if (bone.Child != null) {
                if (bone.Child.Transform != null)  {
                    JointPoint child = bone.Child;
                    Vector3 direction = (child.LandmarkPose - bone.LandmarkPose).normalized; // is equal to (-A + B) / |-A + B|
                    child.WorldPos = bone.Transform.position + direction * bone.DistanceFromChild;
                }
            }
            else {
                // if(i == (int) BodyPoints.RightShoulder || i == (int) BodyPoints.LeftShoulder || 
                //    i == (int) BodyPoints.LeftHip || i== (int) BodyPoints.RightHip || i == (int) BodyPoints.Head || i== (int) BodyPoints.Neck)
                //     continue;
                // if (jointPoints[i].Transform != null)
                // {
                //     if(bodyPartVectors[i].visibility > 0.75f)
                //         jointPoints[i].Transform.position = bodyPartVectors[i].position;
                // }
            }
        }

        for (int i = 0; i < jointPoints.Length; i++){
            jointPoints[i].FilteredPos = jointPoints[i].WorldPos;
        }

        // Setting hip & spine rotation
        Vector3 a = pos[frame][(int) BodyPoints.RightHip];
        Vector3 spine = pos[frame][(int) BodyPoints.Spine];
        Vector3 hip = pos[frame][(int) BodyPoints.Hips];
        Vector3 c = pos[frame][(int) BodyPoints.LeftHip];
        Vector3 d = pos[frame][(int) BodyPoints.RightShoulder];
        Vector3 e = pos[frame][(int) BodyPoints.LeftShoulder];
        Vector3 hipsUpward = spine - hip;
        Vector3 spineUpward = pos[frame][(int) BodyPoints.Neck] - spine;
        
        jointPoints[(int) BodyPoints.Hips].Transform.rotation = Quaternion.LookRotation(
            TriangleNormal(c, spine, a),
            hipsUpward 
        ) * jointPoints[(int) BodyPoints.Hips].InverseRotation;
        
        jointPoints[(int) BodyPoints.Spine].Transform.rotation = Quaternion.LookRotation(
            TriangleNormal(d, spine, e), 
            spineUpward 
        ) * jointPoints[(int) BodyPoints.Spine].InverseRotation;

        // Head Rotation
        // Vector3 mouth = (
        //     pos[frame][(int) BodyPoints.LeftMouth] +
        //     pos[frame][(int) BodyPoints.RightMouth]
        // )/2.0f;
        // Vector3 lEye = pos[frame][(int) BodyPoints.LeftEye];
        // Vector3 rEye = pos[frame][(int) BodyPoints.RightEye];
                
        // var gaze = TriangleNormal(mouth, lEye, rEye);

        Vector3 lEye = pos[frame][(int) BodyPoints.LeftEye];
        Vector3 rEye = pos[frame][(int) BodyPoints.RightEye];
        Vector3 nose = pos[frame][(int) BodyPoints.Nose];

        var gaze = TriangleNormal(nose, lEye, rEye); // вместо mouth
        var head = jointPoints[(int) BodyPoints.Head];
        
        // Vector3 nose = pos[frame][(int) BodyPoints.Nose];
        Vector3 rEar = pos[frame][(int) BodyPoints.RightEar];
        Vector3 lEar = pos[frame][(int) BodyPoints.LeftEar];
        Vector3 normal = TriangleNormal(rEar, nose, lEar);

        //head.Transform.rotation = Quaternion.LookRotation(gaze, normal) * jointPoints[(int) BodyPoints.Head].InverseRotation;

        // rotate each of bones
        Vector3 forward = jointPoints[(int) BodyPoints.Hips].Transform.forward;
        Vector3 leftHip = jointPoints[(int) BodyPoints.LeftHip].FilteredPos;
        Vector3 rightHip = jointPoints[(int) BodyPoints.RightHip].FilteredPos;

        forward = TriangleNormal(
            leftHip, 
            jointPoints[(int) BodyPoints.Spine].FilteredPos, 
            rightHip
        );

        foreach (var jointPoint in jointPoints) {
            if (jointPoint == null)
                continue;
            
            if (jointPoint.Parent != null) {
                Vector3 fv = jointPoint.Parent.FilteredPos - jointPoint.FilteredPos;
                jointPoint.Transform.rotation = Quaternion.LookRotation(
                    jointPoint.FilteredPos - jointPoint.Child.FilteredPos,
                    fv
                ) * jointPoint.InverseRotation;
            }
            else if (jointPoint.Child != null) {
                jointPoint.Transform.rotation = Quaternion.LookRotation(
                    (jointPoint.FilteredPos - jointPoint.Child.FilteredPos).normalized, 
                    forward
                ) * jointPoint.InverseRotation;
            }
        }

        // Calculate feet rotation
        Vector3 r_ankle = pos[frame][(int) BodyPoints.RightAnkle];
        Vector3 r_toe = Vector3.zero;//pos[frame][(int) BodyPoints.RightFootIndex];
        Vector3 r_knee = pos[frame][(int) BodyPoints.RightKnee];
        
        JointPoint r_ankleT = jointPoints[(int) BodyPoints.RightAnkle];
        // r_ankleT.Transform.rotation = Quaternion.LookRotation(
        //     r_ankle - r_toe, 
        //     r_knee - r_ankle
        // ) * r_ankleT.InverseRotation;

        r_ankleT.Transform.rotation = Quaternion.LookRotation(
            (r_ankle - pos[frame][(int) BodyPoints.RightHip]).normalized, 
            r_knee - r_ankle
        ) * r_ankleT.InverseRotation;
        
        Vector3 l_ankle = pos[frame][(int) BodyPoints.LeftAnkle];
        Vector3 l_toe = Vector3.zero;//pos[frame][(int) BodyPoints.LeftFootIndex];
        Vector3 l_knee = pos[frame][(int) BodyPoints.LeftKnee];
        
        JointPoint l_ankleT = jointPoints[(int) BodyPoints.LeftAnkle];
        // l_ankleT.Transform.rotation = Quaternion.LookRotation(
        //     l_ankle - l_toe, 
        //     l_knee - l_ankle
        // ) * l_ankleT.InverseRotation;

        l_ankleT.Transform.rotation = Quaternion.LookRotation(
            (l_ankle - pos[frame][(int) BodyPoints.LeftHip]).normalized,
            l_knee - l_ankle
        ) * l_ankleT.InverseRotation;
	}

	Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c) {
		return Vector3.Cross(a - b, a - c);
	}

	List<Vector3[]> ReadPosData(string filename) {
		List<Vector3[]> data = new List<Vector3[]>();
		List<string> lines = new List<string>();
		StreamReader sr = new StreamReader(filename);

		// Чтение строк из файла
		while (!sr.EndOfStream) {
			lines.Add(sr.ReadLine());
		}
		sr.Close();
		
		foreach (string line in lines) {

			// Убираются все запятые из строки
			string line2 = line.Replace(",", "");

			// Строка разбивается на части (координаты) по пробелам и очищаются пустые элементы
			string[] str = line2.Split(new string[] { " " }, System.StringSplitOptions.RemoveEmptyEntries);

			// Преобразование координат в Vector3
			Vector3[] vs = new Vector3[bone_num];
			for (int i = 0; i < str.Length; i += 4) {
				vs[int.Parse(str[i])] = new Vector3(
					float.Parse(str[i + 1]), 
					float.Parse(str[i + 2]), 
					float.Parse(str[i + 3])
				);
			}
			data.Add(vs);
		}

		return data;
	}
}