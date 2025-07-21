using UnityEngine;


public class JointPoint {
    //Landmark data
    public Vector3 LandmarkPose = new Vector3();
    public Vector3 WorldPos  = new Vector3();
    
    // Bones
    public Transform Transform = null;
    public Vector3 FilteredPos  = new Vector3();
    public Vector3[] LastPoses = new Vector3[6];
    public Quaternion InitRotation;
    public Quaternion Inverse;
    public Quaternion InverseRotation;
    public Vector3 InitialRotation;
    
    public JointPoint Child = null;
    public JointPoint Parent = null;

    public float DistanceFromChild;
    public float DistanceFromDad;
}