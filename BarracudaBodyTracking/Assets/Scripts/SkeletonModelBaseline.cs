using UnityEngine;

/// <summary>
/// Skeleton model aligned to Human3.6M / Martinez joint order.
/// Expects 17 joints in order:
/// 0 hip, 1 rHip, 2 rKnee, 3 rAnkle,
/// 4 lHip, 5 lKnee, 6 lAnkle,
/// 7 spine, 8 thorax, 9 neck, 10 head,
/// 11 lShoulder, 12 lElbow, 13 lWrist,
/// 14 rShoulder, 15 rElbow, 16 rWrist
/// </summary>
public class SkeletonModelBaseline : MonoBehaviour
{
    public class JointPoint
    {
        public Vector3 Pos3D;
        public Transform Bone;
        public JointPoint Parent;
        public JointPoint Child;
        public Quaternion InitRotation;
        public Quaternion InverseRotation;
    }

    public JointPoint[] JointPoints;
    private Animator anim;

    [Header("Avatar Reference")]
    public GameObject ModelObject;

    [Header("Debug")]
    public bool ShowSkeleton = true;
    public float SkeletonScale = 1f;

    [Header("Scaling")]
    public float scale = 0.01f;

    private Vector3 initRootOffset;

    public void Init()
    {
        JointPoints = new JointPoint[17];
        for (int i = 0; i < JointPoints.Length; i++)
            JointPoints[i] = new JointPoint();

        anim = ModelObject.GetComponent<Animator>();

        // Map H36M joints to Unity humanoid bones
        JointPoints[0].Bone  = anim.GetBoneTransform(HumanBodyBones.Hips);
        JointPoints[1].Bone  = anim.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        JointPoints[2].Bone  = anim.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        JointPoints[3].Bone  = anim.GetBoneTransform(HumanBodyBones.RightFoot);
        JointPoints[4].Bone  = anim.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        JointPoints[5].Bone  = anim.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        JointPoints[6].Bone  = anim.GetBoneTransform(HumanBodyBones.LeftFoot);
        JointPoints[7].Bone  = anim.GetBoneTransform(HumanBodyBones.Spine);
        JointPoints[8].Bone  = anim.GetBoneTransform(HumanBodyBones.Chest);
        JointPoints[9].Bone  = anim.GetBoneTransform(HumanBodyBones.Neck);
        JointPoints[10].Bone = anim.GetBoneTransform(HumanBodyBones.Head);
        JointPoints[11].Bone = anim.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        JointPoints[12].Bone = anim.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        JointPoints[13].Bone = anim.GetBoneTransform(HumanBodyBones.LeftHand);
        JointPoints[14].Bone = anim.GetBoneTransform(HumanBodyBones.RightUpperArm);
        JointPoints[15].Bone = anim.GetBoneTransform(HumanBodyBones.RightLowerArm);
        JointPoints[16].Bone = anim.GetBoneTransform(HumanBodyBones.RightHand);

        // Parent/child hierarchy
        Link(0, 1); Link(1, 2); Link(2, 3);
        Link(0, 4); Link(4, 5); Link(5, 6);
        Link(0, 7); Link(7, 8); Link(8, 9); Link(9, 10);
        Link(8, 11); Link(11, 12); Link(12, 13);
        Link(8, 14); Link(14, 15); Link(15, 16);

        // Cache inverse rotations
        foreach (var jp in JointPoints)
        {
            if (jp.Bone != null)
            {
                jp.InitRotation = jp.Bone.rotation;
                if (jp.Child != null && jp.Child.Bone != null)
                {
                    Vector3 fwd = jp.Child.Bone.position - jp.Bone.position;
                    jp.InverseRotation = Quaternion.Inverse(Quaternion.LookRotation(fwd)) * jp.InitRotation;
                }
            }
        }

        // Cache offset between GameObject root and hip
        if (JointPoints[0].Bone != null)
            initRootOffset = transform.position - JointPoints[0].Bone.position;
    }

    private void Link(int parent, int child)
    {
        JointPoints[parent].Child = JointPoints[child];
        JointPoints[child].Parent = JointPoints[parent];
    }

    /// <summary>
    /// Update skeleton pose from Martinez 3D joints.
    /// </summary>
    public void UpdatePose(Vector3[] joints3D)
{
    if (joints3D.Length != 17) return;

    // Store scaled positions
    for (int i = 0; i < joints3D.Length; i++)
        JointPoints[i].Pos3D = joints3D[i] * scale;

    // ✅ Move root by hip
    Vector3 hipWorld = JointPoints[0].Pos3D;
    transform.position = hipWorld;// + initRootOffset;

    // ✅ Compute hip forward from thighs
    Vector3 forward = TriangleNormal(
        JointPoints[0].Pos3D,
        JointPoints[4].Pos3D, // lHip
        JointPoints[1].Pos3D  // rHip
    );
    if (forward.sqrMagnitude < 1e-6f)
        forward = Vector3.forward;

    if (JointPoints[0].Bone)
        JointPoints[0].Bone.rotation =
            Quaternion.LookRotation(forward) * JointPoints[0].InverseRotation;

    // ✅ Update limbs & spine
    foreach (var jp in JointPoints)
    {
        if (!jp.Bone || jp.Child == null) continue;

        Vector3 dir = jp.Child.Pos3D - jp.Pos3D;
        if (dir.sqrMagnitude < 1e-6f) continue;

        Vector3 up = (jp.Parent != null)
            ? jp.Parent.Pos3D - jp.Pos3D
            : forward;

        if (up.sqrMagnitude < 1e-6f)
            up = Vector3.up;

        Quaternion targetRot = Quaternion.LookRotation(dir, up);
        jp.Bone.rotation = targetRot * jp.InverseRotation;
    }

    // ✅ Head orientation (neck → head as forward, spine up as up)
    if (JointPoints[10].Bone) // head
    {
        Vector3 gaze = JointPoints[10].Pos3D - JointPoints[9].Pos3D; // head - neck
        if (gaze.sqrMagnitude < 1e-6f) gaze = Vector3.forward;

        Vector3 up = JointPoints[8].Pos3D - JointPoints[7].Pos3D;    // thorax - spine
        if (up.sqrMagnitude < 1e-6f) up = Vector3.up;

        Quaternion headRot = Quaternion.LookRotation(gaze, up);
        JointPoints[10].Bone.rotation = headRot * JointPoints[10].InverseRotation;
    }

    // ✅ Wrists (stabilize twist)
    if (JointPoints[13].Bone) // left wrist
    {
        Vector3 dir = JointPoints[13].Pos3D - JointPoints[12].Pos3D;
        if (dir.sqrMagnitude < 1e-6f) dir = Vector3.forward;
        Quaternion lwRot = Quaternion.LookRotation(dir, forward);
        JointPoints[13].Bone.rotation = lwRot * JointPoints[13].InverseRotation;
    }
    if (JointPoints[16].Bone) // right wrist
    {
        Vector3 dir = JointPoints[16].Pos3D - JointPoints[15].Pos3D;
        if (dir.sqrMagnitude < 1e-6f) dir = Vector3.forward;
        Quaternion rwRot = Quaternion.LookRotation(dir, forward);
        JointPoints[16].Bone.rotation = rwRot * JointPoints[16].InverseRotation;
    }

    // ✅ Debug skeleton
    if (ShowSkeleton)
    {
        DrawBone(0, 1); DrawBone(1, 2); DrawBone(2, 3);
        DrawBone(0, 4); DrawBone(4, 5); DrawBone(5, 6);
        DrawBone(0, 7); DrawBone(7, 8); DrawBone(8, 9); DrawBone(9, 10);
        DrawBone(8, 11); DrawBone(11, 12); DrawBone(12, 13);
        DrawBone(8, 14); DrawBone(14, 15); DrawBone(15, 16);
    }
}

    private void DrawBone(int start, int end)
    {
        Vector3 a = transform.TransformPoint(JointPoints[start].Pos3D * SkeletonScale);
        Vector3 b = transform.TransformPoint(JointPoints[end].Pos3D * SkeletonScale);
        Debug.DrawLine(a, b, Color.green);
    }

    private Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 d1 = a - b;
        Vector3 d2 = a - c;
        Vector3 dd = Vector3.Cross(d1, d2);
        return dd.normalized;
    }
}
