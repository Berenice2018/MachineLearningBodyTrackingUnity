// HumanSkeletonDriver.cs Martinez (H36M order expected)
using UnityEngine;

public class HumanSkeletonDriver : SkeletonDriverBase
{
    public Animator animator;

    // Cache bones by index (H36M order)
    private Transform[] bones; // length def.NumJoints
    private Quaternion[] invRots;

    public override void Init(SkeletonDefinition def)
    {
        base.Init(def);
        if (!animator) animator = GetComponentInChildren<Animator>();

        bones = new Transform[def.NumJoints];
        invRots = new Quaternion[def.NumJoints];

        // Map H36M indices → Humanoid bones
        // 0 Hip,1 RHip,2 RKnee,3 RAnkle,4 LHip,5 LKnee,6 LAnkle,7 Spine,8 Thorax,
        // 9 Neck,10 Head,11 LShoulder,12 LElbow,13 LWrist,14 RShoulder,15 RElbow,16 RWrist
        bones[0]  = animator.GetBoneTransform(HumanBodyBones.Hips);
        bones[1]  = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        bones[2]  = animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        bones[3]  = animator.GetBoneTransform(HumanBodyBones.RightFoot);
        bones[4]  = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        bones[5]  = animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        bones[6]  = animator.GetBoneTransform(HumanBodyBones.LeftFoot);
        bones[7]  = animator.GetBoneTransform(HumanBodyBones.Spine);
        bones[8]  = animator.GetBoneTransform(HumanBodyBones.Chest);
        bones[9]  = animator.GetBoneTransform(HumanBodyBones.Neck);
        bones[10] = animator.GetBoneTransform(HumanBodyBones.Head);
        bones[11] = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        bones[12] = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        bones[13] = animator.GetBoneTransform(HumanBodyBones.LeftHand);
        bones[14] = animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        bones[15] = animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
        bones[16] = animator.GetBoneTransform(HumanBodyBones.RightHand);

        // Precompute inverse “look-at” offsets (same idea as your baseline)
        for (int i = 0; i < bones.Length; i++)
        {
            if (!bones[i]) continue;
            // pick a child if exists to get forward
            int child = FindChildIndex(i);
            if (child >= 0 && bones[child])
            {
                Vector3 fwd = bones[child].position - bones[i].position;
                invRots[i] = Quaternion.Inverse(Quaternion.LookRotation(fwd)) * bones[i].rotation;
            }
            else invRots[i] = Quaternion.identity;
        }
    }

    public override void UpdatePose(Vector3[] joints3D)
    {
        if (definition == null || joints3D == null || joints3D.Length != definition.NumJoints) return;

        // Convert to local units
        var J = new Vector3[joints3D.Length];
        for (int i = 0; i < J.Length; i++) J[i] = joints3D[i] * metersScale;

        // Place root at hip
        transform.position = J[0];

        // Forward from hips triangle
        Vector3 forward = Vector3.Cross(J[4] - J[0], J[1] - J[0]).normalized;
        if (forward.sqrMagnitude < 1e-6f) forward = Vector3.forward;

        // Rotate bones by “look from parent to child”
        for (int i = 0; i < bones.Length; i++)
        {
            if (!bones[i]) continue;
            int child = FindChildIndex(i);
            if (child < 0) continue;

            Vector3 dir = J[child] - J[i];
            if (dir.sqrMagnitude < 1e-8f) continue;

            Vector3 up = (GetParentIndex(i) >= 0) ? (J[GetParentIndex(i)] - J[i]) : forward;
            if (up.sqrMagnitude < 1e-8f) up = Vector3.up;

            bones[i].rotation = Quaternion.LookRotation(dir, up) * invRots[i];
        }

        DrawBonesLocal(J);
    }

    // H36M hierarchy helpers (same as before)
    private int GetParentIndex(int i)
    {
        return i switch {
            1=>0,2=>1,3=>2, 4=>0,5=>4,6=>5, 7=>0,8=>7,9=>8,10=>9, 11=>8,12=>11,13=>12, 14=>8,15=>14,16=>15, _=>-1
        };
    }
    private int FindChildIndex(int i)
    {
        // return first child for forward reference
        switch(i){
            case 0: return 1; case 1: return 2; case 2: return 3;
            case 4: return 5; case 5: return 6;
            case 7: return 8; case 8: return 9; case 9: return 10;
            case 11: return 12; case 12: return 13;
            case 14: return 15; case 15: return 16;
            default: return -1;
        }
    }
}
