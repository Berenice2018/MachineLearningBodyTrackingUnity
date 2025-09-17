// HorseSkeletonDriver.cs (example; wire up your horse rig bones in Inspector)
using UnityEngine;

public class HorseSkeletonDriver : SkeletonDriverBase
{
    [Header("Horse Bones (assign in Inspector in the same index order as your definition)")]
    public Transform[] horseBones; // length == definition.NumJoints
    private Quaternion[] invRots;

    public override void Init(SkeletonDefinition def)
    {
        base.Init(def);
        invRots = new Quaternion[def.NumJoints];
        for (int i = 0; i < def.NumJoints; i++)
            invRots[i] = horseBones[i] ? horseBones[i].rotation : Quaternion.identity;
    }

    public override void UpdatePose(Vector3[] joints3D)
    {
        if (definition == null || joints3D == null || joints3D.Length != definition.NumJoints) return;

        var J = new Vector3[joints3D.Length];
        for (int i = 0; i < J.Length; i++) J[i] = joints3D[i] * metersScale;

        // Place root at withers or pelvis index (choose according to your horse definition)
        int rootIdx = 0; // e.g., withers
        transform.position = J[rootIdx];

        // Rotate bones by parent→child
        for (int i = 0; i < horseBones.Length; i++)
        {
            if (!horseBones[i]) continue;
            int child = FirstChild(i);
            if (child < 0 || child >= J.Length) continue;
            Vector3 dir = J[child] - J[i];
            if (dir.sqrMagnitude < 1e-8f) continue;

            Vector3 up = Vector3.up;
            horseBones[i].rotation = Quaternion.LookRotation(dir, up) * invRots[i];
        }

        DrawBonesLocal(J);
    }

    // Replace with your horse hierarchy
    private int FirstChild(int i)
    {
        // Example placeholder: no hierarchy known → return -1
        return -1;
    }
}