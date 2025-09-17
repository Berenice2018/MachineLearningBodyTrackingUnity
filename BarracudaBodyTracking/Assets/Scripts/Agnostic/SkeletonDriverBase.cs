// SkeletonDriverBase.cs
using UnityEngine;

public abstract class SkeletonDriverBase : MonoBehaviour
{
    [Header("Debug")]
    public bool showSkeleton = true;
    public float gizmoScale = 1f;

    [Header("Pose Scale")]
    public float metersScale = 0.01f; // mm→m or adjust per your 3D units

    protected SkeletonDefinition definition;

    public virtual void Init(SkeletonDefinition def)
    {
        definition = def;
    }

    public abstract void UpdatePose(Vector3[] joints3D);

    protected void DrawBonesLocal(Vector3[] jointsLocal)
    {
        if (!showSkeleton || definition?.bonePairs == null) return;
        foreach (var bp in definition.bonePairs)
        {
            if (bp.x < 0 || bp.x >= jointsLocal.Length || bp.y < 0 || bp.y >= jointsLocal.Length) continue;
            var a = transform.TransformPoint(jointsLocal[bp.x] * gizmoScale);
            var b = transform.TransformPoint(jointsLocal[bp.y] * gizmoScale);
            Debug.DrawLine(a, b, Color.cyan);
        }
    }
}