// JointMapper.cs
using UnityEngine;

public static class JointMapper
{
    /// <summary>
    /// Map COCO-17 (MoveNet) → H36M-17 indices.
    /// Returns a float2[] (x,y) in [0,1] or your chosen 2D space.
    /// </summary>
    public static Vector2[] COCOMoveNetToH36M(Vector3[] cocoXYC)
    {
        // cocoXYC: Vector3(x,y,conf), length 17
        // Build midpoints to create torso joints required by H36M
        Vector2 mid(Vector2 a, Vector2 b) => 0.5f*(a+b);

        Vector2[] h = new Vector2[17];
        // Coco indexing used by MoveNet: 0 nose,1 lEye,2 rEye,3 lEar,4 rEar,5 lShoulder,6 rShoulder,
        // 7 lElbow,8 rElbow,9 lWrist,10 rWrist,11 lHip,12 rHip,13 lKnee,14 rKnee,15 lAnkle,16 rAnkle
        var c = cocoXYC;

        var pelvis = mid(c[11], c[12]);        // LHip,RHip
        var thorax = mid(c[5],  c[6]);         // LShoulder,RShoulder
        var neck   = mid(thorax, c[0]);        // midpoint(thorax,nose)
        var head   = (Vector2)c[0];            // nose proxy

        h[0]  = pelvis;
        h[1]  = c[12];  // RHip
        h[2]  = c[14];  // RKnee
        h[3]  = c[16];  // RAnkle
        h[4]  = c[11];  // LHip
        h[5]  = c[13];  // LKnee
        h[6]  = c[15];  // LAnkle
        h[7]  = 0.5f*(pelvis + thorax); // Spine
        h[8]  = thorax;
        h[9]  = neck;
        h[10] = head;
        h[11] = c[5];   // LShoulder
        h[12] = c[7];   // LElbow
        h[13] = c[9];   // LWrist
        h[14] = c[6];   // RShoulder
        h[15] = c[8];   // RElbow
        h[16] = c[10];  // RWrist

        return h;
    }

    /// <summary>
    /// Identity mapper for already-matched skeletons
    /// (e.g., Horse detector and lifter both use Horse-21 order).
    /// </summary>
    public static Vector2[] Identity(Vector3[] xyConf)
    {
        var out2d = new Vector2[xyConf.Length];
        for (int i = 0; i < xyConf.Length; i++) out2d[i] = new Vector2(xyConf[i].x, xyConf[i].y);
        return out2d;
    }
}