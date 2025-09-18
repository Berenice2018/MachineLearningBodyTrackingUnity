using UnityEngine;

/// <summary>
/// COCO → H36M Remap + Normalize
/// </summary>
public static class JointMapperCocoH36M
{
    // COCO indices (for clarity)
    const int NOSE = 0;
    const int L_EYE = 1;
    const int R_EYE = 2;
    const int L_EAR = 3;
    const int R_EAR = 4;
    const int L_SHOULDER = 5;
    const int R_SHOULDER = 6;
    const int L_ELBOW = 7;
    const int R_ELBOW = 8;
    const int L_WRIST = 9;
    const int R_WRIST = 10;
    const int L_HIP = 11;
    const int R_HIP = 12;
    const int L_KNEE = 13;
    const int R_KNEE = 14;
    const int L_ANKLE = 15;
    const int R_ANKLE = 16;

    // H36M order we want: Pelvis, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle,
    // Spine, Thorax, Neck, Head, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist
   static Vector2 V2(in Vector3 v) => new Vector2(v.x, v.y);
    static Vector2 Mid(in Vector2 a, in Vector2 b) => (a + b) * 0.5f;

    // H36M order: Pelvis, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle,
    // Spine, Thorax, Neck, Head, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist
    public static Vector2[] CocoToH36M(Vector3[] coco2D, int imgW, int imgH, PoseProcessorAgnostic.NormMode mode = PoseProcessorAgnostic.NormMode.ImageSize)
    {
        // Convert COCO to H36M order (same as before) ...
        Vector2 nose   = V2(coco2D[NOSE]);
        Vector2 lHip   = V2(coco2D[L_HIP]);
        Vector2 rHip   = V2(coco2D[R_HIP]);
        Vector2 pelvis = Mid(lHip, rHip);

        Vector2 lSh    = V2(coco2D[L_SHOULDER]);
        Vector2 rSh    = V2(coco2D[R_SHOULDER]);
        Vector2 thorax = Mid(lSh, rSh);

        Vector2 neck   = Mid(thorax, nose);
        Vector2 head   = nose;               // proxy
        Vector2 spine  = Mid(pelvis, thorax);

        Vector2[] h36m = new Vector2[17];
        h36m[0]  = pelvis;
        h36m[1]  = rHip;
        h36m[2]  = V2(coco2D[R_KNEE]);
        h36m[3]  = V2(coco2D[R_ANKLE]);
        h36m[4]  = lHip;
        h36m[5]  = V2(coco2D[L_KNEE]);
        h36m[6]  = V2(coco2D[L_ANKLE]);
        h36m[7]  = spine;
        h36m[8]  = thorax;
        h36m[9]  = neck;
        h36m[10] = head;
        h36m[11] = lSh;
        h36m[12] = V2(coco2D[L_ELBOW]);
        h36m[13] = V2(coco2D[L_WRIST]);
        h36m[14] = rSh;
        h36m[15] = V2(coco2D[R_ELBOW]);
        h36m[16] = V2(coco2D[R_WRIST]);

        if (mode == PoseProcessorAgnostic.NormMode.ImageSize)
        {
            // Normalize to [-1,1] by image size, pelvis-center
            for (int i = 0; i < h36m.Length; i++)
            {
                float x = (h36m[i].x / (float)imgW) * 2f - 1f;
                float y = (h36m[i].y / (float)imgH) * 2f - 1f;
                h36m[i] = new Vector2(x, y);
            }
            Vector2 pelvisN = h36m[0];
            for (int i = 0; i < h36m.Length; i++) h36m[i] -= pelvisN;
        }
        else if (mode == PoseProcessorAgnostic.NormMode.PelvisHead)
        {
            // Normalize by pelvis–head distance
            Vector2 pelvisN = h36m[0];
            Vector2 headN   = h36m[10];
            float scale = Vector2.Distance(pelvisN, headN);
            if (scale < 1e-5f) scale = 1f; // avoid div0

            for (int i = 0; i < h36m.Length; i++)
                h36m[i] = (h36m[i] - pelvisN) / scale;
        }

        return h36m;
    }
    public static float[] FlattenH36M(Vector2[] h36m)
    {
        float[] flat = new float[h36m.Length * 2];
        for (int i = 0; i < h36m.Length; i++)
        {
            flat[i * 2 + 0] = h36m[i].x;
            flat[i * 2 + 1] = h36m[i].y;
        }
        return flat;
    }
}
