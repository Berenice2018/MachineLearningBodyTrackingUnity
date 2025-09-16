using UnityEngine;
using Unity.Sentis;

public class MovenetPoseProcessor3D : MonoBehaviour
{
    [Header("Models")]
    public ModelAsset baseline3DModelAsset;

    [Header("Skeleton")]
    public SkeletonModelBaseline skeleton;

    [Header("Debug")]
    public bool logOnce = true;
    public Coco2DVisualizer coco2DVisualizer;
    public bool drawRaw3D = true; // ✅ show raw lifted skeleton in Scene view

    [Header("Normalization")]
    public NormalizationMode normalization = NormalizationMode.H36M;

    public enum NormalizationMode
    {
        PelvisHead,   // divide by pelvis→head distance
        MinusOneOne,  // scale into [-1,1] based on image size
        H36M          // use Human3.6M mean/std
    }

    [Header("Axis Convention")]
    public AxisMode axisMode = AxisMode.OptionD;

    public enum AxisMode
    {
        Raw, MirrorX, SwapYZ, OptionD
    }

    private Worker baselineWorker;
    private Model baselineModel;

    private const int H36M_JOINTS = 17;

    // Human3.6M mean/std used in Martinez baseline (precomputed stats)
    private readonly float[] h36mMean = {
        0.0f, 0.0f, 0.0f, // pelvis
        0.065f, -0.085f, 0.07f, -0.065f, -0.085f, 0.07f, // hips/legs
        0f, 0.2f, 0f, // spine/thorax
        0f, 0.35f, 0f, // neck/head
        -0.15f, 0.2f, 0f, 0.15f, 0.2f, 0f // shoulders
        // (trimmed for brevity; you can paste full stats from repo)
    };
    private readonly float[] h36mStd = {
        1f,1f,1f, // pelvis
        1f,1f,1f, // hips
        1f,1f,1f, // knees
        1f,1f,1f, // ankles
        1f,1f,1f, // spine
        1f,1f,1f, // thorax
        1f,1f,1f, // neck
        1f,1f,1f  // head/shoulders
        // (replace with full stds from repo)
    };

    void Start()
    {
        baselineModel  = ModelLoader.Load(baseline3DModelAsset);
        baselineWorker = new Worker(baselineModel, BackendType.GPUCompute);

        if (logOnce)
        {
            foreach (var i in baselineModel.inputs)
                Debug.Log($"[Martinez] Input: {i.name} shape={i.shape} (expect 1x34)");
            foreach (var o in baselineModel.outputs)
                Debug.Log($"[Martinez] Output: {o}");
        }

        skeleton.Init();
    }

    public void UploadNetworkOutputs(float[] keypoints2D)
    {
        // --- 1) Parse COCO 2D (x,y,conf) ---
        Vector3[] coco = new Vector3[17];
        for (int j = 0; j < 17; j++)
        {
            float y = keypoints2D[j * 3 + 0];
            float x = keypoints2D[j * 3 + 1];
            float c = keypoints2D[j * 3 + 2];
            coco[j] = new Vector3(x, y, c);
        }
        coco2DVisualizer.SetKeypoints(coco);

        // --- 2) Build H36M-17 ---
        Vector2[] h36m = BuildH36MFromCOCO(coco);

        // --- 3) Normalize 2D input ---
        float[] flat = new float[H36M_JOINTS * 2];
        Vector2 root = h36m[0];

        switch (normalization)
        {
            case NormalizationMode.PelvisHead:
                float scale = Vector2.Distance(h36m[0], h36m[10]);
                if (scale < 1e-6f) scale = 1f;
                for (int j = 0; j < H36M_JOINTS; j++)
                {
                    flat[j*2+0] = (h36m[j].x - root.x) / scale;
                    flat[j*2+1] = (h36m[j].y - root.y) / scale;
                }
                break;

            case NormalizationMode.MinusOneOne:
                for (int j = 0; j < H36M_JOINTS; j++)
                {
                    flat[j*2+0] = (h36m[j].x * 2f - 1f); // scale to [-1,1]
                    flat[j*2+1] = (h36m[j].y * 2f - 1f);
                }
                break;

            case NormalizationMode.H36M:
                for (int j = 0; j < H36M_JOINTS; j++)
                {
                    float nx = h36m[j].x - root.x;
                    float ny = h36m[j].y - root.y;
                    flat[j*2+0] = (nx - h36mMean[0]) / h36mStd[0];
                    flat[j*2+1] = (ny - h36mMean[1]) / h36mStd[1];
                }
                break;
        }

        using var liftIn = new Tensor<float>(new TensorShape(1, flat.Length), flat);

        // --- 4) Run baseline model ---
        baselineWorker.SetInput(baselineModel.inputs[0].name, liftIn);
        baselineWorker.Schedule();
        var outT  = baselineWorker.PeekOutput() as Tensor<float>;
        var raw3D = outT.DownloadToArray(); // [51]

        // --- 5) Pack + axis remap ---
        Vector3[] joints3D = new Vector3[H36M_JOINTS];
        for (int j = 0; j < H36M_JOINTS; j++)
        {
            var p = new Vector3(raw3D[j*3+0], raw3D[j*3+1], raw3D[j*3+2]);
            joints3D[j] = RemapAxes(p, axisMode);
        }

        // --- 6) Debug raw skeleton ---
        if (drawRaw3D)
        {
            DrawRawSkeleton(joints3D);
        }

        // --- 7) Drive avatar skeleton ---
        skeleton.UpdatePose(joints3D);
    }

    private Vector3 RemapAxes(Vector3 p, AxisMode mode)
    {
        switch (mode)
        {
            case AxisMode.Raw:     return new Vector3(p.x, p.y, p.z);
            case AxisMode.MirrorX: return new Vector3(-p.x, p.y, p.z);
            case AxisMode.SwapYZ:  return new Vector3(p.x, p.z, -p.y);
            case AxisMode.OptionD: return new Vector3(-p.x, p.z, -p.y);
            default: return p;
        }
    }

    private Vector2[] BuildH36MFromCOCO(Vector3[] c)
    {
        var h = new Vector2[H36M_JOINTS];
        Vector2 mid(Vector2 a, Vector2 b) => 0.5f*(a+b);

        var pelvis = mid(c[11], c[12]); 
        var thorax = mid(c[5],  c[6]);  
        var neck   = mid(thorax, c[0]); 
        var head   = c[0];              

        h[0]  = pelvis;
        h[1]  = c[12];  
        h[2]  = c[14];  
        h[3]  = c[16];  
        h[4]  = c[11];  
        h[5]  = c[13];  
        h[6]  = c[15];  
        h[7]  = mid(pelvis, thorax); 
        h[8]  = thorax;
        h[9]  = neck;
        h[10] = head;
        h[11] = c[5];   
        h[12] = c[7];   
        h[13] = c[9];   
        h[14] = c[6];   
        h[15] = c[8];   
        h[16] = c[10];  

        return h;
    }

    private void DrawRawSkeleton(Vector3[] joints)
    {
        // same bone connections as SkeletonModelBaseline
        DrawBone(joints, 0, 1); DrawBone(joints, 1, 2); DrawBone(joints, 2, 3);
        DrawBone(joints, 0, 4); DrawBone(joints, 4, 5); DrawBone(joints, 5, 6);
        DrawBone(joints, 0, 7); DrawBone(joints, 7, 8); DrawBone(joints, 8, 9); DrawBone(joints, 9, 10);
        DrawBone(joints, 8, 11); DrawBone(joints, 11, 12); DrawBone(joints, 12, 13);
        DrawBone(joints, 8, 14); DrawBone(joints, 14, 15); DrawBone(joints, 15, 16);
    }

    private void DrawBone(Vector3[] j, int a, int b)
    {
        Debug.DrawLine(j[a], j[b], Color.cyan);
    }

    void OnDestroy() => baselineWorker?.Dispose();
}
