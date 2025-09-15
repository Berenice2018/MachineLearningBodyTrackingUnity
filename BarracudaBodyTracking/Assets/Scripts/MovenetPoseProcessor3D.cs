using UnityEngine;
using Unity.Sentis;

public class MovenetPoseProcessor3D : MonoBehaviour
{
    [Header("Models")]
    public ModelAsset baseline3DModelAsset;

    [Header("Skeleton")]
    public SkeletonModelBaseline skeleton;

    [Header("Debug")]
    public bool logOnce = true;   // set true to print shapes/mapping once
    public Coco2DVisualizer coco2DVisualizer;
    
    private Worker baselineWorker;
    private Model baselineModel;

    // H36M 17-joint order expected by Martinez:
    // 0 Pelvis, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle,
    // 7 Spine, 8 Thorax, 9 Neck, 10 Head,
    // 11 LShoulder, 12 LElbow, 13 LWrist, 14 RShoulder, 15 RElbow, 16 RWrist
    private const int H36M_JOINTS = 17;

    void Start()
    {
        baselineModel  = ModelLoader.Load(baseline3DModelAsset);
        baselineWorker = new Worker(baselineModel, BackendType.GPUCompute);

        if (logOnce)
        {
            foreach (var i in baselineModel.inputs)
                Debug.Log($"[Martinez] Input: {i.name} shape={i.shape}  (expect 1x34)");
            foreach (var o in baselineModel.outputs)
                Debug.Log($"[Martinez] Output: {o}");
        }

        skeleton.Init();
    }

    /// <summary>
    /// Called by MovenetRunner with MoveNet output: float[17*3] (x,y,conf)
    /// </summary>
    public void UploadNetworkOutputs(float[] keypoints2D)
    {
        // 1) Parse COCO 2D
        Vector2[] coco = new Vector2[17];
        for (int j = 0; j < 17; j++)
            coco[j] = new Vector2(keypoints2D[j*3+0], keypoints2D[j*3+1]);

        // 2) Build H36M-17 from COCO (adds pelvis/spine/thorax/neck midpoints)
        Vector2[] h36m = BuildH36MFromCOCO(coco);

        // Optional one-time sanity print
        if (logOnce)
        {
            Debug.Log(
                $"[Remap] Pelvis={h36m[0]:F3} Thorax={h36m[8]:F3} Neck={h36m[9]:F3} " +
                $"L/R shoulders={h36m[11]:F3}/{h36m[14]:F3}  L/R hips={h36m[4]:F3}/{h36m[1]:F3}"
            );
            logOnce = false;
        }

        // 3) Root-normalize (pelvis centered) and flatten to (1,34)
        Vector2 root = h36m[0];
        float[] flat = new float[H36M_JOINTS*2];
        for (int j = 0; j < H36M_JOINTS; j++)
        {
            flat[j*2+0] = h36m[j].x - root.x;
            flat[j*2+1] = h36m[j].y - root.y;
        }

        // Visualize raw MoveNet 2D
        coco2DVisualizer.SetKeypoints(keypoints2D);
        return;
        
        using var liftIn = new Tensor<float>(new TensorShape(1, flat.Length), flat);

        // 4) Lift to 3D
        baselineWorker.SetInput(baselineModel.inputs[0].name, liftIn);
        baselineWorker.Schedule();
        var outT  = baselineWorker.PeekOutput() as Tensor<float>;
        var raw3D = outT.DownloadToArray(); // length = 17*3 = 51

        // 5) Pack back in H36M order
        Vector3[] joints3D = new Vector3[H36M_JOINTS];
        for (int j = 0; j < H36M_JOINTS; j++)
            joints3D[j] = new Vector3(raw3D[j*3+0], raw3D[j*3+1], raw3D[j*3+2]);

        // 6) Axis fix (try A/B/C if needed)
        for (int i = 0; i < joints3D.Length; i++)
        {
            var p = joints3D[i];
            // A) Common for TF→Unity
            joints3D[i] = new Vector3(-p.x, p.z, -p.y);
            // B) If still tilted: joints3D[i] = new Vector3(p.x, p.y, -p.z);
            // C) Mirror X if left/right inverted: joints3D[i] = new Vector3(-joints3D[i].x, joints3D[i].y, joints3D[i].z);
        }

        // 7) Drive skeleton (expects H36M order)
        skeleton.UpdatePose(joints3D);
    }

    /// <summary>
    /// Convert COCO-17 to H36M-17 using midpoints for torso joints.
    /// </summary>
    private Vector2[] BuildH36MFromCOCO(Vector2[] c)
    {
        var h = new Vector2[H36M_JOINTS];

        // Helpers
        Vector2 mid(Vector2 a, Vector2 b) => 0.5f*(a+b);

        // Torso midpoints
        var pelvis = mid(c[11], c[12]); // LHip, RHip
        var thorax = mid(c[5],  c[6]);  // LShoulder, RShoulder
        var neck   = mid(thorax, c[0]); // (thorax,nose)
        var head   = c[0];              // nose proxy

        // H36M layout in order
        h[0]  = pelvis;
        h[1]  = c[12];  // RHip
        h[2]  = c[14];  // RKnee
        h[3]  = c[16];  // RAnkle
        h[4]  = c[11];  // LHip
        h[5]  = c[13];  // LKnee
        h[6]  = c[15];  // LAnkle
        h[7]  = mid(pelvis, thorax); // Spine (between pelvis and thorax)
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

    void OnDestroy() => baselineWorker?.Dispose();
}
