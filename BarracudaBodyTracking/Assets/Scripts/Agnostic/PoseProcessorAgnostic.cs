// PoseProcessor.cs
using UnityEngine;
using Unity.Sentis;

/// <summary>
/// This ingests any [N×3] detector output, maps → normalizes → (optional) lift → axis fix → driver.
/// </summary>
public class PoseProcessorAgnostic : MonoBehaviour
{
    [Header("Definitions")]
    public SkeletonDefinition detectorDef;   // e.g., Human-COCO-17 (MoveNet)
    public SkeletonDefinition lifterDef;     // e.g., Human-H36M-17 (Martinez)
    public SkeletonDriverBase driver;        // HumanSkeletonDriver or HorseSkeletonDriver

    [Header("3D Lifter (Optional)")]
    public ModelAsset lifterModelAsset;      // FC lifter model
    public bool useLifter = true;

    [Header("Axis/Normalization")]
    public AxisMode axis = AxisMode.NegX_Z_NegY; // field with header above it

    public enum AxisMode { Raw, MirrorX, SwapYZ, NegX_Z_NegY }

    [Header("Normalization Mode")]
    public NormMode norm = NormMode.PelvisHead; // field with header above it

    public enum NormMode { PelvisHead, MinusOneOne, None }


    [Header("Debug")]
    public bool drawRaw3D = true;
    public Skeleton2DVisualizer skeleton2DVisualizer;
    private bool _isCocoSkeleton, _isH36MLifter;
    private Model lifterModel;
    private Worker lifterWorker;

    void Start()
    {
        _isCocoSkeleton = detectorDef.skeletonName.ToLower().Contains("coco");
        _isH36MLifter = lifterDef.skeletonName.ToLower().Contains("h36m");
        
        //init debug skeleton visualizer
        skeleton2DVisualizer.Init(detectorDef);  // e.g., Human-COCO-17

        
        if (useLifter && lifterModelAsset)
        {
            lifterModel = ModelLoader.Load(lifterModelAsset);
            lifterWorker = new Worker(lifterModel, BackendType.GPUCompute);
        }
        driver.Init(lifterDef ?? detectorDef);
    }

    void OnDestroy() => lifterWorker?.Dispose();

    /// <summary>
    /// Public entry: call with detector output as flat float[] length = N*3 (y,x,conf) or (x,y,conf). 
    /// Here we assume MoveNet order (y,x,score) normalized to [0,1].
    /// </summary>
    public void Upload2DFromDetector(float[] flatYXC)
    {
        int n = detectorDef.NumJoints;
        if (flatYXC == null || flatYXC.Length < n*3) return;

        // Parse detector joints → Vector3(x,y,conf)
        var det = new Vector3[n];
        for (int j = 0; j < n; j++)
        {
            float y = flatYXC[j*3+0];
            float x = flatYXC[j*3+1];
            float c = flatYXC[j*3+2];
            det[j] = new Vector3(x, y, c);
        }

        // Forward to COCO visualizer only if skeleton is COCO-17
        if (skeleton2DVisualizer && _isCocoSkeleton)
            skeleton2DVisualizer.SetKeypoints(det); // Vector3[x,y,conf]
        
        // Map detector order → lifter order (or identity if same)
        Vector2[] lift2D;
        if (_isCocoSkeleton && lifterDef && _isH36MLifter)
            lift2D = JointMapper.COCOMoveNetToH36M(det);
        else
            lift2D = JointMapper.Identity(det); // same indexing (e.g., Horse)

        Vector3[] joints3D;
        if (useLifter && lifterWorker != null)
            joints3D = Lift2DTo3D(lift2D);
        else
            joints3D = DepthProjectStub(lift2D); // or return zeros; replace with AR depth version

        // Axis remap for Unity
        for (int i = 0; i < joints3D.Length; i++) joints3D[i] = AxisRemap(joints3D[i], axis);

        if (drawRaw3D) DebugDraw(lifterDef ?? detectorDef, joints3D);

        driver.UpdatePose(joints3D);
    }

    private Vector3 AxisRemap(Vector3 p, AxisMode m)
    {
        return m switch {
            AxisMode.Raw => new Vector3(p.x, p.y, p.z),
            AxisMode.MirrorX => new Vector3(-p.x, p.y, p.z),
            AxisMode.SwapYZ => new Vector3(p.x, p.z, -p.y),
            AxisMode.NegX_Z_NegY => new Vector3(-p.x, p.z, -p.y),
            _ => p
        };
    }

    private Vector3[] Lift2DTo3D(Vector2[] in2D)
    {
        // normalize to pelvis-centered with optional scale
        var h = (Vector2[])in2D.Clone();
        Vector2 root = h[0];
        float s = 1f;
        if (norm == NormMode.PelvisHead)
        {
            float d = Vector2.Distance(h[0], h.Length > 10 ? h[10] : h[0]); // pelvis→head if available
            if (d > 1e-6f) s = d;
        }
        else if (norm == NormMode.MinusOneOne)
        {
            // assumes h in [0,1]
            for (int i = 0; i < h.Length; i++) h[i] = new Vector2(h[i].x * 2f - 1f, h[i].y * 2f - 1f);
            root = Vector2.zero; // already centered if you prefer
        }

        float[] flat = new float[h.Length*2];
        for (int i = 0; i < h.Length; i++)
        {
            flat[i*2+0] = (h[i].x - root.x) / s;
            flat[i*2+1] = (h[i].y - root.y) / s;
        }

        using var liftIn = new Tensor<float>(new TensorShape(1, flat.Length), flat);
        lifterWorker.SetInput(lifterModel.inputs[0].name, liftIn);
        lifterWorker.Schedule();
        var outT  = lifterWorker.PeekOutput() as Tensor<float>;
        var raw   = outT.DownloadToArray(); // length = NumJoints*3

        var J = new Vector3[h.Length];
        for (int i = 0; i < h.Length; i++)
            J[i] = new Vector3(raw[i*3+0], raw[i*3+1], raw[i*3+2]);
        return J;
    }

    private Vector3[] DepthProjectStub(Vector2[] in2D)
    {
        // Placeholder: returns zero Z. Replace with ARFoundation depth sampling per joint.
        var J = new Vector3[in2D.Length];
        for (int i = 0; i < J.Length; i++) J[i] = new Vector3(in2D[i].x, in2D[i].y, 0f);
        return J;
    }

    private void DebugDraw(SkeletonDefinition def, Vector3[] joints)
    {
        if (def?.bonePairs == null) return;
        foreach (var bp in def.bonePairs)
        {
            if (bp.x < 0 || bp.x >= joints.Length || bp.y < 0 || bp.y >= joints.Length) continue;
            Debug.DrawLine(joints[bp.x], joints[bp.y], Color.magenta);
        }
    }
}
