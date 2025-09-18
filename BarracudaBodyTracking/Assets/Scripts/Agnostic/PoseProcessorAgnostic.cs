// PoseProcessor.cs
using UnityEngine;
using Unity.Sentis;
using UnityEngine.Serialization;

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
    private PoseSlidingBuffer _slidingBuffer;
    public int bufferWindowSize = 243; // or 27 for mobile
    int joints = 17;
    int channels = 2;
    private float[] flat;

    [Header("Axis/Normalization")]
    public AxisMode axis = AxisMode.NegX_Z_NegY; // field with header above it

    public enum AxisMode { Raw, MirrorX, SwapYZ, NegX_Z_NegY }

    [Header("Normalization Mode")]
    public NormMode norm = NormMode.PelvisHead; // field with header above it

    public enum NormMode { ImageSize, PelvisHead, MinusOneOne, None }


    [Header("Debug")]
    public Skeleton2DVisualizer skeleton2DVisualizer;
    private bool _isCocoSkeleton, _isH36MLifter;
    private Model lifterModel;
    private Worker lifterWorker;
    const int uiUpdateInterval = 2; // change 2→3 or 4 for more aggressive throttling

    void Start()
    {
        _slidingBuffer = new PoseSlidingBuffer(bufferWindowSize);
        flat = new float[1 * bufferWindowSize * joints * channels];
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
        if (flatYXC == null || flatYXC.Length < n * 3) return;

        // Parse detector joints → Vector3(x,y,conf)
        var det = new Vector3[n];
        for (int j = 0; j < n; j++)
        {
            float y = flatYXC[j * 3 + 0];
            float x = flatYXC[j * 3 + 1];
            float c = flatYXC[j * 3 + 2];
            /*if (c < 0.2f) 
                det[j] = Vector3.zero;
            else*/
                det[j] = new Vector3(x, y, c);
        }

        // --- Throttle UI Debug ---
        // Only update 2D visualizer every N frames to save CPU/UI cost
        if (skeleton2DVisualizer && _isCocoSkeleton)
        {
            //if (Time.frameCount % uiUpdateInterval == 0)
                skeleton2DVisualizer.SetKeypoints(det);
        }

        // --- Always drive avatar ---
        // Map detector → lifter skeleton
        Vector2[] lift2D;
        if (_isCocoSkeleton && lifterDef && _isH36MLifter)
        {
            //We got already normalized MoveNet output
            lift2D = JointMapperCocoH36M.CocoToH36M(det,1, 1 );
            //for (int j = 0; j < lift2D.Length; j++)
                //Debug.Log($"### Lift2D[{j}] {lift2D[j]} ({lifterDef.jointNames[j]})");
        }
        else
            lift2D = JointMapper.Identity(det); // same indexing (e.g., Horse)

        // Lift to 3D or depth project
        // --- Push new frame into sliding buffer ---
        _slidingBuffer.AddFrame(lift2D);

        Vector3[] joints3D;

        if (useLifter && lifterWorker != null && _slidingBuffer.IsReady())
        {
            // --- DEBUG: Peek into sliding buffer contents ---
            var tensor2D = _slidingBuffer.ToTensor(); // [1,frames,17,2]
            int midFrame = bufferWindowSize / 2;
            string bufSample = "";
            for (int j = 0; j < 5; j++) // just first 5 joints for readability
            {
                float x = tensor2D[0, midFrame, j, 0];
                float y = tensor2D[0, midFrame, j, 1];
                bufSample += $"J{j}:({x:F2},{y:F2}) ";
            }
            //Debug.Log($"### SlidingBuffer sample (frame {midFrame}): {bufSample}");
            
            // Export buffer to [1,frames,17,2]
            float[,,,] inputTensor = _slidingBuffer.ToTensor();

            int idx = 0;
            for (int f = 0; f < bufferWindowSize; f++)
            {
                for (int j = 0; j < joints; j++)
                {
                    flat[idx++] = inputTensor[0, f, j, 0];
                    flat[idx++] = inputTensor[0, f, j, 1];
                }
            }

            // Create Sentis tensor with explicit shape
            using var liftIn = new Tensor<float>(new TensorShape(1, bufferWindowSize, joints, channels), flat);
            lifterWorker.SetInput(lifterModel.inputs[0].name, liftIn);
            lifterWorker.Schedule();

            var outT = lifterWorker.PeekOutput() as Tensor<float>;
            var raw = outT.DownloadToArray();

            joints3D = new Vector3[lift2D.Length];
            for (int i = 0; i < lift2D.Length; i++)
                joints3D[i] = new Vector3(raw[i * 3 + 0], raw[i * 3 + 1], raw[i * 3 + 2]);
            
            Debug.Log($"### lifter output: {joints3D[0]} | {joints3D[1]} | {joints3D[2]} | {joints3D[3]}");

            // Axis remap for Unity coordinates
            for (int i = 0; i < joints3D.Length; i++)
                joints3D[i] = AxisRemap(joints3D[i], axis);

            // Drive the skeleton rig
            driver.UpdatePose(joints3D);
        }
        /*else
        {
            // Fallback: depth stub or do nothing until buffer fills
            joints3D = DepthProjectStub(lift2D);
        }*/
    }

    private AxisMode autoAxis = AxisMode.Raw;
    private bool calibrated = false;

    private void AutoCalibrate(Vector3[] joints)
    {
        // Test each axis mode
        foreach (AxisMode mode in System.Enum.GetValues(typeof(AxisMode)))
        {
            Vector3 head   = AxisRemap(joints[10], mode); // H36M index 10 = Head
            Vector3 pelvis = AxisRemap(joints[0], mode);  // H36M index 0 = Pelvis
            Vector3 lwrist = AxisRemap(joints[13], mode); // Left wrist
            Vector3 rwrist = AxisRemap(joints[16], mode); // Right wrist

            bool upright   = head.y > pelvis.y;
            bool leftRight = lwrist.x < rwrist.x;

            if (upright && leftRight)
            {
                autoAxis = mode;
                calibrated = true;
                Debug.Log($"### Auto-calibrated axis mode: {mode}");
                return;
            }
        }

        Debug.LogWarning("### Auto calibration failed: no valid axis mode found.");
    }

    
    private Vector3 AxisRemap(Vector3 p, AxisMode m)
    {
        // Convert H36M coords to Unity-friendly
        return m switch {
            AxisMode.Raw         => new Vector3(p.x, p.y, p.z),
            AxisMode.MirrorX     => new Vector3(-p.x, p.y, p.z),
            AxisMode.SwapYZ      => new Vector3(p.x, p.z, p.y),
            AxisMode.NegX_Z_NegY => new Vector3(-p.x, p.z, -p.y),
            _ => p
        };
    }

    private Vector3[] DepthProjectStub(Vector2[] in2D)
    {
        // Placeholder: returns zero Z. Replace with ARFoundation depth sampling per joint.
        var J = new Vector3[in2D.Length];
        for (int i = 0; i < J.Length; i++) J[i] = new Vector3(in2D[i].x, in2D[i].y, 0f);
        return J;
    }
}
