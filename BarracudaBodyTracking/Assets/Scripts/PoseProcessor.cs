using System.Globalization;
using UnityEngine;

// NEW: Burst + Jobs + Collections + Mathematics
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class PoseProcessor : MonoBehaviour
{
    [Tooltip("VNect model component for pose visualization")]
    public SkeletonModel skeletonModel;

    [Tooltip("Heatmap resolution")]
    public int heatMapCol = 28;

    [Tooltip("Input image size (width and height)")]
    public int inputImageSize = 448;

    [Header("Performance")]
    [Tooltip("Use Burst + Jobs for heatmap scan and joint position compute")]
    public bool useBurstJobs = true;

    #region Filtering Parameters
    [Header("Filtering Parameters")]
    [Tooltip("Kalman filter process noise parameter")]
    public float kalmanParamQ = 0.001f;

    [Tooltip("Kalman filter measurement noise parameter")]
    public float kalmanParamR = 0.0015f;

    [Tooltip("Enable low pass filter for smoothing")]
    public bool useLowPassFilter = true;

    [Tooltip("Low pass filter smoothing parameter")]
    [Range(0f, 1f)]
    public float lowPassParam = 0.1f;
    #endregion

    #region Private fields
    private SkeletonModel.JointPoint[] _jointPoints;
    private const int JointNum = 24;

    // Calculated dimensions
    private float _inputImageSizeHalf;
    private float _inputImageSizeF;
    private float _imageScale;

    // Native buffers for network outputs (CPU side)
    private NativeArray<float> _heatMap3D; // size: JointNum * heatMapCol^3
    private NativeArray<float> _offset3D;  // size: JointNum * heatMapCol^3 * 3

    // Job outputs
    private NativeArray<float3> _jointNow3D;   // per joint
    private NativeArray<float> _jointScores;   // per joint

    // Reusable constants
    private int _hm2; // heatMapCol^2
    #endregion

    #region Lifecycle
    private void OnDisable()
    {
        DisposeNatives();
    }

    private void OnDestroy()
    {
        DisposeNatives();
    }

    private void DisposeNatives()
    {
        if (_heatMap3D.IsCreated) _heatMap3D.Dispose();
        if (_offset3D.IsCreated) _offset3D.Dispose();
        if (_jointNow3D.IsCreated) _jointNow3D.Dispose();
        if (_jointScores.IsCreated) _jointScores.Dispose();
    }
    #endregion

    public void InitJoints()
    {
        _jointPoints = skeletonModel.Init();
    }

    /// <summary>
    /// Initialize calculation parameters and allocate native buffers
    /// </summary>
    public void InitializeParameters()
    {
        // Calculate scaling parameters
        _inputImageSizeF   = inputImageSize;
        _inputImageSizeHalf = _inputImageSizeF / 2f;
        _imageScale        = inputImageSize / (float)heatMapCol;

        _hm2 = heatMapCol * heatMapCol;

        // Allocate/resize natives
        int heatmapLen = JointNum * heatMapCol * heatMapCol * heatMapCol;     // J * Z * Y * X
        int offsetLen  = heatmapLen * 3;                                      // 3 axes

        if (_heatMap3D.IsCreated && _heatMap3D.Length != heatmapLen)
        {
            _heatMap3D.Dispose();
        }
        if (_offset3D.IsCreated && _offset3D.Length != offsetLen)
        {
            _offset3D.Dispose();
        }
        if (!_heatMap3D.IsCreated) _heatMap3D = new NativeArray<float>(heatmapLen, Allocator.Persistent);
        if (!_offset3D.IsCreated)  _offset3D  = new NativeArray<float>(offsetLen, Allocator.Persistent);

        if (_jointNow3D.IsCreated && _jointNow3D.Length != JointNum) _jointNow3D.Dispose();
        if (_jointScores.IsCreated && _jointScores.Length != JointNum) _jointScores.Dispose();
        if (!_jointNow3D.IsCreated) _jointNow3D = new NativeArray<float3>(JointNum, Allocator.Persistent);
        if (!_jointScores.IsCreated) _jointScores = new NativeArray<float>(JointNum, Allocator.Persistent);

        Debug.Log($"PoseProcessor (Burst={useBurstJobs}) initialized - Image:{inputImageSize} Heatmap:{heatMapCol}");
    }

    /// <summary>
    /// Push latest network outputs into NativeArrays (called each frame from SentisRunner)
    /// </summary>
    public void UploadNetworkOutputs(float[] offset3DManaged, float[] heatMap3DManaged)
    {
        // Defensive: ensure arrays are allocated
        if (!_heatMap3D.IsCreated || !_offset3D.IsCreated)
        {
            InitializeParameters();
        }

        // Length checks (optional but helpful in dev)
        if (heatMap3DManaged.Length != _heatMap3D.Length ||
            offset3DManaged.Length  != _offset3D.Length)
        {
            Debug.LogError($"UploadNetworkOutputs: size mismatch. Expected heat:{_heatMap3D.Length} off:{_offset3D.Length} but got heat:{heatMap3DManaged.Length} off:{offset3DManaged.Length}");
            return;
        }

        // Copy managed → native
        _heatMap3D.CopyFrom(heatMap3DManaged);
        _offset3D.CopyFrom(offset3DManaged);
    }

    #region Pose Prediction (Burst jobified)
    /// <summary>
    /// Predict 3D joint positions from network outputs
    /// </summary>
    public void PredictPose()
    {
        if (!useBurstJobs)
        {
            // Fallback to scalar main-thread version if you ever need it again
            ScalarPredictPose();
            return;
        }

        // Schedule Burst job to compute Now3D & score per joint
        var job = new FindMaxAndPosJob
        {
            heatMap3D = _heatMap3D,
            offset3D  = _offset3D,
            heatMapCol = heatMapCol,
            jointNum   = JointNum,
            imageScale = _imageScale,
            inputImageSizeHalf = _inputImageSizeHalf,
            hm2 = _hm2,
            jointNow3D = _jointNow3D,
            jointScores = _jointScores
        };

        JobHandle handle = job.Schedule(JointNum, 1);
        handle.Complete(); // results needed immediately for filtering & rig update

        // Map job outputs back to SkeletonModel joint points
        for (int j = 0; j < JointNum; j++)
        {
            float3 p = _jointNow3D[j];
            _jointPoints[j].Now3D = new Vector3(p.x, p.y, p.z);
            _jointPoints[j].score3D = _jointScores[j];
        }

        // Derived joints + filtering on main thread (cheap)
        CalculateDerivedJoints();
        ApplyFiltering();
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    private struct FindMaxAndPosJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> heatMap3D;  // (1, J*Z, Y, X) NCHW
        [ReadOnly] public NativeArray<float> offset3D;   // (1, 3*J*Z, Y, X) NCHW

        public int heatMapCol;         // = Z = Y = X = 28
        public int jointNum;           // = 24
        public float imageScale;       // inputImageSize / heatMapCol
        public float inputImageSizeHalf;
        public int hm2;                // heatMapCol^2

        [WriteOnly] public NativeArray<float3> jointNow3D; // per joint
        [WriteOnly] public NativeArray<float> jointScores; // per joint

        public void Execute(int j)
        {
            int hm = heatMapCol;
            int hm2Local = hm2;

            float best = float.NegativeInfinity;
            int bx = 0, by = 0, bz = 0;

            // Scan heatmap for this joint: channel = j*Z + z
            for (int z = 0; z < hm; z++)
            {
                int c = j * hm + z;
                int cBase = c * hm2Local;
                for (int y = 0; y < hm; y++)
                {
                    int row = cBase + y * hm;
                    for (int x = 0; x < hm; x++)
                    {
                        float v = heatMap3D[row + x];
                        if (v > best)
                        {
                            best = v; bx = x; by = y; bz = z;
                        }
                    }
                }
            }

            jointScores[j] = best;

            // Offsets indexing
            int channelsPerAxis = jointNum * hm; // 24*28 = 672

            int cX = j * hm + bz;                   // axis 0
            int cY = channelsPerAxis + j * hm + bz; // axis 1
            int cZ = 2 * channelsPerAxis + j * hm + bz; // axis 2

            int baseX = cX * hm2Local + by * hm + bx;
            int baseY = cY * hm2Local + by * hm + bx;
            int baseZ = cZ * hm2Local + by * hm + bx;

            float offX = offset3D[baseX];
            float offY = offset3D[baseY];
            float offZ = offset3D[baseZ];

            float3 pos;
            pos.x = (offX + 0.5f + bx) * imageScale - inputImageSizeHalf;
            // flip Y for Unity
            pos.y = inputImageSizeHalf - (offY + 0.5f + by) * imageScale;
            // center Z around middle slice (14 for 28)
            pos.z = (offZ + 0.5f + (bz - 14)) * imageScale;

            jointNow3D[j] = pos;
        }
    }
    #endregion

    #region Fallback scalar path (optional)
    private int c, cBase, row;
    private int maxXIndex = 0, maxYIndex = 0, maxZIndex = 0;
    private float v;

    private void ScalarPredictPose()
    {
        // In case you toggle off useBurstJobs for debugging.
        for (int j = 0; j < JointNum; j++)
        {
            _jointPoints[j].score3D = 0.0f;

            for (int z = 0; z < heatMapCol; z++)
            {
                c = j * heatMapCol + z;
                cBase = c * _hm2;
                for (int y = 0; y < heatMapCol; y++)
                {
                    row = cBase + y * heatMapCol;
                    for (int x = 0; x < heatMapCol; x++)
                    {
                        v = _heatMap3D[row + x];
                        if (v > _jointPoints[j].score3D)
                        {
                            _jointPoints[j].score3D = v;
                            maxXIndex = x; maxYIndex = y; maxZIndex = z;
                        }
                    }
                }
            }

            CalculateJointPositionScalar(j, maxXIndex, maxYIndex, maxZIndex);
        }

        CalculateDerivedJoints();
        ApplyFiltering();
    }

    private void CalculateJointPositionScalar(int jointIndex, int maxX, int maxY, int maxZ)
    {
        int channelsPerAxis = JointNum * heatMapCol;

        int cX = jointIndex * heatMapCol + maxZ;
        int cY = channelsPerAxis + jointIndex * heatMapCol + maxZ;
        int cZ = 2 * channelsPerAxis + jointIndex * heatMapCol + maxZ;

        int baseX = cX * _hm2 + maxY * heatMapCol + maxX;
        int baseY = cY * _hm2 + maxY * heatMapCol + maxX;
        int baseZ = cZ * _hm2 + maxY * heatMapCol + maxX;

        float offX = _offset3D[baseX];
        float offY = _offset3D[baseY];
        float offZ = _offset3D[baseZ];

        _jointPoints[jointIndex].Now3D.x =
            (offX + 0.5f + maxX) * _imageScale - _inputImageSizeHalf;

        _jointPoints[jointIndex].Now3D.y =
            _inputImageSizeHalf - (offY + 0.5f + maxY) * _imageScale;

        _jointPoints[jointIndex].Now3D.z =
            (offZ + 0.5f + (maxZ - 14)) * _imageScale;
    }
    #endregion

    #region Derived joints (unchanged from your logic)
    private Vector3 leftThigh, rightThigh, abdomenUpper, hipCenter, leftShoulder, rightShoulder;
    private Vector3 leftEar, rightEar, earCenter, neck, headVector, normalizedHeadVector, noseVector;

    private void CalculateDerivedJoints()
    {
        leftThigh     = _jointPoints[PositionIndex.lThighBend.Int()].Now3D;
        rightThigh    = _jointPoints[PositionIndex.rThighBend.Int()].Now3D;
        abdomenUpper  = _jointPoints[PositionIndex.abdomenUpper.Int()].Now3D;
        hipCenter     = (leftThigh + rightThigh) / 2f;
        _jointPoints[PositionIndex.hip.Int()].Now3D = (abdomenUpper + hipCenter) / 2f;

        leftShoulder = _jointPoints[PositionIndex.lShldrBend.Int()].Now3D;
        rightShoulder = _jointPoints[PositionIndex.rShldrBend.Int()].Now3D;
        _jointPoints[PositionIndex.neck.Int()].Now3D = (leftShoulder + rightShoulder) / 2f;

        leftEar  = _jointPoints[PositionIndex.lEar.Int()].Now3D;
        rightEar = _jointPoints[PositionIndex.rEar.Int()].Now3D;
        earCenter = (leftEar + rightEar) / 2f;
        neck = _jointPoints[PositionIndex.neck.Int()].Now3D;
        headVector = earCenter - neck;
        normalizedHeadVector = Vector3.Normalize(headVector);
        noseVector = _jointPoints[PositionIndex.Nose.Int()].Now3D - neck;
        _jointPoints[PositionIndex.head.Int()].Now3D =
            neck + normalizedHeadVector * Vector3.Dot(normalizedHeadVector, noseVector);
    }
    #endregion

    #region Filtering (unchanged from your logic)
    private void ApplyFiltering()
    {
        foreach (var jointPoint in _jointPoints)
            ApplyKalmanFilter(jointPoint);

        if (useLowPassFilter)
            ApplyLowPassFilter();
    }

    private void ApplyKalmanFilter(SkeletonModel.JointPoint jointPoint)
    {
        UpdateKalmanGain(jointPoint);

        jointPoint.Pos3D.x = jointPoint.X.x + (jointPoint.Now3D.x - jointPoint.X.x) * jointPoint.K.x;
        jointPoint.Pos3D.y = jointPoint.X.y + (jointPoint.Now3D.y - jointPoint.X.y) * jointPoint.K.y;
        jointPoint.Pos3D.z = jointPoint.X.z + (jointPoint.Now3D.z - jointPoint.X.z) * jointPoint.K.z;
        jointPoint.X = jointPoint.Pos3D;
    }

    private void UpdateKalmanGain(SkeletonModel.JointPoint jointPoint)
    {
        jointPoint.K.x = (jointPoint.P.x + kalmanParamQ) / (jointPoint.P.x + kalmanParamQ + kalmanParamR);
        jointPoint.K.y = (jointPoint.P.y + kalmanParamQ) / (jointPoint.P.y + kalmanParamQ + kalmanParamR);
        jointPoint.K.z = (jointPoint.P.z + kalmanParamQ) / (jointPoint.P.z + kalmanParamQ + kalmanParamR);

        jointPoint.P.x = kalmanParamR * (jointPoint.P.x + kalmanParamQ) / (kalmanParamR + jointPoint.P.x + kalmanParamQ);
        jointPoint.P.y = kalmanParamR * (jointPoint.P.y + kalmanParamQ) / (kalmanParamR + jointPoint.P.y + kalmanParamQ);
        jointPoint.P.z = kalmanParamR * (jointPoint.P.z + kalmanParamQ) / (kalmanParamR + jointPoint.P.z + kalmanParamQ);
    }

    private void ApplyLowPassFilter()
    {
        foreach (var jointPoint in _jointPoints)
        {
            jointPoint.PrevPos3D[0] = jointPoint.Pos3D;
            for (int i = 1; i < jointPoint.PrevPos3D.Length; i++)
            {
                jointPoint.PrevPos3D[i] = jointPoint.PrevPos3D[i] * lowPassParam +
                                          jointPoint.PrevPos3D[i - 1] * (1f - lowPassParam);
            }
            jointPoint.Pos3D = jointPoint.PrevPos3D[jointPoint.PrevPos3D.Length - 1];
        }
    }
    #endregion

    /// <summary>Get current joint points</summary>
    public SkeletonModel.JointPoint[] GetJointPoints() => _jointPoints;
}
