using System.Globalization;
using UnityEngine;

public class PoseProcessor : MonoBehaviour
{
    [Tooltip("VNect model component for pose visualization")]
    public SkeletonModel skeletonModel;
    [Tooltip("Heatmap resolution")]
    public int heatMapCol = 28;
    [Tooltip("Input image size (width and height)")]
    public int inputImageSize = 448;
    
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
    private int _heatMapColSquared;
    private int _heatMapColCubed;
    private float _imageScale;
    #endregion
    
    // Buffer arrays
    [HideInInspector] public float[] heatMap3D;
    [HideInInspector] public float[] offset3D;
    
    public void InitJoints()
    {
        // Initialize joint points
        _jointPoints = skeletonModel.Init();
    }
    
    /// <summary>
    /// Initialize calculation parameters and buffer arrays
    /// </summary>
    public void InitializeParameters()
    {
        // Calculate derived dimensions
        _heatMapColSquared = heatMapCol * heatMapCol;
        _heatMapColCubed = heatMapCol * heatMapCol * heatMapCol;
        
        // Initialize buffer arrays
        heatMap3D = new float[JointNum * _heatMapColCubed];
        offset3D = new float[JointNum * _heatMapColCubed * 3];
        
        // Calculate scaling parameters
        _inputImageSizeF = inputImageSize;
        _inputImageSizeHalf = _inputImageSizeF / 2f;
        _imageScale = inputImageSize / (float)heatMapCol;
        
        Debug.Log($"PoseProcessor parameters initialized - Image size: {inputImageSize}, Heatmap: {heatMapCol}");
    }
    
    #region Pose Prediction
    
    /// <summary>
    /// Predict 3D joint positions from network outputs
    /// </summary>
    public void PredictPose()
    {
        // Find maximum activation for each joint
        for (int j = 0; j < JointNum; j++)
        {
            FindMaxActivation(j);
        }
        
        // Calculate derived joint positions
        CalculateDerivedJoints();
        
        // Apply filtering
        ApplyFiltering();
    }
    
    /// <summary>
    /// Find maximum activation position for a specific joint
    /// </summary>
    private int maxXIndex = 0, maxYIndex = 0, maxZIndex = 0;
    private int c, cBase, row;
    private float v;
    private void FindMaxActivation(int jointIndex)
    {
        _jointPoints[jointIndex].score3D = 0.0f;
        
        // NCHW: (1, 672, 28, 28) where channel = j*heatMapCol + z
        for (int z = 0; z < heatMapCol; z++)
        {
             c = jointIndex * heatMapCol + z;                  // channel index
             cBase = c * _heatMapColSquared;                   // stride = H*W
            for (int y = 0; y < heatMapCol; y++)
            {
                 row = cBase + y * heatMapCol;
                for (int x = 0; x < heatMapCol; x++)
                {
                    v = heatMap3D[row + x];
                    if (v > _jointPoints[jointIndex].score3D)
                    {
                        _jointPoints[jointIndex].score3D = v;
                        maxXIndex = x; maxYIndex = y; maxZIndex = z;
                    }
                }
            }
        }
        
        // Calculate 3D position from offsets
        CalculateJointPosition(jointIndex, maxXIndex, maxYIndex, maxZIndex);
    }

    /// <summary>
    /// Calculate final joint position from maximum activation indices
    /// </summary>
    private int channelsPerAxis, cX, cY, cZ, baseX, baseY, baseZ;
    float offX, offY, offZ;
    private void CalculateJointPosition(int jointIndex, int maxX, int maxY, int maxZ)
    {
        // NCHW: (1, 2016, 28, 28) = 3 * (24 * 28) channels
         channelsPerAxis = JointNum * heatMapCol; // 24*28 = 672

         cX = /* axis 0 */ 0 * channelsPerAxis + jointIndex * heatMapCol + maxZ;
         cY = /* axis 1 */ 1 * channelsPerAxis + jointIndex * heatMapCol + maxZ;
         cZ = /* axis 2 */ 2 * channelsPerAxis + jointIndex * heatMapCol + maxZ;

         baseX = cX * _heatMapColSquared + maxY * heatMapCol + maxX;
         baseY = cY * _heatMapColSquared + maxY * heatMapCol + maxX;
         baseZ = cZ * _heatMapColSquared + maxY * heatMapCol + maxX;

         offX = offset3D[baseX];
         offY = offset3D[baseY];
         offZ = offset3D[baseZ];

        _jointPoints[jointIndex].Now3D.x =
            (offX + 0.5f + maxX) * _imageScale - _inputImageSizeHalf;

        // flip Y for Unity
        _jointPoints[jointIndex].Now3D.y =
            _inputImageSizeHalf - (offY + 0.5f + maxY) * _imageScale;

        _jointPoints[jointIndex].Now3D.z =
            (offZ + 0.5f + (maxZ - 14)) * _imageScale;
    }

    /// <summary>
    /// Calculate derived joint positions (hip, neck, head)
    /// </summary>
    private Vector3 leftThigh, rightThigh, abdomenUpper, hipCenter, leftShoulder, rightShoulder;

    private Vector3 leftEar, rightEar, earCenter, neck, headVector, normalizedHeadVector, noseVector;
    private void CalculateDerivedJoints()
    {
        // Calculate hip location
         leftThigh = _jointPoints[PositionIndex.lThighBend.Int()].Now3D;
         rightThigh = _jointPoints[PositionIndex.rThighBend.Int()].Now3D;
         abdomenUpper = _jointPoints[PositionIndex.abdomenUpper.Int()].Now3D;
         hipCenter = (leftThigh + rightThigh) / 2f;
        _jointPoints[PositionIndex.hip.Int()].Now3D = (abdomenUpper + hipCenter) / 2f;
        
        // Calculate neck location
         leftShoulder = _jointPoints[PositionIndex.lShldrBend.Int()].Now3D;
         rightShoulder = _jointPoints[PositionIndex.rShldrBend.Int()].Now3D;
        _jointPoints[PositionIndex.neck.Int()].Now3D = (leftShoulder + rightShoulder) / 2f;
        
        // Calculate head location
         leftEar = _jointPoints[PositionIndex.lEar.Int()].Now3D;
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
    
    #region Filtering
    
    /// <summary>
    /// Apply Kalman and low-pass filtering to joint positions
    /// </summary>
    private void ApplyFiltering()
    {
        // Apply Kalman filter to all joints
        foreach (var jointPoint in _jointPoints)
        {
            ApplyKalmanFilter(jointPoint);
        }
        
        // Apply low-pass filter if enabled
        if (useLowPassFilter)
        {
            ApplyLowPassFilter();
        }
    }
    
    /// <summary>
    /// Apply Kalman filter to a single joint point
    /// </summary>
    private void ApplyKalmanFilter(SkeletonModel.JointPoint jointPoint)
    {
        // Measurement update
        UpdateKalmanGain(jointPoint);
        
        // State update
        jointPoint.Pos3D.x = jointPoint.X.x + (jointPoint.Now3D.x - jointPoint.X.x) * jointPoint.K.x;
        jointPoint.Pos3D.y = jointPoint.X.y + (jointPoint.Now3D.y - jointPoint.X.y) * jointPoint.K.y;
        jointPoint.Pos3D.z = jointPoint.X.z + (jointPoint.Now3D.z - jointPoint.X.z) * jointPoint.K.z;
        jointPoint.X = jointPoint.Pos3D;
    }
    
    /// <summary>
    /// Update Kalman gain and covariance
    /// </summary>
    private void UpdateKalmanGain(SkeletonModel.JointPoint jointPoint)
    {
        // Calculate Kalman gain for each axis
        jointPoint.K.x = (jointPoint.P.x + kalmanParamQ) / (jointPoint.P.x + kalmanParamQ + kalmanParamR);
        jointPoint.K.y = (jointPoint.P.y + kalmanParamQ) / (jointPoint.P.y + kalmanParamQ + kalmanParamR);
        jointPoint.K.z = (jointPoint.P.z + kalmanParamQ) / (jointPoint.P.z + kalmanParamQ + kalmanParamR);
        
        // Update covariance
        jointPoint.P.x = kalmanParamR * (jointPoint.P.x + kalmanParamQ) / (kalmanParamR + jointPoint.P.x + kalmanParamQ);
        jointPoint.P.y = kalmanParamR * (jointPoint.P.y + kalmanParamQ) / (kalmanParamR + jointPoint.P.y + kalmanParamQ);
        jointPoint.P.z = kalmanParamR * (jointPoint.P.z + kalmanParamQ) / (kalmanParamR + jointPoint.P.z + kalmanParamQ);
    }
    
    /// <summary>
    /// Apply low-pass filter to all joint points
    /// </summary>
    private void ApplyLowPassFilter()
    {
        foreach (var jointPoint in _jointPoints)
        {
            // Update position history
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
    
    /// <summary>
    /// Get current joint points
    /// </summary>
    public SkeletonModel.JointPoint[] GetJointPoints() => _jointPoints;
}
