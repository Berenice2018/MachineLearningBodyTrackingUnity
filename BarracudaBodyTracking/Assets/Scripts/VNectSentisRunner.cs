using UnityEngine;
using Unity.Sentis;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;

/// <summary>
/// VNect pose estimation runner using Unity Sentis 2.1.3
/// Processes video input to detect 3D human pose joint points in real-time
/// </summary>
public class VNectSentisRunner : MonoBehaviour
{
    #region Public Configuration
    [Header("Model Configuration")]
    [Tooltip("Neural network model asset for pose estimation")]
    public ModelAsset modelAsset;
    
    [Tooltip("Computation backend type for model inference")]
    public BackendType backendType = BackendType.GPUCompute;
    
    [Tooltip("Enable verbose logging for debugging")]
    public bool verbose = true;

    [Header("Components")]
    [Tooltip("VNect model component for pose visualization")]
    public VNectModel vNectModel;
    
    [Tooltip("Video capture component for input")]
    public VideoCapture videoCapture;
    
    [Tooltip("Pose retargeter component for avatar animation")]
    public VNectPoseRetargeter poseRetargeter;

    [Header("Input Parameters")]
    [Tooltip("Input image size (width and height in pixels)")]
    public int inputImageSize = 448;
    
    [Tooltip("Heat map column count")]
    public int heatMapCol = 28;

    [Header("Filtering Parameters")]
    [Tooltip("Kalman filter process noise parameter")]
    public float kalmanParamQ = 0.001f;
    
    [Tooltip("Kalman filter measurement noise parameter")]
    public float kalmanParamR = 0.0015f;
    
    [Tooltip("Enable low pass filter for smoothing")]
    public bool useLowPassFilter = false;
    
    [Tooltip("Low pass filter parameter (0-1, higher = more smoothing)")]
    [Range(0f, 1f)]
    public float lowPassParam = 0.5f;

    [Header("Initialization")]
    [Tooltip("Initial image for model warm-up")]
    public Texture2D initImg;
    
    [Tooltip("Wait time before starting real-time processing")]
    public float waitTimeModelLoad = 2f;
    #endregion

    #region Private Fields - Model and Worker
    private Model m_RuntimeModel;
    private Worker m_Worker;
    private bool m_IsModelLoaded = false;
    private string[] m_InputNames;   // length 1..3 depending on model

    #endregion

    #region Private Fields - Joint Processing
    /// <summary>Coordinates of joint points</summary>
    private VNectModel.JointPoint[] m_JointPoints;
    
    /// <summary>Number of joint points detected by the model</summary>
    private const int JOINT_NUM = 24;
    
    /// <summary>Number of joints in 2D space (x, y coordinates)</summary>
    private const int JOINT_NUM_2D = JOINT_NUM * 2;
    
    /// <summary>Number of joints in 3D space (x, y, z coordinates)</summary>
    private const int JOINT_NUM_3D = JOINT_NUM * 3;
    #endregion

    #region Private Fields - Image Processing
    /// <summary>Input image size as float for calculations</summary>
    private float m_InputImageSizeF;
    
    /// <summary>Half of input image size for centering calculations</summary>
    private float m_InputImageSizeHalf;
    
    /// <summary>Scale factor from heat map to image coordinates</summary>
    private float m_ImageScale;
    
    /// <summary>Unit size in heat map coordinates</summary>
    private float m_Unit;
    #endregion

    #region Private Fields - Heat Map Dimensions
    /// <summary>Heat map area (HeatMapCol * HeatMapCol)</summary>
    private int m_HeatMapColSquared;
    
    /// <summary>Heat map volume (HeatMapCol^3)</summary>
    private int m_HeatMapColCube;
    
    /// <summary>HeatMapCol * JOINT_NUM for indexing</summary>
    private int m_HeatMapColJointNum;
    
    /// <summary>Linear cube offset for 3D coordinate calculations</summary>
    private int m_CubeOffsetLinear;
    
    /// <summary>Squared cube offset for 3D coordinate calculations</summary>
    private int m_CubeOffsetSquared;
    #endregion

    #region Private Fields - Model Tensors and Processing
    /// <summary>Current input tensor</summary>
    private Tensor<float> m_CurrentInput;
    
    /// <summary>Ring buffer for temporal input frames</summary>
    private readonly Dictionary<string, Tensor<float>> m_InputTensors = new Dictionary<string, Tensor<float>>();
    
    /// <summary>Model output tensors</summary>
    private Tensor<float> m_Output3DOffset;
    private Tensor<float> m_Output3DHeatMap;
    
    #region Private Fields - Heat Map Data Processing
    /// <summary>Buffer memory has 2D heat map</summary>
    private float[] m_HeatMap2D;

    /// <summary>Buffer memory has offset 2D</summary>
    private float[] m_Offset2D;
    
    /// <summary>Buffer memory has 3D heat map</summary>
    private float[] m_HeatMap3D;
    
    /// <summary>Buffer memory has 3D offset</summary>
    private float[] m_Offset3D;
    #endregion
    
    /// <summary>Input names for the temporal model</summary>
    private const string INPUT_NAME_1 = "input.1";  // Current frame
    private const string INPUT_NAME_2 = "input.4";  // Previous frame 1
    private const string INPUT_NAME_3 = "input.7";  // Previous frame 2
    
    /// <summary>Processing lock to prevent concurrent model execution</summary>
    private bool m_IsProcessingLocked = true;
    
    /// <summary>Flag to track if model is currently executing</summary>
    private bool m_IsExecuting = false;
    #endregion

    #region Unity Lifecycle
    /// <summary>
    /// Initialize the pose estimation system
    /// </summary>
    private void Start()
    {
        InitializeSystem();
        LoadModelAsync();
    }

    /// <summary>
    /// Update pose estimation each frame
    /// </summary>
    private void Update()
    {
        if (!m_IsProcessingLocked && !m_IsExecuting)
        {
            UpdatePoseEstimation();
        }
    }

    /// <summary>
    /// Clean up resources when destroyed
    /// </summary>
    private void OnDestroy()
    {
        CleanupResources();
    }
    #endregion

    #region System Initialization
    /// <summary>
    /// Initialize system parameters and allocate memory
    /// </summary>
    private void InitializeSystem()
    {
        // Disable screen sleep for continuous processing
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
        
        // Calculate derived dimensions
        m_HeatMapColSquared = heatMapCol * heatMapCol;
        m_HeatMapColCube = heatMapCol * heatMapCol * heatMapCol;
        m_HeatMapColJointNum = heatMapCol * JOINT_NUM;
        m_CubeOffsetLinear = heatMapCol * JOINT_NUM_3D;
        m_CubeOffsetSquared = m_HeatMapColSquared * JOINT_NUM_3D;
        
        // Allocate heat map processing buffers
        m_HeatMap2D = new float[JOINT_NUM * m_HeatMapColSquared];
        m_Offset2D = new float[JOINT_NUM * m_HeatMapColSquared * 2];
        m_HeatMap3D = new float[JOINT_NUM * m_HeatMapColCube];
        m_Offset3D = new float[JOINT_NUM * m_HeatMapColCube * 3];
        
        // Calculate image processing parameters
        m_Unit = 1f / (float)heatMapCol;
        m_InputImageSizeF = inputImageSize;
        m_InputImageSizeHalf = m_InputImageSizeF * 0.5f;
        m_ImageScale = inputImageSize / (float)heatMapCol;
        
        // Initialize input tensor dictionary
        m_InputTensors[INPUT_NAME_1] = null;
        m_InputTensors[INPUT_NAME_2] = null;
        m_InputTensors[INPUT_NAME_3] = null;
        
        if (verbose)
        {
            Debug.Log($"VNect Sentis Runner initialized - Image Size: {inputImageSize}, HeatMap: {heatMapCol}x{heatMapCol}");
        }
    }

    /// <summary>
    /// Load and initialize the neural network model
    /// </summary>
    private async void LoadModelAsync()
    {
        try
        {
            // Load model from asset
            m_RuntimeModel = ModelLoader.Load(modelAsset);
            
            // Create worker with specified backend
            m_Worker = new Worker(m_RuntimeModel, backendType);
            
            m_RuntimeModel = ModelLoader.Load(modelAsset);
            m_Worker = new Worker(m_RuntimeModel, backendType);

// Cache actual input names
            int inCount = m_RuntimeModel.inputs.Count;
            m_InputNames = new string[inCount];
            for (int i = 0; i < inCount; i++)
            {
                m_InputNames[i] = m_RuntimeModel.inputs[i].name;
                if (verbose) Debug.Log($"Model input[{i}]: {m_InputNames[i]}");
            }

// Init the dict with the actual keys
            m_InputTensors.Clear();
            for (int i = 0; i < inCount; i++)
                m_InputTensors[m_InputNames[i]] = null;

            
            if (verbose)
            {
                Debug.Log($"Model loaded successfully with {backendType} backend");
                Debug.Log($"Model inputs: {m_RuntimeModel.inputs.Count}, outputs: {m_RuntimeModel.outputs.Count}");
            }
            
            m_IsModelLoaded = true;
            
            // Initialize model asynchronously
            await InitializeModelAsync();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to load model: {e.Message}");
        }
    }
    #endregion

    #region Model Initialization
    /// <summary>
    /// Warm up the model and initialize pose tracking
    /// </summary>
    private async Task InitializeModelAsync()
    {
        if (!m_IsModelLoaded || initImg == null)
        {
            Debug.LogError("Model not loaded or initial image not set");
            return;
        }

        try
        {
            // Create initial input tensors for warm-up
            var initTensor = CreateInputTensor(initImg);

            if (m_InputNames.Length >= 3)
            {
                m_InputTensors[m_InputNames[0]] = initTensor;                      // current
                m_InputTensors[m_InputNames[1]] = CreateInputTensor(initImg);      // prev1
                m_InputTensors[m_InputNames[2]] = CreateInputTensor(initImg);      // prev2
            }
            else if (m_InputNames.Length == 1)
            {
                // Single-frame model: set only once
                m_InputTensors[m_InputNames[0]] = initTensor;
            }
            else
            {
                Debug.LogError($"Unexpected input count: {m_InputNames.Length}");
                return;
            }

            // Execute model for warm-up
            await ExecuteModelAsync();

            // Initialize VNect model joint points
            m_JointPoints = vNectModel.Init();
            
            // Run initial pose prediction
            PredictPose();

            // Wait for specified time
            await Task.Delay((int)(waitTimeModelLoad * 1000));

            // Initialize video capture
            if (videoCapture != null)
            {
                videoCapture.Init(inputImageSize, inputImageSize);
            }

            // Unlock processing
            m_IsProcessingLocked = false;
            
            if (verbose)
            {
                Debug.Log("VNect model initialization completed");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Model initialization failed: {e.Message}");
        }
    }
    #endregion

    #region Input Processing
    /// <summary>
    /// Create input tensor from texture
    /// </summary>
    /// <param name="texture">Source texture</param>
    /// <returns>Formatted input tensor</returns>
    private Tensor<float> CreateInputTensor(Texture texture)
    {
        return TextureConverter.ToTensor(
            texture, inputImageSize, inputImageSize, 3) as Tensor<float>;
    }

    /// <summary>
    /// Update pose estimation with new video frame
    /// </summary>
    private void UpdatePoseEstimation()
    {
        if (videoCapture?.MainTexture == null)
            return;

        // Create new input tensor from current video frame
        m_CurrentInput = CreateInputTensor(videoCapture.MainTexture);

        // Update temporal input ring buffer
        UpdateInputTensorBuffer();

        // Execute model asynchronously
        _ = ExecuteModelAsync();
    }

    /// <summary>
    /// Update the temporal input tensor ring buffer
    /// </summary>
    private void UpdateInputTensorBuffer()
    {
        var mainTex = videoCapture.MainTexture;
        //if (mainTex == null) return;

        if (m_InputNames.Length == 1)
        {
            DisposeTensorSafely(m_InputTensors[m_InputNames[0]]);
            m_InputTensors[m_InputNames[0]] = CreateInputTensor(mainTex);
            return;
        }

        // 3-frame temporal model
        if (m_InputTensors[m_InputNames[0]] == null)
        {
            m_InputTensors[m_InputNames[0]] = CreateInputTensor(mainTex);
            m_InputTensors[m_InputNames[1]] = CreateInputTensor(mainTex);
            m_InputTensors[m_InputNames[2]] = CreateInputTensor(mainTex);
        }
        else
        {
            DisposeTensorSafely(m_InputTensors[m_InputNames[2]]);
            m_InputTensors[m_InputNames[2]] = m_InputTensors[m_InputNames[1]];
            m_InputTensors[m_InputNames[1]] = m_InputTensors[m_InputNames[0]];
            m_InputTensors[m_InputNames[0]] = CreateInputTensor(mainTex);
        }
    }

    #endregion

    #region Model Execution
    /// <summary>
    /// Execute the neural network model asynchronously
    /// </summary>
    private async Task ExecuteModelAsync()
    {
        m_IsExecuting = true;
        try
        {
            // Feed inputs
            for (int i = 0; i < m_InputNames.Length; i++)
                m_Worker.SetInput(m_InputNames[i], m_InputTensors[m_InputNames[i]]);

            // Run and sync
            m_Worker.Schedule();

            // Read outputs (indices 2 and 3 match your existing code)
            var tOff = m_Worker.PeekOutput(m_RuntimeModel.outputs[2].name) as Tensor<float>;
            var tHm  = m_Worker.PeekOutput(m_RuntimeModel.outputs[3].name) as Tensor<float>;

            if (tOff == null || tHm == null)
            {
                Debug.LogError("Expected 3D outputs are missing.");
                return;
            }

            m_Offset3D  = tOff.DownloadToArray();  // float[]
            m_HeatMap3D = tHm.DownloadToArray();   // float[]

            PredictPose(); // proceed
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Model execution failed: {e.Message}");
        }
        finally { m_IsExecuting = false; }
    }


    /// <summary>
    /// Retrieve and process model outputs
    /// </summary>
    private void RetrieveModelOutputs()
    {
        try
        {
            // Get output tensors - check which indices correspond to your model outputs
            // You may need to adjust these indices based on your specific VNect model
            
            if (m_RuntimeModel.outputs.Count >= 4)
            {
                // Typical VNect model output order:
                // outputs[0] - 2D heatmap (optional)
                // outputs[1] - 2D offset (optional)  
                // outputs[2] - 3D offset
                // outputs[3] - 3D heatmap
                
                var output3DOffset = m_Worker.PeekOutput(m_RuntimeModel.outputs[2].name) as Tensor<float>;
                var output3DHeatMap = m_Worker.PeekOutput(m_RuntimeModel.outputs[3].name) as Tensor<float>;

                // Download data to CPU
                if (output3DOffset != null && output3DHeatMap != null)
                {
                    m_Offset3D = output3DOffset.DownloadToArray();
                    m_HeatMap3D = output3DHeatMap.DownloadToArray();
                    
                    if (verbose && m_FrameCounter % 60 == 0) // Log every 60 frames
                    {
                        Debug.Log($"Retrieved outputs - 3D Offset: {m_Offset3D.Length}, 3D HeatMap: {m_HeatMap3D.Length}");
                        Debug.Log($"Expected 3D Offset size: {JOINT_NUM * m_HeatMapColCube * 3}");
                        Debug.Log($"Expected 3D HeatMap size: {JOINT_NUM * m_HeatMapColCube}");
                    }
                }
                else
                {
                    Debug.LogWarning("Failed to retrieve 3D output tensors");
                }
                
                // Optionally get 2D outputs if available
                if (m_RuntimeModel.outputs.Count >= 2)
                {
                    var output2DOffset = m_Worker.PeekOutput(m_RuntimeModel.outputs[1].name) as Tensor<float>;
                    var output2DHeatMap = m_Worker.PeekOutput(m_RuntimeModel.outputs[0].name) as Tensor<float>;
                    
                    if (output2DOffset != null && output2DHeatMap != null)
                    {
                        m_Offset2D = output2DOffset.DownloadToArray();
                        m_HeatMap2D = output2DHeatMap.DownloadToArray();
                    }
                }
            }
            else
            {
                Debug.LogError($"Model has insufficient outputs. Expected at least 4, got {m_RuntimeModel.outputs.Count}");
                
                // Log available outputs for debugging
                for (int i = 0; i < m_RuntimeModel.outputs.Count; i++)
                {
                    var output = m_RuntimeModel.outputs[i];
                    Debug.Log($"Output {i}: {output.name}, Index: {string.Join(",", output.index)}");
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to retrieve model outputs: {e.Message}");
        }
    }
    
    /// <summary>Frame counter for debugging</summary>
    private int m_FrameCounter = 0;
    #endregion

    #region Pose Prediction
    /// <summary>
    /// Predict 3D joint positions from model outputs using heatmap analysis
    /// This mirrors the original Barracuda implementation more closely
    /// </summary>
    private void PredictPose()
    {
        if (m_HeatMap3D == null || m_Offset3D == null || m_JointPoints == null)
        {
            if (verbose && m_FrameCounter % 60 == 0)
            {
                Debug.LogWarning("Missing pose prediction data");
            }
            return;
        }

        m_FrameCounter++;

        // Process each joint using the original heatmap analysis method
        for (int jointIndex = 0; jointIndex < JOINT_NUM; jointIndex++)
        {
            FindMaxActivationIn3DHeatmap(jointIndex, out int maxX, out int maxY, out int maxZ, out float maxScore);
            
            // Store confidence score
            m_JointPoints[jointIndex].score3D = maxScore;
            
            // Calculate 3D position using offset data (matching original implementation)
            Calculate3DJointPositionFromHeatmap(jointIndex, maxX, maxY, maxZ);
        }

        // Calculate derived joint positions (hip, neck, head) - from original implementation
        CalculateDerivedJointsOriginal();

        // Apply filtering
        ApplyKalmanFilter();
        
        if (useLowPassFilter)
        {
            ApplyLowPassFilter();
        }
        
        /*/ Apply pose to avatar through retargeter
        if (poseRetargeter != null)// && poseRetargeter.IsInitialized
        {
            poseRetargeter.ApplyPose(m_JointPoints);
        }*/
        
        // Debug logging
        if (verbose && m_FrameCounter % 60 == 0)
        {
            LogPoseDebugInfo();
        }
    }

    /// <summary>
    /// Find maximum activation in 3D heatmap for specific joint (original algorithm)
    /// </summary>
    private void FindMaxActivationIn3DHeatmap(int jointIndex, out int maxX, out int maxY, out int maxZ, out float maxScore)
    {
        maxX = maxY = maxZ = 0;
        maxScore = 0.0f;
        
        int jointOffset = jointIndex * heatMapCol;
        
        // Scan through 3D heatmap volume (matching original nested loop structure)
        for (int z = 0; z < heatMapCol; z++)
        {
            int zIndex = jointOffset + z;
            
            for (int y = 0; y < heatMapCol; y++)
            {
                int yIndex = y * m_HeatMapColSquared * JOINT_NUM + zIndex;
                
                for (int x = 0; x < heatMapCol; x++)
                {
                    int dataIndex = yIndex + x * m_HeatMapColJointNum;
                    
                    // Check bounds to prevent array access errors
                    if (dataIndex >= 0 && dataIndex < m_HeatMap3D.Length)
                    {
                        float value = m_HeatMap3D[dataIndex];
                        
                        if (value > maxScore)
                        {
                            maxScore = value;
                            maxX = x;
                            maxY = y;
                            maxZ = z;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Calculate 3D position from heatmap indices and offset (original algorithm)
    /// </summary>
    private void Calculate3DJointPositionFromHeatmap(int jointIndex, int maxX, int maxY, int maxZ)
    {
        try
        {
            int baseIndex = maxY * m_CubeOffsetSquared + maxX * m_CubeOffsetLinear;
            
            // Calculate X coordinate (original formula)
            int xOffsetIndex = baseIndex + jointIndex * heatMapCol + maxZ;
            if (xOffsetIndex >= 0 && xOffsetIndex < m_Offset3D.Length)
            {
                float offsetX = m_Offset3D[xOffsetIndex];
                m_JointPoints[jointIndex].Now3D.x = (offsetX + 0.5f + maxX) * m_ImageScale - m_InputImageSizeHalf;
            }
            
            // Calculate Y coordinate (original formula with inversion)
            int yOffsetIndex = baseIndex + (jointIndex + JOINT_NUM) * heatMapCol + maxZ;
            if (yOffsetIndex >= 0 && yOffsetIndex < m_Offset3D.Length)
            {
                float offsetY = m_Offset3D[yOffsetIndex];
                m_JointPoints[jointIndex].Now3D.y = m_InputImageSizeHalf - (offsetY + 0.5f + maxY) * m_ImageScale;
            }
            
            // Calculate Z coordinate (original formula)
            int zOffsetIndex = baseIndex + (jointIndex + JOINT_NUM_2D) * heatMapCol + maxZ;
            if (zOffsetIndex >= 0 && zOffsetIndex < m_Offset3D.Length)
            {
                float offsetZ = m_Offset3D[zOffsetIndex];
                m_JointPoints[jointIndex].Now3D.z = (offsetZ + 0.5f + (maxZ - 14)) * m_ImageScale;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Error calculating joint {jointIndex} position: {e.Message}");
        }
    }

    /// <summary>
    /// Calculate derived joints exactly as in original implementation
    /// </summary>
    private void CalculateDerivedJointsOriginal()
    {
        try
        {
            // Calculate hip location (original algorithm)
            Vector3 leftThigh = m_JointPoints[PositionIndex.lThighBend.Int()].Now3D;
            Vector3 rightThigh = m_JointPoints[PositionIndex.rThighBend.Int()].Now3D;
            Vector3 hipCenter = (leftThigh + rightThigh) / 2f;
            m_JointPoints[PositionIndex.hip.Int()].Now3D = 
                (m_JointPoints[PositionIndex.abdomenUpper.Int()].Now3D + hipCenter) / 2f;

            // Calculate neck location (original algorithm)
            Vector3 leftShoulder = m_JointPoints[PositionIndex.lShldrBend.Int()].Now3D;
            Vector3 rightShoulder = m_JointPoints[PositionIndex.rShldrBend.Int()].Now3D;
            m_JointPoints[PositionIndex.neck.Int()].Now3D = (leftShoulder + rightShoulder) / 2f;

            // Calculate head location (original algorithm)
            Vector3 leftEar = m_JointPoints[PositionIndex.lEar.Int()].Now3D;
            Vector3 rightEar = m_JointPoints[PositionIndex.rEar.Int()].Now3D;
            Vector3 earCenter = (leftEar + rightEar) / 2f;
            Vector3 neckPos = m_JointPoints[PositionIndex.neck.Int()].Now3D;
            
            Vector3 headVector = earCenter - neckPos;
            Vector3 normalizedHeadVector = Vector3.Normalize(headVector);
            Vector3 noseVector = m_JointPoints[PositionIndex.Nose.Int()].Now3D - neckPos;
            
            m_JointPoints[PositionIndex.head.Int()].Now3D = 
                neckPos + normalizedHeadVector * Vector3.Dot(normalizedHeadVector, noseVector);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Error calculating derived joints: {e.Message}");
        }
    }
    
    /// <summary>
    /// Log pose debug information
    /// </summary>
    private void LogPoseDebugInfo()
    {
        if (m_JointPoints == null) return;
        
        // Log a few key joint positions and scores
        var hip = m_JointPoints[PositionIndex.hip.Int()];
        var neck = m_JointPoints[PositionIndex.neck.Int()];
        var head = m_JointPoints[PositionIndex.head.Int()];
        
        Debug.Log($"VNect Pose Debug - Hip: {hip.Now3D} (score: {hip.score3D:F3}), " +
                  $"Neck: {neck.Now3D} (score: {neck.score3D:F3}), " +
                  $"Head: {head.Now3D} (score: {head.score3D:F3})");
    }
    #endregion

    #region Filtering
    /// <summary>
    /// Apply Kalman filter to all joint points for temporal smoothing
    /// </summary>
    private void ApplyKalmanFilter()
    {
        foreach (var jointPoint in m_JointPoints)
        {
            ApplyKalmanFilterToJoint(jointPoint);
        }
    }

    /// <summary>
    /// Apply Kalman filter to a single joint point
    /// </summary>
    private void ApplyKalmanFilterToJoint(VNectModel.JointPoint jointPoint)
    {
        // Measurement update
        UpdateKalmanMeasurement(jointPoint);
        
        // State update
        jointPoint.Pos3D.x = jointPoint.X.x + (jointPoint.Now3D.x - jointPoint.X.x) * jointPoint.K.x;
        jointPoint.Pos3D.y = jointPoint.X.y + (jointPoint.Now3D.y - jointPoint.X.y) * jointPoint.K.y;
        jointPoint.Pos3D.z = jointPoint.X.z + (jointPoint.Now3D.z - jointPoint.X.z) * jointPoint.K.z;
        jointPoint.X = jointPoint.Pos3D;
    }

    /// <summary>
    /// Update Kalman filter measurement parameters
    /// </summary>
    private void UpdateKalmanMeasurement(VNectModel.JointPoint jointPoint)
    {
        float denomX = jointPoint.P.x + kalmanParamQ + kalmanParamR;
        float denomY = jointPoint.P.y + kalmanParamQ + kalmanParamR;
        float denomZ = jointPoint.P.z + kalmanParamQ + kalmanParamR;
        
        jointPoint.K.x = (jointPoint.P.x + kalmanParamQ) / denomX;
        jointPoint.K.y = (jointPoint.P.y + kalmanParamQ) / denomY;
        jointPoint.K.z = (jointPoint.P.z + kalmanParamQ) / denomZ;
        
        jointPoint.P.x = kalmanParamR * (jointPoint.P.x + kalmanParamQ) / denomX;
        jointPoint.P.y = kalmanParamR * (jointPoint.P.y + kalmanParamQ) / denomY;
        jointPoint.P.z = kalmanParamR * (jointPoint.P.z + kalmanParamQ) / denomZ;
    }

    /// <summary>
    /// Apply low pass filter for additional smoothing
    /// </summary>
    private void ApplyLowPassFilter()
    {
        foreach (var jointPoint in m_JointPoints)
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

    #region Resource Management
    /// <summary>
    /// Safely dispose of a tensor
    /// </summary>
    private void DisposeTensorSafely(Tensor<float> tensor)
    {
        tensor?.Dispose();
    }

    /// <summary>
    /// Clean up all allocated resources
    /// </summary>
    private void CleanupResources()
    {
        // Dispose input tensors
        foreach (var kvp in m_InputTensors)
        {
            DisposeTensorSafely(kvp.Value);
        }
        m_InputTensors.Clear();

        // Dispose output tensors
        DisposeTensorSafely(m_Output3DOffset);
        DisposeTensorSafely(m_Output3DHeatMap);
        DisposeTensorSafely(m_CurrentInput);

        // Dispose worker and model
        m_Worker?.Dispose();
        
        if (verbose)
        {
            Debug.Log("VNect Sentis Runner resources cleaned up");
        }
    }
    #endregion
}