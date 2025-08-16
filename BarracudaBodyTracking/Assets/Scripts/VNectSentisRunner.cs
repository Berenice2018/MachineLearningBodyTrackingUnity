using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using System.Collections;
using System.Collections.Generic;
using System;

/// <summary>
/// VNect pose estimation runner using Unity Sentis 2.1.3
/// Handles real-time human pose detection from video input
/// </summary>
public class VNectSentisRunner : MonoBehaviour
{
    #region Model Configuration
    
    [Header("Model Settings")]
    [Tooltip("The ResNet model asset for pose estimation")]
    public ModelAsset modelAsset;
    
    [Tooltip("Backend type for model execution")]
    public BackendType backendType = BackendType.GPUCompute;
    
    [Tooltip("Enable verbose logging for debugging")]
    public bool verbose = true;
    
    #endregion
    
    #region Component References
    
    [Header("Component References")]
    [Tooltip("VNect model component for pose visualization")]
    public VNectModel vNectModel;
    
    [Tooltip("Video capture component for input")]
    public VideoCapture videoCapture;
    
    [Tooltip("Initial image for model warm-up")]
    public Texture2D initImage;
    
    #endregion
    
    #region Input/Output Configuration
    
    [Header("Input Configuration")]
    [Tooltip("Input image size (width and height)")]
    public int inputImageSize = 448;
    
    [Tooltip("Heatmap resolution")]
    public int heatMapCol = 28;
    
    [Header("Model Loading")]
    [Tooltip("Wait time after model loading before starting inference")]
    public float waitTimeModelLoad = 10f;
    
    #endregion
    
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
    public float lowPassParam = 0.8f;
    
    #endregion
    
    #region Private Fields - Sentis Components
    
    private Model _model;
    private Worker _worker;
    
    #endregion
    
    #region Private Fields - Processing Data
    
    private VNectModel.JointPoint[] _jointPoints;
    private const int JointNum = 24;
    
    // Calculated dimensions
    private float _inputImageSizeHalf;
    private float _inputImageSizeF;
    private int _heatMapColSquared;
    private int _heatMapColCubed;
    private float _imageScale;
    private float _unit;
    
    // Index calculations
    private int _jointNumSquared = JointNum * 2;
    private int _jointNumCubed = JointNum * 3;
    private int _heatMapColJointNum;
    private int _cubeOffsetLinear;
    private int _cubeOffsetSquared;
    
    // Buffer arrays
    private float[] _heatMap3D;
    private float[] _offset3D;
    
    #endregion
    
    #region Private Fields - Input Management
    
    // Input tensor names based on model specification
    private const string InputName1 = "input.1";
    private const string InputName4 = "input.4"; 
    private const string InputName7 = "input.7";
    
    // Output indices based on model specification
    private const int Output2Index = 2; // 530: offset 3D
    private const int Output3Index = 3; // 516: heatmap 3D
    
    private Dictionary<string, Tensor<float>> _inputTensors;
    private bool _isProcessing = false;
    private bool _isModelReady = false;
    
    #endregion
    
    #region Unity Lifecycle
    
    private void Start()
    {
        InitializeSystem();
    }
    
    private void Update()
    {
        if (_isModelReady && !_isProcessing)
        {
            ProcessFrame();
        }
    }
    
    private void OnDestroy()
    {
        CleanupResources();
    }
    
    #endregion
    
    #region Initialization
    
    /// <summary>
    /// Initialize the pose estimation system
    /// </summary>
    private void InitializeSystem()
    {
        try
        {
            InitializeParameters();
            InitializeModel();
            InitializeInputTensors();
            
            // Prevent screen sleep for mobile devices
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            
            StartCoroutine(WarmupModel());
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize VNect system: {e.Message}");
        }
    }
    
    /// <summary>
    /// Initialize calculation parameters and buffer arrays
    /// </summary>
    private void InitializeParameters()
    {
        // Calculate derived dimensions
        _heatMapColSquared = heatMapCol * heatMapCol;
        _heatMapColCubed = heatMapCol * heatMapCol * heatMapCol;
        _heatMapColJointNum = heatMapCol * JointNum;
        _cubeOffsetLinear = heatMapCol * _jointNumCubed;
        _cubeOffsetSquared = _heatMapColSquared * _jointNumCubed;
        
        // Initialize buffer arrays
        _heatMap3D = new float[JointNum * _heatMapColCubed];
        _offset3D = new float[JointNum * _heatMapColCubed * 3];
        
        // Calculate scaling parameters
        _unit = 1f / (float)heatMapCol;
        _inputImageSizeF = inputImageSize;
        _inputImageSizeHalf = _inputImageSizeF / 2f;
        _imageScale = inputImageSize / (float)heatMapCol;
        
        if (verbose)
        {
            Debug.Log($"VNect parameters initialized - Image size: {inputImageSize}, Heatmap: {heatMapCol}");
        }
    }
    
    /// <summary>
    /// Initialize the Sentis model and worker
    /// </summary>
    private void InitializeModel()
    {
        if (modelAsset == null)
        {
            throw new ArgumentNullException(nameof(modelAsset), "Model asset is not assigned");
        }
        
        _model = ModelLoader.Load(modelAsset);
        _worker = new Worker(_model,backendType);
        
        if (verbose)
        {
            Debug.Log($"Model loaded with backend: {backendType}");
            LogModelInfo();
        }
    }
    
    /// <summary>
    /// Initialize input tensor dictionary
    /// </summary>
    private void InitializeInputTensors()
    {
        _inputTensors = new Dictionary<string, Tensor<float>>();
        
        // Initialize with actual model input names
        foreach (var input in _model.inputs)
        {
            _inputTensors[input.name] = null;
        }
    }
    
    /// <summary>
    /// Log model information for debugging
    /// </summary>
    private void LogModelInfo()
    {
        Debug.Log("=== Model Information ===");
        Debug.Log($"Inputs: {_model.inputs.Count}");
        foreach (var input in _model.inputs)
        {
            Debug.Log($"  {input.name}: {input.shape}");
        }
        
        Debug.Log($"Outputs: {_model.outputs.Count}");
        foreach (var output in _model.outputs)
        {
            Debug.Log($"  {output.name}: {output.index}");
        }
    }
    
    #endregion
    
    #region Model Warmup
    
    /// <summary>
    /// Warm up the model with initial image
    /// </summary>
    private IEnumerator WarmupModel()
    {
        if (initImage == null)
        {
            Debug.LogWarning("No initial image provided for model warmup");
            yield break;
        }
        
        Debug.Log("Starting model warmup...");
        
        //try
        {
            // Create initial tensors for warmup using proper input names from model
            var warmupTensor1 = CreateInputTensor(initImage);
            var warmupTensor2 = CreateInputTensor(initImage);
            var warmupTensor3 = CreateInputTensor(initImage);
            
            // Set inputs using the correct model input names
            _worker.SetInput(_model.inputs[0].name, warmupTensor1);
            _worker.SetInput(_model.inputs[1].name, warmupTensor2);
            _worker.SetInput(_model.inputs[2].name, warmupTensor3);
            
            // Store tensors for disposal
            _inputTensors[_model.inputs[0].name] = warmupTensor1;
            _inputTensors[_model.inputs[1].name] = warmupTensor2;
            _inputTensors[_model.inputs[2].name] = warmupTensor3;
            
            // Execute model for warmup
            _worker.Schedule();
            yield return null; // let the scheduled work run
            
// Get outputs (GPU tensors)
            var offset3DGpu   = _worker.PeekOutput(_model.outputs[Output2Index].name) as Tensor<float>;
            var heatMap3DGpu  = _worker.PeekOutput(_model.outputs[Output3Index].name) as Tensor<float>;

// Read back to CPU and use the returned clones
            using var offset3D = offset3DGpu.ReadbackAndClone();    // CPU tensor
            using var heatMap3D = heatMap3DGpu.ReadbackAndClone();  // CPU tensor
            _offset3D = offset3D.DownloadToArray(); //Flatten to float[]
            _heatMap3D = heatMap3D.DownloadToArray();
            
            // Initialize joint points
            _jointPoints = vNectModel.Init();
            
            // Process initial pose
            PredictPose();
            
            // Wait for specified time
            yield return new WaitForSeconds(waitTimeModelLoad);
            
            // Initialize video capture
            if (videoCapture != null)
            {
                videoCapture.Init(inputImageSize, inputImageSize);
            }
            
            _isModelReady = true;
            Debug.Log("Model warmup completed and ready for inference");
        }
        /*catch (Exception e)
        {
            Debug.LogError($"Model warmup failed: {e.Message}");
        }*/
    }
    
    #endregion
    
    #region Frame Processing
    
    /// <summary>
    /// Process a single frame from video capture
    /// </summary>
    private void ProcessFrame()
    {
        if (videoCapture?.MainTexture == null)
            return;
        
        _isProcessing = true;
        StartCoroutine(ExecuteModelAsync());
    }
    
    /// <summary>
    /// Execute model inference asynchronously
    /// </summary>
    private IEnumerator ExecuteModelAsync()
    {
        //try
        {
            // Update input tensors with new frame
            UpdateInputTensors();
            
            // Execute model
            _worker.Schedule();
            yield return null; // let the scheduled work run
            
// Get outputs (GPU tensors)
            var offset3DGpu   = _worker.PeekOutput(_model.outputs[Output2Index].name) as Tensor<float>;
            var heatMap3DGpu  = _worker.PeekOutput(_model.outputs[Output3Index].name) as Tensor<float>;

// Read back to CPU and use the returned clones
            using var offset3D = offset3DGpu.ReadbackAndClone();    // CPU tensor
            using var heatMap3D = heatMap3DGpu.ReadbackAndClone();  // CPU tensor
            _offset3D = offset3D.DownloadToArray(); //Flatten to float[]
            _heatMap3D = heatMap3D.DownloadToArray();
            
            // Process pose prediction
            PredictPose();
        }
        /*catch (Exception e)
        {
            Debug.LogError($"Model execution failed: {e.Message}");
        }
        finally*/
        {
            _isProcessing = false;
        }
    }
    
    /// <summary>
    /// Update input tensors with new frame, maintaining 3-frame history
    /// </summary>
    private void UpdateInputTensors()
    {
        var newTensor = CreateInputTensor(videoCapture.MainTexture);
        
        // Get the actual input names from the model
        string input1Name = _model.inputs[0].name;
        string input2Name = _model.inputs[1].name;
        string input3Name = _model.inputs[2].name;
        
        // Dispose oldest tensor and shift the ring buffer
        if (_inputTensors.ContainsKey(input3Name))
        {
            _inputTensors[input3Name]?.Dispose();
        }
        
        // Shift tensors in ring buffer
        if (_inputTensors.ContainsKey(input2Name))
        {
            _inputTensors[input3Name] = _inputTensors[input2Name];
            _worker.SetInput(input3Name, _inputTensors[input3Name]);
        }
        
        if (_inputTensors.ContainsKey(input1Name))
        {
            _inputTensors[input2Name] = _inputTensors[input1Name];
            _worker.SetInput(input2Name, _inputTensors[input2Name]);
        }
        
        // Set new tensor
        _inputTensors[input1Name] = newTensor;
        _worker.SetInput(input1Name, newTensor);
    }
    
    /// <summary>
    /// Create input tensor from texture
    /// </summary>
    private Tensor<float> CreateInputTensor(Texture texture)
    {
        return TextureConverter.ToTensor(texture, inputImageSize, inputImageSize, 3);
    }
    
    #endregion
    
    #region Pose Prediction
    
    /// <summary>
    /// Predict 3D joint positions from network outputs
    /// </summary>
    private void PredictPose()
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
    private void FindMaxActivation(int jointIndex)
    {
        int maxXIndex = 0, maxYIndex = 0, maxZIndex = 0;
        _jointPoints[jointIndex].score3D = 0.0f;
        
        int jj = jointIndex * heatMapCol;
        
        for (int z = 0; z < heatMapCol; z++)
        {
            int zz = jj + z;
            for (int y = 0; y < heatMapCol; y++)
            {
                int yy = y * _heatMapColSquared * JointNum + zz;
                for (int x = 0; x < heatMapCol; x++)
                {
                    float value = _heatMap3D[yy + x * _heatMapColJointNum];
                    if (value > _jointPoints[jointIndex].score3D)
                    {
                        _jointPoints[jointIndex].score3D = value;
                        maxXIndex = x;
                        maxYIndex = y;
                        maxZIndex = z;
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
    private void CalculateJointPosition(int jointIndex, int maxX, int maxY, int maxZ)
    {
        int baseIndex = maxY * _cubeOffsetSquared + maxX * _cubeOffsetLinear;
        
        // X coordinate
        _jointPoints[jointIndex].Now3D.x = 
            (_offset3D[baseIndex + jointIndex * heatMapCol + maxZ] + 0.5f + maxX) * _imageScale - _inputImageSizeHalf;
        
        // Y coordinate (flipped)
        _jointPoints[jointIndex].Now3D.y = 
            _inputImageSizeHalf - (_offset3D[baseIndex + (jointIndex + JointNum) * heatMapCol + maxZ] + 0.5f + maxY) * _imageScale;
        
        // Z coordinate
        _jointPoints[jointIndex].Now3D.z = 
            (_offset3D[baseIndex + (jointIndex + _jointNumSquared) * heatMapCol + maxZ] + 0.5f + (maxZ - 14)) * _imageScale;
    }
    
    /// <summary>
    /// Calculate derived joint positions (hip, neck, head)
    /// </summary>
    private void CalculateDerivedJoints()
    {
        // Calculate hip location
        var leftThigh = _jointPoints[PositionIndex.lThighBend.Int()].Now3D;
        var rightThigh = _jointPoints[PositionIndex.rThighBend.Int()].Now3D;
        var abdomenUpper = _jointPoints[PositionIndex.abdomenUpper.Int()].Now3D;
        var hipCenter = (leftThigh + rightThigh) / 2f;
        _jointPoints[PositionIndex.hip.Int()].Now3D = (abdomenUpper + hipCenter) / 2f;
        
        // Calculate neck location
        var leftShoulder = _jointPoints[PositionIndex.lShldrBend.Int()].Now3D;
        var rightShoulder = _jointPoints[PositionIndex.rShldrBend.Int()].Now3D;
        _jointPoints[PositionIndex.neck.Int()].Now3D = (leftShoulder + rightShoulder) / 2f;
        
        // Calculate head location
        var leftEar = _jointPoints[PositionIndex.lEar.Int()].Now3D;
        var rightEar = _jointPoints[PositionIndex.rEar.Int()].Now3D;
        var earCenter = (leftEar + rightEar) / 2f;
        var neck = _jointPoints[PositionIndex.neck.Int()].Now3D;
        var headVector = earCenter - neck;
        var normalizedHeadVector = Vector3.Normalize(headVector);
        var noseVector = _jointPoints[PositionIndex.Nose.Int()].Now3D - neck;
        _jointPoints[PositionIndex.head.Int()].Now3D = neck + normalizedHeadVector * Vector3.Dot(normalizedHeadVector, noseVector);
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
    private void ApplyKalmanFilter(VNectModel.JointPoint jointPoint)
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
    private void UpdateKalmanGain(VNectModel.JointPoint jointPoint)
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
    
    #region Resource Management
    
    /// <summary>
    /// Clean up all allocated resources
    /// </summary>
    private void CleanupResources()
    {
        try
        {
            // Dispose input tensors
            if (_inputTensors != null)
            {
                foreach (var tensor in _inputTensors.Values)
                {
                    tensor?.Dispose();
                }
                _inputTensors.Clear();
            }
            
            // Dispose worker and model
            _worker?.Dispose();
            
            if (verbose)
            {
                Debug.Log("VNect resources cleaned up successfully");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error during resource cleanup: {e.Message}");
        }
    }
    
    #endregion
    
    #region Public API
    
    /// <summary>
    /// Check if the model is ready for inference
    /// </summary>
    public bool IsModelReady => _isModelReady;
    
    /// <summary>
    /// Check if currently processing a frame
    /// </summary>
    public bool IsProcessing => _isProcessing;
    
    /// <summary>
    /// Get current joint points
    /// </summary>
    public VNectModel.JointPoint[] GetJointPoints() => _jointPoints;
    
    /// <summary>
    /// Manually trigger model reinitialization
    /// </summary>
    public void ReinitializeModel()
    {
        StopAllCoroutines();
        _isModelReady = false;
        CleanupResources();
        InitializeSystem();
    }
    
    #endregion
}