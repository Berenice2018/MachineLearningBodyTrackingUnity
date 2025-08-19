using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine.Serialization;

/// <summary>
/// VNect pose estimation runner using Unity Sentis 2.1.3
/// Handles real-time human pose detection from video input
/// </summary>
public class SentisRunner : MonoBehaviour
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

    [Tooltip("Video capture component for input")]
    public VideoCapture videoCapture;

    [SerializeField] private SentisYOLODetector yolo;
    public Texture ExternalInputTexture { get; set; } //cropped image from YOLO detector

    [Tooltip("Initial image for model warm-up")]
    public Texture2D initImage;

    [SerializeField] private PoseProcessor _poseProcessor;
    #endregion
    
    #region Input/Output Configuration
    
    [Header("Input Configuration")]
    [Tooltip("Input image size (width and height)")]
    public int inputImageSize = 448;

    [Header("Model Loading")]
    [Tooltip("Wait time after model loading before starting inference")]
    public float waitTimeModelLoad = 10f;
    
    #endregion
    
    #region Private Fields - Sentis Components
    
    private Model _model;
    private Worker _worker;
    
    #endregion
    
    
    #region Private Fields - Input Management

    private Texture _inputTextureToUse;
    private string _input1Name, _input2Name, _input3Name;
    
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
            if (yolo.CroppedTexture != null)
            {
                ExternalInputTexture = yolo.CroppedTexture;
            }

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
            _poseProcessor.InitializeParameters();
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
        
        // Get the actual input names from the model
        _input1Name = _model.inputs[0].name;
        _input2Name = _model.inputs[1].name;
        _input3Name = _model.inputs[2].name;
        
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

            GetOutputs();
            
            // Initialize joint points
            _poseProcessor.InitJoints();
            
            // Process initial pose
            _poseProcessor.PredictPose();
            
            // Wait for specified time
            yield return new WaitForSeconds(waitTimeModelLoad);
            
            /*/ Initialize video capture
            if (videoCapture != null)
            {
                videoCapture.Init(inputImageSize, inputImageSize);
            }*/
            
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
        _isProcessing = true;
        StartCoroutine(ExecuteModel());
    }
    
    /// <summary>
    /// Execute model inference 
    /// </summary>
    private IEnumerator ExecuteModel()
    {
        // Update input tensors with new frame
        UpdateInputTensors();
            
        // Execute model
        _worker.Schedule();
        yield return null; // let the scheduled work run

        GetOutputs();
            
        // Process pose prediction
        _poseProcessor.PredictPose();
        
        _isProcessing = false;
    }

    private void GetOutputs()
    {
        // Get outputs (GPU tensors)
        var offset3DGpu   = _worker.PeekOutput(_model.outputs[Output2Index].name) as Tensor<float>;
        var heatMap3DGpu  = _worker.PeekOutput(_model.outputs[Output3Index].name) as Tensor<float>;

        // Read back to CPU and use the returned clones
        using var offset3D = offset3DGpu.ReadbackAndClone();    // CPU tensor
        using var heatMap3D = heatMap3DGpu.ReadbackAndClone();  // CPU tensor
        
        // Push into NativeArrays inside PoseProcessor (no per-frame Native allocations)
        _poseProcessor.UploadNetworkOutputs(
            offset3D.DownloadToArray(),
            heatMap3D.DownloadToArray()
        );
    }
    
    /// <summary>
    /// Update input tensors with new frame, maintaining 3-frame history
    /// </summary>
    private void UpdateInputTensors()
    {
        _inputTextureToUse = ExternalInputTexture != null ? ExternalInputTexture : videoCapture.MainTexture;
        var newTensor = CreateInputTensor(_inputTextureToUse);
        
        // Dispose oldest tensor and shift the ring buffer
        if (_inputTensors.ContainsKey(_input3Name))
        {
            _inputTensors[_input3Name]?.Dispose();
        }
        
        // Shift tensors in ring buffer
        if (_inputTensors.ContainsKey(_input2Name))
        {
            _inputTensors[_input3Name] = _inputTensors[_input2Name];
            _worker.SetInput(_input3Name, _inputTensors[_input3Name]);
        }
        
        if (_inputTensors.ContainsKey(_input1Name))
        {
            _inputTensors[_input2Name] = _inputTensors[_input1Name];
            _worker.SetInput(_input2Name, _inputTensors[_input2Name]);
        }
        
        // Set new tensor
        _inputTensors[_input1Name] = newTensor;
        _worker.SetInput(_input1Name, newTensor);
    }
    
    /// <summary>
    /// Create input tensor from texture
    /// </summary>
    private Tensor<float> CreateInputTensor(Texture texture)
    {
        return TextureConverter.ToTensor(texture, inputImageSize, inputImageSize, 3);
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