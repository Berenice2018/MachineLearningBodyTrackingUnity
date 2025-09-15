using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using System.Collections;
using System.Collections.Generic;
using System;

/// <summary>
/// MoveNet Lightning v3 runner using Unity Sentis
/// Handles real-time human pose detection from video input
/// </summary>
public class MovenetRunner : MonoBehaviour
{
    #region Model Configuration
    
    [Header("Model Settings")]
    [Tooltip("The MoveNet model asset for pose estimation")]
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

    [SerializeField] private SentisYOLODetector yolo;   // optional cropper
    public Texture ExternalInputTexture { get; set; }

    [Tooltip("Initial image for model warm-up")]
    public Texture2D initImage;

    [SerializeField] private MovenetPoseProcessor3D _poseProcessor;
    #endregion
    
    #region Input/Output Configuration
    
    [Header("Input Configuration")]
    [Tooltip("Input image size (width and height)")]
    public int inputImageSize = 192;

    [Header("Model Loading")]
    [Tooltip("Wait time after model loading before starting inference")]
    public float waitTimeModelLoad = 1f;
    
    #endregion
    
    #region Private Fields - Sentis Components
    
    private Model _model;
    private Worker _worker;
    private string _inputName;
    
    private Dictionary<string, Tensor<float>> _inputTensors;
    private bool _isProcessing = false;
    private bool _isModelReady = false;
    private Texture _inputTextureToUse;

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
            if (yolo != null && yolo.CroppedTexture != null)
                ExternalInputTexture = yolo.CroppedTexture;

            ProcessFrame();
        }
    }
    
    private void OnDestroy()
    {
        CleanupResources();
    }
    
    #endregion
    
    #region Initialization
    
    private void InitializeSystem()
    {
        try
        {
           //todo _poseProcessor.InitializeParameters();
            InitializeModel();
            InitializeInputTensors();
            
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            
            StartCoroutine(WarmupModel());
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize network system: {e.Message}");
        }
    }
    
    private void InitializeModel()
    {
        if (modelAsset == null)
            throw new ArgumentNullException(nameof(modelAsset), "Model asset is not assigned");
        
        _model = ModelLoader.Load(modelAsset);
        _worker = new Worker(_model, backendType);
        
        // MoveNet has a single input
        _inputName = _model.inputs[0].name;
        
        if (verbose)
        {
            Debug.Log($"Model loaded with backend: {backendType}");
            LogModelInfo();
        }
    }
    
    private void InitializeInputTensors()
    {
        _inputTensors = new Dictionary<string, Tensor<float>>();
        foreach (var input in _model.inputs)
            _inputTensors[input.name] = null;
    }
    
    private void LogModelInfo()
    {
        Debug.Log("=== Model Information ===");
        Debug.Log($"Inputs: {_model.inputs.Count}");
        foreach (var input in _model.inputs)
            Debug.Log($"  {input.name}: {input.shape}");
        
        Debug.Log($"Outputs: {_model.outputs.Count}");
        foreach (var output in _model.outputs)
            Debug.Log($"  {output.name}: {output.index}");
    }
    
    #endregion
    
    #region Model Warmup
    
    private IEnumerator WarmupModel()
    {
        if (initImage == null)
        {
            Debug.LogWarning("No initial image provided for model warmup");
            yield break;
        }
        
        Debug.Log("Starting model warmup...");
        
        var warmupTensor = CreateInputTensor(initImage);
        _worker.SetInput(_inputName, warmupTensor);
        _inputTensors[_inputName] = warmupTensor;
        
        _worker.Schedule();
        yield return null;

        GetOutputs();
        //_poseProcessor.InitJoints();
        //_poseProcessor.PredictPose();
        
        yield return new WaitForSeconds(waitTimeModelLoad);
        
        _isModelReady = true;
        Debug.Log("Model warmup completed and ready for inference");
    }
    
    #endregion
    
    #region Frame Processing
    
    private void ProcessFrame()
    {
        _isProcessing = true;
        StartCoroutine(ExecuteModel());
    }
    
    private IEnumerator ExecuteModel()
    {
        UpdateInputTensors();
        _worker.Schedule();
        yield return null;

        GetOutputs();
        //_poseProcessor.PredictPose();
        
        _isProcessing = false;
    }

    private void GetOutputs()
    {
        // MoveNet typically has one output tensor: [1,1,17,3]
        var keypointsGpu = _worker.PeekOutput(_model.outputs[0].name) as Tensor<float>;
        using var keypoints = keypointsGpu.ReadbackAndClone();
        
        _poseProcessor.UploadNetworkOutputs(keypoints.DownloadToArray());
    }
    
    private void UpdateInputTensors()
    {
        _inputTextureToUse = ExternalInputTexture != null ? ExternalInputTexture : videoCapture.MainTexture;
        var newTensor = CreateInputTensor(_inputTextureToUse);

        if (_inputTensors.ContainsKey(_inputName))
            _inputTensors[_inputName]?.Dispose();

        _inputTensors[_inputName] = newTensor;
        _worker.SetInput(_inputName, newTensor);
    }
    
    private Tensor<float> CreateInputTensor(Texture texture)
    {
        return TextureConverter.ToTensor(texture, inputImageSize, inputImageSize, 3);
    }
    
    #endregion
    
    #region Resource Management
    
    private void CleanupResources()
    {
        try
        {
            if (_inputTensors != null)
            {
                foreach (var tensor in _inputTensors.Values)
                    tensor?.Dispose();
                _inputTensors.Clear();
            }
            
            _worker?.Dispose();
            
            if (verbose)
                Debug.Log("MoveNet resources cleaned up successfully");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error during resource cleanup: {e.Message}");
        }
    }
    
    #endregion
    
    #region Public API
    
    public bool IsModelReady => _isModelReady;
    public bool IsProcessing => _isProcessing;
    
    public void ReinitializeModel()
    {
        StopAllCoroutines();
        _isModelReady = false;
        CleanupResources();
        InitializeSystem();
    }
    
    #endregion
}
