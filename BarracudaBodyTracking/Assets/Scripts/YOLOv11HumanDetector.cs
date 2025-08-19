using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;

namespace PoseDetection
{
    /// <summary>
    /// YOLOv11 Human Detection for Unity using Sentis
    /// Detects single human bounding box for bottom-up pose estimation pipeline
    /// </summary>
    public class YOLOv11HumanDetector : MonoBehaviour
    {
        #region Serialized Fields
        
        [Header("Model Configuration")]
        [SerializeField] private ModelAsset yoloModelAsset;
        [SerializeField] private int inputWidth = 640;
        [SerializeField] private int inputHeight = 640;
        [SerializeField] private float confidenceThreshold = 0.5f;
        [SerializeField] private float nmsThreshold = 0.4f;
        
        [Header("Detection Settings")]
        [SerializeField] private bool enableVisualization = true;
        [SerializeField] private Material boundingBoxMaterial;
        [SerializeField] private Transform debugCanvas;
        
        [Header("Performance")]
        [SerializeField] private BackendType backendType = BackendType.GPUCompute;
        [SerializeField] private int maxDetectionsPerFrame = 1;
        
        #endregion

        #region Private Fields
        
        // Sentis components
        private Worker worker;
        private Model runtimeModel;
        private Tensor<float> inputTensor;
        private Tensor outputTensor;
        
        // Input processing
        private RenderTexture preprocessedTexture;
        private Material preprocessMaterial;
        private Texture2D inputTexture2D;
        
        // Detection results
        private DetectionResult currentDetection;
        private List<DetectionResult> allDetections = new List<DetectionResult>();
        
        // Debug visualization
        private GameObject boundingBoxVisualizer;
        private LineRenderer boundingBoxRenderer;
        
        // Performance tracking
        private float lastInferenceTime;
        private int frameCount;
        
        // COCO class index for person (0 in COCO dataset)
        private const int PERSON_CLASS_ID = 0;
        
        #endregion

        #region Data Structures
        
        /// <summary>
        /// Represents a single detection result
        /// </summary>
        [System.Serializable]
        public struct DetectionResult
        {
            public Rect boundingBox;        // Normalized coordinates (0-1)
            public float confidence;
            public int classId;
            public Vector2 center;          // Normalized center point
            public bool isValid;
            
            public DetectionResult(float x, float y, float width, float height, float conf, int cls)
            {
                boundingBox = new Rect(x, y, width, height);
                confidence = conf;
                classId = cls;
                center = new Vector2(x + width * 0.5f, y + height * 0.5f);
                isValid = true;
            }
        }
        
        /// <summary>
        /// Event triggered when human is detected
        /// </summary>
        public System.Action<DetectionResult> OnHumanDetected;
        
        /// <summary>
        /// Event triggered when no human is detected
        /// </summary>
        public System.Action OnNoHumanDetected;
        
        #endregion

        #region Unity Lifecycle
        
        private void Start()
        {
            InitializeDetector();
            SetupVisualization();
        }
        
        private void Update()
        {
            frameCount++;
        }
        
        private void OnDestroy()
        {
            CleanupResources();
        }
        
        private void OnDisable()
        {
            CleanupResources();
        }
        
        #endregion

        #region Initialization
        
        /// <summary>
        /// Initialize the YOLO detector with Sentis
        /// </summary>
        private void InitializeDetector()
        {
            try
            {
                // Validate model asset
                if (yoloModelAsset == null)
                {
                    Debug.LogError("[YOLOv11HumanDetector] Model asset is not assigned!");
                    return;
                }

                // Load and optimize model
                runtimeModel = ModelLoader.Load(yoloModelAsset);
                
                // Create worker with specified backend
                worker = new Worker(runtimeModel, backendType);
                
                // Initialize preprocessing materials
                CreatePreprocessingResources();
                
                Debug.Log($"[YOLOv11HumanDetector] Initialized successfully with {backendType} backend");
                Debug.Log($"[YOLOv11HumanDetector] Input resolution: {inputWidth}x{inputHeight}");
                Debug.Log($"[YOLOv11HumanDetector] Confidence threshold: {confidenceThreshold}");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[YOLOv11HumanDetector] Failed to initialize: {e.Message}");
                enabled = false;
            }
        }
        
        /// <summary>
        /// Create resources needed for input preprocessing
        /// </summary>
        private void CreatePreprocessingResources()
        {
            // Create render texture for preprocessing
            preprocessedTexture = new RenderTexture(inputWidth, inputHeight, 0, RenderTextureFormat.ARGB32);//todo which render format?
            preprocessedTexture.enableRandomWrite = true;
            preprocessedTexture.Create();
            
            // Create material for preprocessing (normalization, etc.)
            Shader preprocessShader = Shader.Find("Hidden/YOLOPreprocess");
            if (preprocessShader != null)
            {
                preprocessMaterial = new Material(preprocessShader);
            }
            
            // Create texture2D for tensor conversion
            inputTexture2D = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);
        }
        
        /// <summary>
        /// Setup debug visualization components
        /// </summary>
        private void SetupVisualization()
        {
            if (!enableVisualization) return;
            
            // Create bounding box visualizer
            boundingBoxVisualizer = new GameObject("BoundingBoxVisualizer");
            boundingBoxVisualizer.transform.SetParent(debugCanvas != null ? debugCanvas : transform);
            
            boundingBoxRenderer = boundingBoxVisualizer.AddComponent<LineRenderer>();
            boundingBoxRenderer.material = boundingBoxMaterial != null ? boundingBoxMaterial : 
                                         new Material(Shader.Find("Sprites/Default"));
            boundingBoxRenderer.material.color = Color.green;
            boundingBoxRenderer.widthMultiplier = 2f;
            boundingBoxRenderer.useWorldSpace = false;
            boundingBoxRenderer.positionCount = 5; // Rectangle + closing line
            
            boundingBoxVisualizer.SetActive(false);
        }
        
        #endregion

        #region Public Methods
        
        /// <summary>
        /// Perform human detection on input video texture
        /// </summary>
        /// <param name="inputVideoTexture">Input video texture from camera/video player</param>
        /// <returns>Detection result for single human, invalid result if none found</returns>
        public DetectionResult DetectHuman(RenderTexture inputVideoTexture)
        {
            if (worker == null || inputVideoTexture == null)
            {
                return new DetectionResult { isValid = false };
            }
            
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            try
            {
                // Preprocess input
                Tensor<float> processedInput = PreprocessInput(inputVideoTexture);
                
                // Run inference
                worker.Schedule();
                
                // Get output tensor
                Tensor<float> output = worker.PeekOutput() as Tensor<float>;
                
                // Process detections
                DetectionResult humanDetection = ProcessDetections(output, inputVideoTexture.width, inputVideoTexture.height);
                
                // Cleanup
                processedInput.Dispose();
                output.Dispose();
                
                stopwatch.Stop();
                lastInferenceTime = stopwatch.ElapsedMilliseconds;
                
                // Update visualization
                if (enableVisualization)
                {
                    UpdateVisualization(humanDetection);
                }
                
                // Trigger events
                if (humanDetection.isValid)
                {
                    OnHumanDetected?.Invoke(humanDetection);
                }
                else
                {
                    OnNoHumanDetected?.Invoke();
                }
                
                currentDetection = humanDetection;
                return humanDetection;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[YOLOv11HumanDetector] Detection failed: {e.Message}");
                return new DetectionResult { isValid = false };
            }
        }
        
        /// <summary>
        /// Get the current detection result
        /// </summary>
        public DetectionResult GetCurrentDetection()
        {
            return currentDetection;
        }
        
        /// <summary>
        /// Check if a human is currently detected
        /// </summary>
        public bool IsHumanDetected()
        {
            return currentDetection.isValid && currentDetection.confidence >= confidenceThreshold;
        }
        
        /// <summary>
        /// Get performance metrics
        /// </summary>
        public float GetLastInferenceTime()
        {
            return lastInferenceTime;
        }
        
        /// <summary>
        /// Convert normalized detection to screen coordinates
        /// </summary>
        public Rect GetScreenSpaceBoundingBox(DetectionResult detection, int screenWidth, int screenHeight)
        {
            if (!detection.isValid) return Rect.zero;
            
            return new Rect(
                detection.boundingBox.x * screenWidth,
                detection.boundingBox.y * screenHeight,
                detection.boundingBox.width * screenWidth,
                detection.boundingBox.height * screenHeight
            );
        }
        
        #endregion

        #region Private Methods - Input Processing
        
        /// <summary>
        /// Preprocess input texture for YOLO inference
        /// </summary>
        private Tensor<float> PreprocessInput(Texture inputTexture)
        {
            // Resize and normalize input texture
            if (preprocessMaterial != null)
            {
                Graphics.Blit(inputTexture, preprocessedTexture, preprocessMaterial);
            }
            else
            {
                Graphics.Blit(inputTexture, preprocessedTexture);
            }
            
            // Convert to Texture2D
            RenderTexture.active = preprocessedTexture;
            inputTexture2D.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
            inputTexture2D.Apply();
            RenderTexture.active = null;
            
            // Convert to tensor (NCHW format for YOLO)
            Color32[] pixels = inputTexture2D.GetPixels32();
            float[] tensorData = new float[3 * inputHeight * inputWidth];
            
            // Normalize to [0, 1] and arrange in CHW format
            for (int y = 0; y < inputHeight; y++)
            {
                for (int x = 0; x < inputWidth; x++)
                {
                    int pixelIndex = y * inputWidth + x;
                    Color32 pixel = pixels[pixelIndex];
                    
                    // RGB channels (normalized to 0-1)
                    tensorData[0 * inputHeight * inputWidth + pixelIndex] = pixel.r / 255f; // R channel
                    tensorData[1 * inputHeight * inputWidth + pixelIndex] = pixel.g / 255f; // G channel
                    tensorData[2 * inputHeight * inputWidth + pixelIndex] = pixel.b / 255f; // B channel
                }
            }
            
            // Create tensor with shape [1, 3, height, width]
            Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth), tensorData);
            return inputTensor;
        }
        
        #endregion

        #region Private Methods - Output Processing
        
        /// <summary>
        /// Process YOLO output and extract human detections
        /// </summary>
        private DetectionResult ProcessDetections(Tensor<float> output, int originalWidth, int originalHeight)
        {
            // YOLO output format: [batch, detections, attributes]
            // attributes: [x_center, y_center, width, height, objectness, class_scores...]
            
            var shape = output.shape;
            float[] outputData = output.DownloadToArray();
            
            allDetections.Clear();
            
            int numDetections = shape[1];
            int numAttributes = shape[2];
            
            // Extract detections
            for (int i = 0; i < numDetections; i++)
            {
                int baseIndex = i * numAttributes;
                
                // Extract basic detection data
                float centerX = outputData[baseIndex + 0];
                float centerY = outputData[baseIndex + 1];
                float width = outputData[baseIndex + 2];
                float height = outputData[baseIndex + 3];
                float objectness = outputData[baseIndex + 4];
                
                // Skip low confidence detections
                if (objectness < confidenceThreshold) continue;
                
                // Find class with highest score
                float maxClassScore = 0f;
                int maxClassIndex = 0;
                
                for (int classIndex = 0; classIndex < numAttributes - 5; classIndex++)
                {
                    float classScore = outputData[baseIndex + 5 + classIndex];
                    if (classScore > maxClassScore)
                    {
                        maxClassScore = classScore;
                        maxClassIndex = classIndex;
                    }
                }
                
                // Only process person detections (class 0 in COCO)
                if (maxClassIndex != PERSON_CLASS_ID) continue;
                
                float finalConfidence = objectness * maxClassScore;
                if (finalConfidence < confidenceThreshold) continue;
                
                // Convert from center format to corner format
                float x = (centerX - width * 0.5f) / inputWidth;
                float y = (centerY - height * 0.5f) / inputHeight;
                float w = width / inputWidth;
                float h = height / inputHeight;
                
                // Clamp to [0, 1]
                x = Mathf.Clamp01(x);
                y = Mathf.Clamp01(y);
                w = Mathf.Clamp01(w);
                h = Mathf.Clamp01(h);
                
                DetectionResult detection = new DetectionResult(x, y, w, h, finalConfidence, maxClassIndex);
                allDetections.Add(detection);
            }
            
            // Apply Non-Maximum Suppression and return best human detection
            return ApplyNMSAndGetBestHuman();
        }
        
        /// <summary>
        /// Apply Non-Maximum Suppression and return the best human detection
        /// </summary>
        private DetectionResult ApplyNMSAndGetBestHuman()
        {
            if (allDetections.Count == 0)
            {
                return new DetectionResult { isValid = false };
            }
            
            // Sort by confidence (descending)
            allDetections.Sort((a, b) => b.confidence.CompareTo(a.confidence));
            
            List<DetectionResult> nmsResults = new List<DetectionResult>();
            bool[] suppressed = new bool[allDetections.Count];
            
            // Apply NMS
            for (int i = 0; i < allDetections.Count; i++)
            {
                if (suppressed[i]) continue;
                
                nmsResults.Add(allDetections[i]);
                
                // Suppress overlapping detections
                for (int j = i + 1; j < allDetections.Count; j++)
                {
                    if (suppressed[j]) continue;
                    
                    float iou = CalculateIoU(allDetections[i].boundingBox, allDetections[j].boundingBox);
                    if (iou > nmsThreshold)
                    {
                        suppressed[j] = true;
                    }
                }
                
                // Limit to max detections per frame
                if (nmsResults.Count >= maxDetectionsPerFrame) break;
            }
            
            // Return the best detection (highest confidence)
            return nmsResults.Count > 0 ? nmsResults[0] : new DetectionResult { isValid = false };
        }
        
        /// <summary>
        /// Calculate Intersection over Union (IoU) between two bounding boxes
        /// </summary>
        private float CalculateIoU(Rect boxA, Rect boxB)
        {
            float intersectionArea = Mathf.Max(0, Mathf.Min(boxA.xMax, boxB.xMax) - Mathf.Max(boxA.xMin, boxB.xMin)) *
                                   Mathf.Max(0, Mathf.Min(boxA.yMax, boxB.yMax) - Mathf.Max(boxA.yMin, boxB.yMin));
            
            float boxAArea = boxA.width * boxA.height;
            float boxBArea = boxB.width * boxB.height;
            float unionArea = boxAArea + boxBArea - intersectionArea;
            
            return unionArea > 0 ? intersectionArea / unionArea : 0;
        }
        
        #endregion

        #region Private Methods - Visualization
        
        /// <summary>
        /// Update debug visualization
        /// </summary>
        private void UpdateVisualization(DetectionResult detection)
        {
            if (boundingBoxRenderer == null) return;
            
            if (detection.isValid && detection.confidence >= confidenceThreshold)
            {
                boundingBoxVisualizer.SetActive(true);
                
                // Convert normalized coordinates to screen space for visualization
                Rect box = detection.boundingBox;
                
                // Create rectangle points (assuming UI space from -1 to 1)
                Vector3[] points = new Vector3[5];
                points[0] = new Vector3(box.xMin * 2 - 1, box.yMin * 2 - 1, 0); // Bottom-left
                points[1] = new Vector3(box.xMax * 2 - 1, box.yMin * 2 - 1, 0); // Bottom-right
                points[2] = new Vector3(box.xMax * 2 - 1, box.yMax * 2 - 1, 0); // Top-right
                points[3] = new Vector3(box.xMin * 2 - 1, box.yMax * 2 - 1, 0); // Top-left
                points[4] = points[0]; // Close the rectangle
                
                boundingBoxRenderer.positionCount = 5;
                boundingBoxRenderer.SetPositions(points);
                
                // Color based on confidence
                Color boxColor = Color.Lerp(Color.yellow, Color.green, detection.confidence);
                boundingBoxRenderer.material.color = boxColor;
            }
            else
            {
                boundingBoxVisualizer.SetActive(false);
            }
        }
        
        #endregion

        #region Resource Management
        
        /// <summary>
        /// Clean up all allocated resources
        /// </summary>
        private void CleanupResources()
        {
            // Dispose Sentis resources
            inputTensor?.Dispose();
            outputTensor?.Dispose();
            worker?.Dispose();
            
            // Destroy Unity resources
            if (preprocessedTexture != null)
            {
                preprocessedTexture.Release();
                DestroyImmediate(preprocessedTexture);
            }
            
            if (preprocessMaterial != null)
            {
                DestroyImmediate(preprocessMaterial);
            }
            
            if (inputTexture2D != null)
            {
                DestroyImmediate(inputTexture2D);
            }
            
            if (boundingBoxVisualizer != null)
            {
                DestroyImmediate(boundingBoxVisualizer);
            }
            
            Debug.Log("[YOLOv11HumanDetector] Resources cleaned up");
        }
        
        #endregion

        #region Debug and Diagnostics
        
        private void OnGUI()
        {
            if (!enableVisualization) return;
            
            GUILayout.BeginArea(new Rect(10, 10, 300, 200));
            GUILayout.Label("YOLO Human Detection Debug", EditorGUIUtility.isProSkin ? 
                           GUI.skin.box : GUI.skin.label);
            GUILayout.Label($"Inference Time: {lastInferenceTime:F1}ms");
            GUILayout.Label($"FPS: {1000f / Mathf.Max(lastInferenceTime, 1f):F1}");
            GUILayout.Label($"Backend: {backendType}");
            
            if (currentDetection.isValid)
            {
                GUILayout.Label($"Human Detected: {currentDetection.confidence:F2}");
                GUILayout.Label($"Position: ({currentDetection.center.x:F2}, {currentDetection.center.y:F2})");
                GUILayout.Label($"Size: {currentDetection.boundingBox.width:F2} x {currentDetection.boundingBox.height:F2}");
            }
            else
            {
                GUILayout.Label("No Human Detected");
            }
            
            GUILayout.EndArea();
        }
        
        #endregion
    }
}