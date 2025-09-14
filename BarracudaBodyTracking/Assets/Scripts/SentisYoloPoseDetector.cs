using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Collections;

public class SentisYOLOPoseDetector : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;  
    public BackendType backendType = BackendType.GPUCompute;
    public int inputImageSize = 640;

    [Header("Input Source")]
    public VideoCapture videoCapture;  // video texture

    [Header("UI")]
    public RectTransform keypointPrefab; // small Image prefab (circle/dot)
    public Transform overlayCanvas;      
    private List<RectTransform> _kpPool = new();

    private Model _model;
    private Worker _worker;
    private string _inputName;
    private string _outputName;

    // Pairs of indices to connect with lines
    private readonly int[,] cocoPairs = new int[,]
    {
        {0,1}, {0,2}, {1,3}, {2,4},        // head
        {5,6},                             // shoulders
        {5,7}, {7,9},                      // left arm
        {6,8}, {8,10},                     // right arm
        {11,12},                           // hips
        {5,11}, {6,12},                    // torso
        {11,13}, {13,15},                  // left leg
        {12,14}, {14,16}                   // right leg
    };
    public RectTransform lineUiPrefab; // assign UI Image prefab
    private List<RectTransform> _lineUiPool = new();

    
    private void Start()
    {
        _model = ModelLoader.Load(modelAsset);
        _worker = new Worker(_model, backendType);
        _inputName = _model.inputs[0].name;
        _outputName = _model.outputs[0].name;

        videoCapture.Init(inputImageSize, inputImageSize);
    }

    private void Update()
    {
        if (videoCapture.MainTexture != null)
            StartCoroutine(RunYOLOPose(videoCapture.MainTexture));
    }

    private IEnumerator RunYOLOPose(Texture tex)
    {
        // 1. Letterbox the video frame into 640×640 with padding
        //RenderTexture lbTex = Letterbox(tex, inputImageSize);

        // 2. Convert that RT into a tensor for YOLO
        using var input = TextureConverter.ToTensor(tex, inputImageSize, inputImageSize, 3);
        
// Convert tensor → array
        float[] data = input.DownloadToArray();

// Reorder RGB → BGR in-place (channels = 3)
        int H = inputImageSize;
        int W = inputImageSize;
        int stride = H * W;

        for (int i = 0; i < stride; i++)
        {
            float r = data[i];
            float g = data[i + stride];
            float b = data[i + 2 * stride];

            // Swap
            data[i]             = b;
            data[i + stride]    = g;
            data[i + 2*stride]  = r;
        }

// Upload back into tensor
        var bgr = new Tensor<float>(input.shape, data);

        _worker.SetInput(_inputName, bgr);

        
        _worker.SetInput(_inputName, input);

        _worker.Schedule();
        yield return null;

        // 3. Run inference
        using var output = _worker.PeekOutput(_outputName).ReadbackAndClone() as Tensor<float>;

        // 4. Decode using the SAME letterboxed size, not the raw video
        List<Vector3> keypoints = DecodeYOLOPose(output);

        // 5. Draw on canvas
        DrawKeypoints(keypoints);
        DrawSkeletonUI(keypoints); 
    }
    
    private List<Vector3> DecodeYOLOPose(Tensor<float> output, float confThresh = 0.4f)
    {
        var keypoints = new List<Vector3>();

        int numPreds = output.shape[1];   // N predictions
        float bestConf = 0f;
        int bestIdx = -1;

        // Find highest-confidence detection
        for (int i = 0; i < numPreds; i++)
        {
            float conf = output[0, i, 4];
            if (conf > bestConf)
            {
                bestConf = conf;
                bestIdx = i;
            }
        }

        if (bestIdx == -1 || bestConf < confThresh)
            return keypoints;

        // Each detection: [x, y, w, h, obj_conf, class_conf, kpts...]
        // Keypoints = 17 × (x, y, conf) in pixel space [0..640]
        for (int k = 0; k < 17; k++)
        {
            float x = output[0, bestIdx, 6 + k * 3];
            float y = output[0, bestIdx, 6 + k * 3 + 1];
            float kconf = output[0, bestIdx, 6 + k * 3 + 2];

            // Normalize to [0,1] (since input is fixed 640×640)
            float nx = x / inputImageSize;
            float ny = y / inputImageSize;

            keypoints.Add(new Vector3(nx, ny, kconf));
        }

        return keypoints;
    }

    private void DrawKeypoints(List<Vector3> keypoints)
    {
        var canvasRect = overlayCanvas.GetComponent<RectTransform>().rect;
        float canvasW = canvasRect.width;
        float canvasH = canvasRect.height;

        foreach (var kp in _kpPool) kp.gameObject.SetActive(false);

        for (int i = 0; i < keypoints.Count; i++)
        {
            if (keypoints[i].z < 0.3f) continue; // skip low confidence

            RectTransform dot;
            if (i < _kpPool.Count) dot = _kpPool[i];
            else
            {
                dot = Instantiate(keypointPrefab, overlayCanvas);
                _kpPool.Add(dot);
            }

            dot.gameObject.SetActive(true);

            // Scale normalized coords to canvas
            float cx = keypoints[i].x * canvasW;
            float cy = (1f - keypoints[i].y) * canvasH; // flip Y
            //Debug.Log($"keypoint {i}: x={cx}, y= {cy}");

            dot.anchoredPosition = new Vector2(cx, cy);
        }
    }

    public LineRenderer linePrefab;
    private List<LineRenderer> _linePool = new();

    private void DrawSkeletonWorld(List<Vector3> keypoints)
    {
        // deactivate old lines
        foreach (var ln in _linePool) ln.gameObject.SetActive(false);

        var canvasRect = overlayCanvas.GetComponent<RectTransform>().rect;
        float canvasW = canvasRect.width;
        float canvasH = canvasRect.height;

        int numPairs = cocoPairs.GetLength(0);
        for (int i = 0; i < numPairs; i++)
        {
            int a = cocoPairs[i,0];
            int b = cocoPairs[i,1];

            if (a >= keypoints.Count || b >= keypoints.Count) continue;
            if (keypoints[a].z < 0.3f || keypoints[b].z < 0.3f) continue;

            // get normalized coords
            float ax = keypoints[a].x * canvasW;
            float ay = (1f - keypoints[a].y) * canvasH;
            float bx = keypoints[b].x * canvasW;
            float by = (1f - keypoints[b].y) * canvasH;

            LineRenderer line;
            if (i < _linePool.Count) line = _linePool[i];
            else
            {
                line = Instantiate(linePrefab, overlayCanvas);
                _linePool.Add(line);
            }

            line.gameObject.SetActive(true);
            Vector3[] pts = new Vector3[2]
            {
                new Vector3(ax, ay, 0),
                new Vector3(bx, by, 0)
            };
            line.SetPositions(pts);
        }
    }
    
    private void DrawSkeletonUI(List<Vector3> keypoints)
    {
        var canvasRect = overlayCanvas.GetComponent<RectTransform>().rect;
        float canvasW = canvasRect.width;
        float canvasH = canvasRect.height;

        foreach (var ln in _linePool) ln.gameObject.SetActive(false);

        int numPairs = cocoPairs.GetLength(0);
        for (int i = 0; i < numPairs; i++)
        {
            int a = cocoPairs[i,0];
            int b = cocoPairs[i,1];

            if (a >= keypoints.Count || b >= keypoints.Count) continue;
            if (keypoints[a].z < 0.3f || keypoints[b].z < 0.3f) continue;

            float ax = keypoints[a].x * canvasW;
            float ay = (1f - keypoints[a].y) * canvasH;
            float bx = keypoints[b].x * canvasW;
            float by = (1f - keypoints[b].y) * canvasH;

            RectTransform line;
            if (i < _lineUiPool.Count) line = _lineUiPool[i];
            else
            {
                line = Instantiate(lineUiPrefab, overlayCanvas);
                _lineUiPool.Add(line);
            }

            line.gameObject.SetActive(true);

            // start at point A
            line.anchoredPosition = new Vector2(ax, ay);

            // direction vector
            Vector2 dir = new Vector2(bx - ax, by - ay);
            float length = dir.magnitude;

            // stretch and rotate
            line.sizeDelta = new Vector2(length, 3f); // 3px thick
            float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;
            line.localRotation = Quaternion.Euler(0, 0, angle);
        }
    }

    
    
    private void OnDestroy()
    {
        _worker?.Dispose();
    }
}
