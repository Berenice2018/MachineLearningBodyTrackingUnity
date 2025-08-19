using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Collections;

public class SentisYOLODetector : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;  // YOLOv11 .sentis or .onnx
    public BackendType backendType = BackendType.GPUCompute;
    public int inputImageSize = 640;

    [Header("Input Source")]
    public VideoCapture videoCapture;  // video texture

    [Header("UI")]
    public RectTransform boundingBoxPrefab;  // prefab with Image/Outline
    public Transform overlayCanvas;          // parent canvas or panel
    public RenderTexture CroppedTexture { get; private set; }
    [SerializeField] private int cropSize = 448; // match SentisRunner input
    
    private Model _model;
    private Worker _worker;
    private string _inputName;
    private string _outputName;
    private List<RectTransform> _boxPool = new();

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
            StartCoroutine(RunYOLO(videoCapture.MainTexture));
    }

    private IEnumerator RunYOLO(Texture tex)
    {
        using var input = TextureConverter.ToTensor(tex, inputImageSize, inputImageSize, 3);
        _worker.SetInput(_inputName, input);

        _worker.Schedule();
        yield return null;

        /*using var output = _worker.PeekOutput(_outputName).ReadbackAndClone() as Tensor<float>;
        List<(Rect box, int classId, float score)> results = DecodeYOLOv11(output);
// pass only Rects
        List<Rect> rects = results.ConvertAll(r => r.box);
        DrawBoxes(rects);*/
        
        using var output = _worker.PeekOutput(_outputName).ReadbackAndClone() as Tensor<float>;

        Rect? human = DetectHuman(output);

        //PrintHumanDetection(output);
        //DrawHumanBox(human);
        
        if (human.HasValue)
        {
            CropAndStore(tex, human.Value);
        }
    }

    private List<(Rect box, int classId, float score)> DecodeYOLOv11(Tensor<float> output, float confThresh = 0.4f, float iouThresh = 0.45f)
    {
        var results = new List<(Rect, int, float)>();
        var data = output.DownloadToArray();
        int numClasses = output.shape[1] - 4;   // 80
        int numBoxes = output.shape[2];         // 8400

        for (int i = 0; i < numBoxes; i++)
        {
            float x = data[0 * numBoxes + i]; // center x
            float y = data[1 * numBoxes + i]; // center y
            float w = data[2 * numBoxes + i];
            float h = data[3 * numBoxes + i];

            // class scores
            int bestClass = -1;
            float bestScore = 0f;
            for (int c = 0; c < numClasses; c++)
            {
                float score = data[(4 + c) * numBoxes + i];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            if (bestScore < confThresh) continue;

            // convert xywh → rect
            float xMin = (x - w / 2f) / inputImageSize;
            float yMin = (y - h / 2f) / inputImageSize;
            float boxW = w / inputImageSize;
            float boxH = h / inputImageSize;

            results.Add((new Rect(xMin, yMin, boxW, boxH), bestClass, bestScore));
        }

        return NMS(results, iouThresh);
    }

    private List<(Rect, int, float)> NMS(List<(Rect box, int classId, float score)> boxes, float iouThresh)
    {
        var final = new List<(Rect, int, float)>();
        boxes.Sort((a, b) => b.score.CompareTo(a.score));

        while (boxes.Count > 0)
        {
            var best = boxes[0];
            final.Add(best);
            boxes.RemoveAt(0);

            boxes.RemoveAll(b =>
                b.classId == best.classId &&
                IoU(b.box, best.box) > iouThresh);
        }

        return final;
    }

    private float IoU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.xMin, b.xMin);
        float y1 = Mathf.Max(a.yMin, b.yMin);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);

        float inter = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        float union = a.width * a.height + b.width * b.height - inter;
        return inter / union;
    }

    private Rect? DetectHuman(Tensor<float> output, float confThresh = 0.4f)
    {
        var data = output.DownloadToArray();
        int numClasses = output.shape[1] - 4;   // e.g. 80
        int numBoxes = output.shape[2];         // e.g. 8400

        int targetClass = 0; // COCO class 0 = person
        float bestScore = 0f;
        Rect bestBox = new Rect();

        for (int i = 0; i < numBoxes; i++)
        {
            float x = data[0 * numBoxes + i];
            float y = data[1 * numBoxes + i];
            float w = data[2 * numBoxes + i];
            float h = data[3 * numBoxes + i];

            float score = data[(4 + targetClass) * numBoxes + i];
            if (score < confThresh || score < bestScore) continue;

            // normalize to [0–1]
            float xMin = (x - w / 2f) / inputImageSize;
            float yMin = (y - h / 2f) / inputImageSize;
            float boxW = w / inputImageSize;
            float boxH = h / inputImageSize;

            bestScore = score;
            bestBox = new Rect(xMin, yMin, boxW, boxH);
        }

        if (bestScore > 0f)
            return bestBox;

        return null;
    }

    private void DrawHumanBox(Rect? humanBox)
    {
        // Hide all previous boxes
        foreach (var b in _boxPool) b.gameObject.SetActive(false);

        if (humanBox.HasValue)
        {
            Rect rect = humanBox.Value;

            RectTransform box;
            if (_boxPool.Count == 0)
            {
                box = Instantiate(boundingBoxPrefab, overlayCanvas);
                _boxPool.Add(box);
            }
            else
            {
                box = _boxPool[0];
            }

            box.gameObject.SetActive(true);

            // Map normalized box [0–1] to canvas anchors
            box.anchorMin = new Vector2(rect.xMin, rect.yMin);
            box.anchorMax = new Vector2(rect.xMax, rect.yMax);
            box.offsetMin = box.offsetMax = Vector2.zero;
        }
    }

    
    private void CropAndStore(Texture source, Rect humanBox)
{
    // Expand YOLO box by 20% in both directions
    float expandFactor = 0.2f;
    float newX = humanBox.xMin - humanBox.width * expandFactor * 0.5f;
    float newY = humanBox.yMin - humanBox.height * expandFactor * 0.5f;
    float newW = humanBox.width  * (1f + expandFactor);
    float newH = humanBox.height * (1f + expandFactor);

    // Clamp to [0–1] range
    newX = Mathf.Clamp01(newX);
    newY = Mathf.Clamp01(newY);
    if (newX + newW > 1f) newW = 1f - newX;
    if (newY + newH > 1f) newH = 1f - newY;

    // Convert normalized rect → pixel coords
    int px = Mathf.RoundToInt(newX * source.width);
    int py = Mathf.RoundToInt(newY * source.height);
    int pw = Mathf.RoundToInt(newW * source.width);
    int ph = Mathf.RoundToInt(newH * source.height);

    // Maintain aspect ratio: choose the larger side and make square crop
    int side = Mathf.Max(pw, ph);
    // expand crop region to a square, centered on original box
    int cx = px + pw / 2;
    int cy = py + ph / 2;
    px = Mathf.Clamp(cx - side / 2, 0, source.width - side);
    py = Mathf.Clamp(cy - side / 2, 0, source.height - side);
    pw = Mathf.Min(side, source.width - px);
    ph = Mathf.Min(side, source.height - py);

    // Lazy-init RT
    if (CroppedTexture == null || CroppedTexture.width != cropSize || CroppedTexture.height != cropSize)
    {
        CroppedTexture = new RenderTexture(cropSize, cropSize, 0, RenderTextureFormat.ARGB32);
    }

    // Copy square region from source → CroppedTexture (scaled down to cropSize)
    RenderTexture.active = CroppedTexture;
    GL.PushMatrix();
    GL.LoadPixelMatrix(0, cropSize, cropSize, 0);
    Graphics.DrawTexture(
        new Rect(0, 0, cropSize, cropSize),     // target rect
        source,
        new Rect((float)px / source.width, (float)py / source.height,
                 (float)pw / source.width, (float)ph / source.height), // source rect
        0, 0, 0, 0);
    GL.PopMatrix();
    RenderTexture.active = null;
}


    private void DrawBoxes(List<Rect> boxes)
    {
        // recycle existing pool
        foreach (var b in _boxPool) b.gameObject.SetActive(false);

        for (int i = 0; i < boxes.Count; i++)
        {
            RectTransform box;
            if (i < _boxPool.Count) box = _boxPool[i];
            else
            {
                box = Instantiate(boundingBoxPrefab, overlayCanvas);
                _boxPool.Add(box);
            }

            box.gameObject.SetActive(true);

            // place box in overlay canvas (assuming RawImage is full-screen)
            var rect = boxes[i];
            box.anchorMin = new Vector2(rect.xMin, rect.yMin);
            box.anchorMax = new Vector2(rect.xMax, rect.yMax);
            box.offsetMin = box.offsetMax = Vector2.zero;
        }
    }

    private void PrintHumanDetection(Tensor<float> output, float confThresh = 0.4f)
    {
        var data = output.DownloadToArray();
        int numClasses = output.shape[1] - 4;   // e.g. 80
        int numBoxes = output.shape[2];         // e.g. 8400

        int targetClass = 0; // COCO: 0 = person
        float bestScore = 0f;
        Rect bestBox = new Rect();

        for (int i = 0; i < numBoxes; i++)
        {
            float x = data[0 * numBoxes + i];
            float y = data[1 * numBoxes + i];
            float w = data[2 * numBoxes + i];
            float h = data[3 * numBoxes + i];

            float score = data[(4 + targetClass) * numBoxes + i];
            if (score < confThresh || score < bestScore) continue;

            // normalize coords to [0–1]
            float xMin = (x - w / 2f) / inputImageSize;
            float yMin = (y - h / 2f) / inputImageSize;
            float boxW = w / inputImageSize;
            float boxH = h / inputImageSize;

            bestScore = score;
            bestBox = new Rect(xMin, yMin, boxW, boxH);
        }

        if (bestScore > 0f)
        {
            Debug.Log($"Human detected: Conf {bestScore:F2}, " +
                      $"BBox (x:{bestBox.x:F2}, y:{bestBox.y:F2}, w:{bestBox.width:F2}, h:{bestBox.height:F2})");
        }
        else
        {
            Debug.Log("No human detected");
        }
    }

    
    private void OnDestroy()
    {
        _worker?.Dispose();
    }
}
