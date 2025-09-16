using UnityEngine;
using UnityEngine.UI;

public class Coco2DVisualizer : MonoBehaviour
{
    [Header("References")]
    public RectTransform canvasRect;     // parent canvas (Screen Space)
    public GameObject pointPrefab;       // small UI circle prefab
    public GameObject linePrefab;        // thin UI image prefab

    [Range(0f, 1f)] public float confidenceThreshold = 0.3f;
    public int inputResolution = 192; // must match MoveNet input

    private RectTransform[] pointMarkers;
    private RectTransform[,] lineMarkers;

    private static readonly int[,] CocoEdges = new int[,]
    {
        {0,1}, {0,2}, {1,3}, {2,4},        // face
        {5,6},                             // shoulders
        {5,7}, {7,9},                      // left arm
        {6,8}, {8,10},                     // right arm
        {11,12},                           // hips
        {5,11}, {6,12},                    // torso
        {11,13}, {13,15},                  // left leg
        {12,14}, {14,16}                   // right leg
    };

    void Awake()
    {
        // Create 17 point markers
        pointMarkers = new RectTransform[17];
        for (int i = 0; i < 17; i++)
        {
            var go = Instantiate(pointPrefab, canvasRect);
            pointMarkers[i] = go.GetComponent<RectTransform>();
        }

        // Create line markers
        int edgeCount = CocoEdges.GetLength(0);
        lineMarkers = new RectTransform[edgeCount, 1];
        for (int i = 0; i < edgeCount; i++)
        {
            var go = Instantiate(linePrefab, canvasRect);
            lineMarkers[i,0] = go.GetComponent<RectTransform>();
        }
    }

    public int targetWidth = 1280;   // camera/video width
    public int targetHeight = 720;   // camera/video height

    public void SetKeypoints(Vector3[] keypoints)
    {
        for (int i = 0; i < 17; i++)
        {
            if (keypoints[i].z < confidenceThreshold)
            {
                pointMarkers[i].gameObject.SetActive(false);
                continue;
            }

            pointMarkers[i].gameObject.SetActive(true);

            // 🔥 convert normalized MoveNet coords to pixels
            Vector2 pos = MoveNetToPixel(keypoints[i], targetWidth, targetHeight, flipY:true);
            pointMarkers[i].anchoredPosition = pos;
        }

        // Edges: same conversion
        for (int e = 0; e < CocoEdges.GetLength(0); e++)
        {
            int a = CocoEdges[e,0];
            int b = CocoEdges[e,1];

            if (keypoints[a].z < confidenceThreshold || keypoints[b].z < confidenceThreshold)
            {
                lineMarkers[e,0].gameObject.SetActive(false);
                continue;
            }

            lineMarkers[e,0].gameObject.SetActive(true);

            Vector2 posA = MoveNetToPixel(keypoints[a], targetWidth, targetHeight, flipY:true);
            Vector2 posB = MoveNetToPixel(keypoints[b], targetWidth, targetHeight, flipY:true);

            DrawLine(lineMarkers[e,0], posA, posB);
        }
    }

    /// <summary>
    /// Convert MoveNet keypoints (x,y in [0,1]) into pixel coordinates
    /// matching the YOLO output space.
    /// </summary>
    public static Vector2 MoveNetToPixel(Vector3 kp, int imageWidth, int imageHeight, bool flipY = true)
    {
        float px = kp.x * imageWidth;
        float py = kp.y * imageHeight;

        if (flipY)
            py = imageHeight - py; // flip so (0,0)=top-left

        return new Vector2(px, py);
    }

    
    private Vector2 ToPixel(Vector3 kp)
    {
        // Convert normalized [0,1] into canvas space
        float px = kp.x * canvasRect.sizeDelta.x;
        float py = (kp.y) * canvasRect.sizeDelta.y;
        return new Vector2(px, py);
    }

    private void DrawLine(RectTransform line, Vector2 start, Vector2 end)
    {
        Vector2 diff = end - start;
        float dist = diff.magnitude;

        // Set line size (x = length, y = thickness)
        line.sizeDelta = new Vector2(dist, 2f);

        // Position at midpoint
        line.anchoredPosition = (start + end) * 0.5f;

        // Rotate to angle
        float angle = Mathf.Atan2(diff.y, diff.x) * Mathf.Rad2Deg;
        line.localRotation = Quaternion.Euler(0, 0, angle);
    }

}
