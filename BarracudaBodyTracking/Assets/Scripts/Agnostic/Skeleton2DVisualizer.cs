using UnityEngine;
using UnityEngine.UI;

public class Skeleton2DVisualizer : MonoBehaviour
{
    [Header("References")]
    public RectTransform canvasRect;     // parent canvas (Screen Space overlay canvas)
    public GameObject pointPrefab;       // small UI circle prefab
    public GameObject linePrefab;        // thin UI image prefab

    [Header("Settings")]
    [Range(0f, 1f)] public float confidenceThreshold = 0.3f;
    public int targetWidth = 1280;   // camera/video width
    public int targetHeight = 720;   // camera/video height
    public bool flipY = true;

    private RectTransform[] pointMarkers;
    private RectTransform[] lineMarkers;

    private SkeletonDefinition definition;

    public void Init(SkeletonDefinition def)
    {
        definition = def;

        // Clean up old markers if re-initialized
        if (pointMarkers != null)
        {
            foreach (var p in pointMarkers) if (p) Destroy(p.gameObject);
            foreach (var l in lineMarkers) if (l) Destroy(l.gameObject);
        }

        // Create point markers
        pointMarkers = new RectTransform[def.NumJoints];
        for (int i = 0; i < def.NumJoints; i++)
        {
            var go = Instantiate(pointPrefab, canvasRect);
            go.name = $"Point_{def.jointNames[i]}";
            pointMarkers[i] = go.GetComponent<RectTransform>();
        }

        // Create line markers
        lineMarkers = new RectTransform[def.bonePairs.Length];
        for (int i = 0; i < def.bonePairs.Length; i++)
        {
            var go = Instantiate(linePrefab, canvasRect);
            go.name = $"Bone_{def.bonePairs[i].x}-{def.bonePairs[i].y}";
            lineMarkers[i] = go.GetComponent<RectTransform>();
        }
    }

    /// <summary>
    /// Pass in normalized (x,y,conf) for each joint. Length must equal definition.NumJoints.
    /// </summary>
    public void SetKeypoints(Vector3[] keypoints)
    {
        if (definition == null || keypoints == null || keypoints.Length != definition.NumJoints)
        {
            Debug.LogWarning("Skeleton2DVisualizer: definition mismatch or missing keypoints.");
            return;
        }

        // Points
        for (int i = 0; i < definition.NumJoints; i++)
        {
            bool visible = keypoints[i].z >= confidenceThreshold;
            pointMarkers[i].gameObject.SetActive(visible);
            if (!visible) continue;

            Vector2 pos = NormalizedToPixel(keypoints[i], targetWidth, targetHeight, flipY);
            pointMarkers[i].anchoredPosition = pos;
        }

        // Bones
        for (int e = 0; e < definition.bonePairs.Length; e++)
        {
            int a = definition.bonePairs[e].x;
            int b = definition.bonePairs[e].y;
            if (a < 0 || a >= keypoints.Length || b < 0 || b >= keypoints.Length) continue;

            bool visible = keypoints[a].z >= confidenceThreshold && keypoints[b].z >= confidenceThreshold;
            lineMarkers[e].gameObject.SetActive(visible);
            if (!visible) continue;

            Vector2 posA = NormalizedToPixel(keypoints[a], targetWidth, targetHeight, flipY);
            Vector2 posB = NormalizedToPixel(keypoints[b], targetWidth, targetHeight, flipY);

            DrawLine(lineMarkers[e], posA, posB);
        }
    }

    public static Vector2 NormalizedToPixel(Vector3 kp, int imageWidth, int imageHeight, bool flipY)
    {
        float px = kp.x * imageWidth;
        float py = kp.y * imageHeight;
        if (flipY) py = imageHeight - py;
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
