using UnityEngine;

public class Coco2DVisualizer : MonoBehaviour
{
    [Header("Debug")]
    public bool showSkeleton = true;
    public bool showLabels = true;
    [Range(0f, 1f)] public float confidenceThreshold = 0.3f;
    public Color lineColor = Color.cyan;
    public Color labelColor = Color.yellow;
    public float pointSize = 0.02f;
    public float scale = 192f; // match MoveNet input resolution
    public Vector3 offset = new Vector3(0, 0, 5f); // push in front of camera

    private static readonly string[] MoveNetJointNames = new string[]
    {
        "nose", "lEye", "rEye", "lEar", "rEar",
        "lShoulder", "rShoulder", "lElbow", "rElbow",
        "lWrist", "rWrist", "lHip", "rHip",
        "lKnee", "rKnee", "lAnkle", "rAnkle"
    };

    private static readonly int[,] CocoEdges = new int[,]
    {
        {0,1}, {0,2}, {1,3}, {2,4},        // face
        {5,6},                             // shoulders
        {5,7}, {7,9},                      // left arm
        {6,8}, {8,10},                     // right arm
        {11,12},                           // hips
        {5,11}, {6,12},                    // torso sides
        {11,13}, {13,15},                  // left leg
        {12,14}, {14,16}                   // right leg
    };

    private Vector2[] keypoints;
    private float[] confidences;

    /// <summary>
    /// Call this with raw MoveNet keypoints [17*3].
    /// Format: (y, x, confidence).
    /// </summary>
    public void SetKeypoints(float[] keypoints2D)
    {
        keypoints = new Vector2[17];
        confidences = new float[17];

        for (int i = 0; i < 17; i++)
        {
            float y = keypoints2D[i * 3 + 0];
            float x = keypoints2D[i * 3 + 1];
            float c = keypoints2D[i * 3 + 2];

            keypoints[i] = new Vector2(x, y);
            confidences[i] = c;
        }
    }

    void OnDrawGizmos()
    {
        if (!showSkeleton || keypoints == null) return;

        Gizmos.color = lineColor;

#if UNITY_EDITOR
        GUIStyle style = new GUIStyle();
        style.normal.textColor = labelColor;
        style.fontSize = 14;
#endif

        // Draw points + labels
        for (int i = 0; i < keypoints.Length; i++)
        {
            if (confidences[i] < confidenceThreshold) continue;

            Vector3 p = new Vector3(
                (keypoints[i].x - 0.5f) * scale,
                -(keypoints[i].y - 0.5f) * scale,
                0
            ) + offset;

            Gizmos.DrawSphere(p, pointSize);

#if UNITY_EDITOR
            if (showLabels)
            {
                Vector3 labelPos = p + new Vector3(0.03f, 0.02f, 0);
                UnityEditor.Handles.Label(labelPos, MoveNetJointNames[i], style);
            }
#endif
        }

        // Draw edges
        for (int i = 0; i < CocoEdges.GetLength(0); i++)
        {
            int a = CocoEdges[i, 0];
            int b = CocoEdges[i, 1];
            if (confidences[a] < confidenceThreshold || confidences[b] < confidenceThreshold)
                continue;

            Vector3 p1 = new Vector3(
                (keypoints[a].x - 0.5f) * scale,
                -(keypoints[a].y - 0.5f) * scale,
                0
            ) + offset;

            Vector3 p2 = new Vector3(
                (keypoints[b].x - 0.5f) * scale,
                -(keypoints[b].y - 0.5f) * scale,
                0
            ) + offset;

            Gizmos.DrawLine(p1, p2);
        }
    }
}
