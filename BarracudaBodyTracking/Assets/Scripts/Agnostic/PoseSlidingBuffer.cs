using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Sliding window buffer for 2D pose keypoints (for VideoPose3D).
/// Each element is [joints x 2] (x,y).
/// Output is [1, frames, 17, 2] tensor for Sentis.
/// </summary>
public class PoseSlidingBuffer
{
    private int windowSize;
    private int numJoints = 17;
    private Queue<Vector2[]> buffer;

    public PoseSlidingBuffer(int windowSize)
    {
        this.windowSize = windowSize;
        buffer = new Queue<Vector2[]>(windowSize);
    }

    /// <summary>
    /// Push a new frame of 2D joints into the buffer.
    /// </summary>
    public void AddFrame(Vector2[] joints2D)
    {
        Debug.Log("### AddFrame to buffer");
        if (joints2D.Length != numJoints)
        {
            Debug.LogError($"Expected {numJoints} joints, got {joints2D.Length}");
            return;
        }

        if (buffer.Count >= windowSize)
            buffer.Dequeue();

        buffer.Enqueue(joints2D);
    }

    /// <summary>
    /// Check if the buffer is "full" and ready for inference.
    /// </summary>
    public bool IsReady()
    {
        return buffer.Count == windowSize;
    }

    /// <summary>
    /// Export the buffer as a float array [1, frames, 17, 2].
    /// Unity Sentis can wrap this into a tensor.
    /// </summary>
    public float[,,,] ToTensor()
    {
        float[,,,] tensor = new float[1, windowSize, numJoints, 2];
        Vector2[][] frames = buffer.ToArray();

        for (int f = 0; f < windowSize; f++)
        {
            for (int j = 0; j < numJoints; j++)
            {
                tensor[0, f, j, 0] = frames[f][j].x;
                tensor[0, f, j, 1] = frames[f][j].y;
            }
        }

        return tensor;
    }
}
