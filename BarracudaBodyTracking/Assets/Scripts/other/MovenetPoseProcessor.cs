using UnityEngine;

public class MovenetPoseProcessor : MonoBehaviour
{
    public Camera cam;
    public SkeletonModel skeletonModel;
    private Vector2[] _jointPositions;
    private float[] _jointConfidences;
    private const int NumJoints = 17;

    [Tooltip("Minimum confidence required for a joint to be used")]
    public float confidenceThreshold = 0.3f;

    public void InitializeParameters()
    {
        _jointPositions = new Vector2[NumJoints];
        _jointConfidences = new float[NumJoints];
        if (skeletonModel != null)
            skeletonModel.Init();
    }

    public void UploadNetworkOutputs(float[] data)
    {
        for (int i = 0; i < NumJoints; i++)
        {
            int offset = i * 3;
            float y = data[offset + 0];
            float x = data[offset + 1];
            float conf = data[offset + 2];

            _jointPositions[i] = new Vector2(x, y);
            _jointConfidences[i] = conf;
        }

        if (skeletonModel != null)
            MapToSkeletonModel();
    }

    private void MapToSkeletonModel()
    {
        var joints = skeletonModel.JointPoints;

        // --- Face ---
        SetJoint(PositionIndex.Nose, 0);
        SetJoint(PositionIndex.lEye, 1);
        SetJoint(PositionIndex.rEye, 2);
        SetJoint(PositionIndex.lEar, 3);
        SetJoint(PositionIndex.rEar, 4);

        // --- Upper body ---
        SetJoint(PositionIndex.lShldrBend, 5);   // left shoulder
        SetJoint(PositionIndex.rShldrBend, 6);   // right shoulder
        SetJoint(PositionIndex.lForearmBend, 7); // left elbow
        SetJoint(PositionIndex.rForearmBend, 8); // right elbow
        SetJoint(PositionIndex.lHand, 9);        // left wrist
        SetJoint(PositionIndex.rHand, 10);       // right wrist

        // --- Lower body ---
        // MoveNet 11+12 = hips → average as pelvis center
        if (_jointConfidences[11] >= confidenceThreshold &&
            _jointConfidences[12] >= confidenceThreshold)
        {
            Vector3 leftHip = To3D(_jointPositions[11]);
            Vector3 rightHip = To3D(_jointPositions[12]);
            Vector3 hipCenter = (leftHip + rightHip) * 0.5f;

            joints[PositionIndex.hip.Int()].Pos3D = hipCenter;

            // Use left/right hip as "anchors" for thighs
            joints[PositionIndex.lThighBend.Int()].Pos3D = leftHip;
            joints[PositionIndex.rThighBend.Int()].Pos3D = rightHip;
        }

        SetJoint(PositionIndex.lShin, 13); // left knee
        SetJoint(PositionIndex.rShin, 14); // right knee
        SetJoint(PositionIndex.lFoot, 15); // left ankle
        SetJoint(PositionIndex.rFoot, 16); // right ankle
    }

    private void SetJoint(PositionIndex idx, int movenetIndex)
    {
        if (_jointConfidences[movenetIndex] < confidenceThreshold)
            return;

        var joints = skeletonModel.JointPoints;
        joints[idx.Int()].Pos3D = To3D(_jointPositions[movenetIndex]);
    }
    
    private Vector3 To3D(Vector2 joint)
    {
        // MoveNet normalized [0..1] → image-space (192x192)
        float px = joint.x * 192f;
        float py = (1f - joint.y) * 192f; // flip Y so bottom = 0
        return new Vector3(px, py, 0f);
    }

    /*private Vector3 To3D(Vector2 joint)
    {
        // Scale normalized coords into actual screen pixels
        float px = joint.x * Screen.width;
        float py = (1f - joint.y) * Screen.height;

        // Project into world space, ~2m in front of camera
        Vector3 screenPos = new Vector3(px, py, 3f);
        return cam.ScreenToWorldPoint(screenPos);
    }
    */
    /// <summary>
    /// Draws MoveNet skeleton in Scene view for debugging
    /// </summary>
    private void OnDrawGizmos()
    {
        if (_jointPositions == null) return;

        Gizmos.color = Color.red;
        for (int i = 0; i < NumJoints; i++)
        {
            if (_jointConfidences != null && _jointConfidences[i] < confidenceThreshold)
                continue;

            Gizmos.DrawSphere(To3D(_jointPositions[i]), 0.02f);
        }

        Gizmos.color = Color.green;
        DrawBone(5, 7);   // left shoulder → left elbow
        DrawBone(7, 9);   // left elbow → left wrist
        DrawBone(6, 8);   // right shoulder → right elbow
        DrawBone(8, 10);  // right elbow → right wrist

        DrawBone(11, 13); // left hip → left knee
        DrawBone(13, 15); // left knee → left ankle
        DrawBone(12, 14); // right hip → right knee
        DrawBone(14, 16); // right knee → right ankle

        DrawBone(5, 6);   // shoulders
        DrawBone(11, 12); // hips
        DrawBone(5, 11);  // left shoulder → left hip
        DrawBone(6, 12);  // right shoulder → right hip
        DrawBone(0, 1);   // nose → left eye
        DrawBone(0, 2);   // nose → right eye
        DrawBone(1, 3);   // left eye → left ear
        DrawBone(2, 4);   // right eye → right ear
    }

    private void DrawBone(int i1, int i2)
    {
        if (_jointConfidences[i1] < confidenceThreshold || _jointConfidences[i2] < confidenceThreshold)
            return;

        Gizmos.DrawLine(To3D(_jointPositions[i1]), To3D(_jointPositions[i2]));
    }
}
