using UnityEngine;

public class MovenetPoseProcessor : MonoBehaviour
{
    public SkeletonModel skeletonModel;
    private Vector2[] _jointPositions;
    private float[] _jointConfidences;
    private const int NumJoints = 17;

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

        // MoveNet indices → SkeletonModel.PositionIndex
        joints[PositionIndex.Nose.Int()].Pos3D         = To3D(_jointPositions[0]);
        joints[PositionIndex.lEye.Int()].Pos3D         = To3D(_jointPositions[1]);
        joints[PositionIndex.rEye.Int()].Pos3D         = To3D(_jointPositions[2]);
        joints[PositionIndex.lEar.Int()].Pos3D         = To3D(_jointPositions[3]);
        joints[PositionIndex.rEar.Int()].Pos3D         = To3D(_jointPositions[4]);

        joints[PositionIndex.lShldrBend.Int()].Pos3D   = To3D(_jointPositions[5]);
        joints[PositionIndex.rShldrBend.Int()].Pos3D   = To3D(_jointPositions[6]);
        joints[PositionIndex.lForearmBend.Int()].Pos3D = To3D(_jointPositions[7]);
        joints[PositionIndex.rForearmBend.Int()].Pos3D = To3D(_jointPositions[8]);
        joints[PositionIndex.lHand.Int()].Pos3D        = To3D(_jointPositions[9]);
        joints[PositionIndex.rHand.Int()].Pos3D        = To3D(_jointPositions[10]);

        joints[PositionIndex.lThighBend.Int()].Pos3D   = To3D(_jointPositions[11]);
        joints[PositionIndex.rThighBend.Int()].Pos3D   = To3D(_jointPositions[12]);
        joints[PositionIndex.lShin.Int()].Pos3D        = To3D(_jointPositions[13]);
        joints[PositionIndex.rShin.Int()].Pos3D        = To3D(_jointPositions[14]);
        joints[PositionIndex.lFoot.Int()].Pos3D        = To3D(_jointPositions[15]);
        joints[PositionIndex.rFoot.Int()].Pos3D        = To3D(_jointPositions[16]);

        // Optional: approximate abdomen/hip center as mid of left & right hips
        Vector3 hipCenter = (To3D(_jointPositions[11]) + To3D(_jointPositions[12])) * 0.5f;
        joints[PositionIndex.hip.Int()].Pos3D          = hipCenter;
    }

    private Vector3 To3D(Vector2 joint)
    {
        // Convert normalized [0..1] coords into 3D placeholder
        // Flip Y since MoveNet origin is top-left, Unity is bottom-left
        return new Vector3(joint.x, 1f - joint.y, 0f);
    }

    public void PredictPose()
    {
        // SkeletonModel.Update() will apply transforms automatically
    }
}
