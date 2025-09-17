// SkeletonDefinition.cs
using UnityEngine;

[CreateAssetMenu(fileName="SkeletonDefinition", menuName="BC/Skeleton Definition")]
public class SkeletonDefinition : ScriptableObject
{
    [Tooltip("Readable skeleton name, e.g. Human-COCO-17")]
    public string skeletonName;

    [Tooltip("Joint names in the detector/lifter order.")]
    public string[] jointNames;

    [Tooltip("Pairs of indices (start,end) for debug drawing.")]
    public Vector2Int[] bonePairs;

    public int NumJoints => jointNames?.Length ?? 0;
}

/*
 COCO human bone pairs
 * public Vector2Int[] bonePairs = new Vector2Int[] {
   // Face
   new Vector2Int(0,1), new Vector2Int(0,2),
   new Vector2Int(1,3), new Vector2Int(2,4),
   
   // Shoulders & torso
   new Vector2Int(5,6),      // LShoulder–RShoulder
   new Vector2Int(5,11),     // LShoulder–LHip
   new Vector2Int(6,12),     // RShoulder–RHip
   new Vector2Int(11,12),    // LHip–RHip
   
   // Left arm
   new Vector2Int(5,7), new Vector2Int(7,9),
   
   // Right arm
   new Vector2Int(6,8), new Vector2Int(8,10),
   
   // Left leg
   new Vector2Int(11,13), new Vector2Int(13,15),
   
   // Right leg
   new Vector2Int(12,14), new Vector2Int(14,16),
   };
   
*/

/* Human-H36M-17 (Martinez order) bone pairs
public Vector2Int[] bonePairs = new Vector2Int[] {
   // Pelvis → legs
   new Vector2Int(0,1), new Vector2Int(1,2), new Vector2Int(2,3), // right leg
   new Vector2Int(0,4), new Vector2Int(4,5), new Vector2Int(5,6), // left leg
   
   // Spine
   new Vector2Int(0,7), new Vector2Int(7,8), new Vector2Int(8,9), new Vector2Int(9,10),
   
   // Left arm
   new Vector2Int(8,11), new Vector2Int(11,12), new Vector2Int(12,13),
   
   // Right arm
   new Vector2Int(8,14), new Vector2Int(14,15), new Vector2Int(15,16),
   };

   */