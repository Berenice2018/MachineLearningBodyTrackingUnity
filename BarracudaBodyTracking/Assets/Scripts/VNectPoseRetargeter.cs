using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Retargets VNect 3D pose keypoints to a Unity Humanoid avatar by driving the muscle system.
/// Designed for VNect joint order used in your project (24 joints).
/// Key ideas:
///  - Convert VNect joints to a hip-relative, height-normalized space.
///  - Compute per-bone aim directions (eg. shoulder→elbow) and map to muscles.
///  - Keep avatar root stable; rotate body from shoulders/hips, not raw pixel coords.
/// </summary>
[System.Serializable]
public class VNectPoseRetargeter : MonoBehaviour
{
    #region Public Configuration
    [Header("Avatar Configuration")]
    [Tooltip("Animator of the target humanoid avatar")]
    public Animator targetAvatar;

    [Tooltip("Optional: Draw debug lines for computed directions")]
    public bool debugDraw;

    [Header("Pose Scaling/Filtering")]
    public bool enableHeightNormalization = true;
    [Tooltip("Fallback avatar height (meters) if bone-based measure fails")] public float referenceHeight = 1.8f;
    [Tooltip("Overall scale for the retargeted motion in avatar space")] public float poseScale = 1f;
    [Range(0f,1f)] public float smoothing = 0.1f;
    #endregion

    #region Private State
    private HumanPoseHandler _poseHandler;
    private HumanPose _pose;
    private HumanPose _prevPose;
    private bool _ready;

    // Cached bones & default directions
    private readonly Dictionary<HumanBodyBones, Transform> _boneT = new();
    private readonly Dictionary<HumanBodyBones, Vector3> _boneDefaultDir = new();

    // Quick access to muscle limits
    private float[] _muscleMin; // HumanTrait.MuscleMin
    private float[] _muscleMax; // HumanTrait.MuscleMax

    // Map VNect joint index -> HumanBodyBones
    // NOTE: aligns with your existing VNect order
    private readonly Dictionary<int, HumanBodyBones> _map = new();

    // VNect joints we care about (indices must match your VNectModel)
    private const int J_Head=0, J_Neck=1,
                      J_RShoulder=2, J_RUpperArm=3, J_RLowerArm=4, J_RHand=5,
                      J_LShoulder=6, J_LUpperArm=7, J_LLowerArm=8, J_LHand=9,
                      J_Spine=10, J_Hips=11,
                      J_RElbow=4, J_LElbow=8,
                      J_RHip=12, J_RKnee=13, J_RFoot=14,
                      J_LHip=15, J_LKnee=16, J_LFoot=17;

    // Avatar height from Hips→Head at bind pose
    private float _avatarHeight;

    // Cached last joint set for smoothing/damping (hip-relative, avatar space)
    private Vector3[] _lastJoints;
    #endregion

    #region Unity
    private void Start()
    {
        Init();
    }

    private void OnDestroy()
    {
        if (_poseHandler != null) { _poseHandler = null; }
        _ready = false;
    }
    #endregion

    #region Init
    private void Init()
    {
        if (targetAvatar == null)
        {
            Debug.LogError("VNectPoseRetargeter: targetAvatar not assigned.");
            return;
        }
        if (!targetAvatar.isHuman)
        {
            Debug.LogError("VNectPoseRetargeter: target avatar must be Humanoid.");
            return;
        }

        _poseHandler = new HumanPoseHandler(targetAvatar.avatar, targetAvatar.transform);
        _pose = new HumanPose();
        _prevPose = new HumanPose();
        _poseHandler.GetHumanPose(ref _pose);
        _prevPose = _pose;

        CacheBones();
        CacheDefaultDirs();
        //CacheMuscleLimits();
        BuildVNectMap();
        MeasureAvatarHeight();

        _ready = true;
    }

    private void CacheBones()
    {
        HumanBodyBones[] bones =
        {
            HumanBodyBones.Hips,
            HumanBodyBones.Spine,
            HumanBodyBones.Chest,
            HumanBodyBones.UpperChest,
            HumanBodyBones.Neck,
            HumanBodyBones.Head,
            HumanBodyBones.RightShoulder,
            HumanBodyBones.RightUpperArm,
            HumanBodyBones.RightLowerArm,
            HumanBodyBones.RightHand,
            HumanBodyBones.LeftShoulder,
            HumanBodyBones.LeftUpperArm,
            HumanBodyBones.LeftLowerArm,
            HumanBodyBones.LeftHand,
            HumanBodyBones.RightUpperLeg,
            HumanBodyBones.RightLowerLeg,
            HumanBodyBones.RightFoot,
            HumanBodyBones.LeftUpperLeg,
            HumanBodyBones.LeftLowerLeg,
            HumanBodyBones.LeftFoot,
        };
        _boneT.Clear();
        foreach (var b in bones)
        {
            var t = targetAvatar.GetBoneTransform(b);
            if (t != null) _boneT[b] = t;
        }
    }

    private void CacheDefaultDirs()
    {
        _boneDefaultDir.Clear();
        // Default aim directions from current bind pose (bone → child center)
        CacheDefaultDir(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm);
        CacheDefaultDir(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm);
        CacheDefaultDir(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand);
        CacheDefaultDir(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand);
        CacheDefaultDir(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg);
        CacheDefaultDir(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg);
        CacheDefaultDir(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot);
        CacheDefaultDir(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot);
        CacheDefaultDir(HumanBodyBones.Spine, HumanBodyBones.Chest);
        CacheDefaultDir(HumanBodyBones.Neck, HumanBodyBones.Head);
    }

    private void CacheDefaultDir(HumanBodyBones parent, HumanBodyBones child)
    {
        if (!_boneT.TryGetValue(parent, out var p) || !_boneT.TryGetValue(child, out var c)) return;
        var dir = (c.position - p.position);
        if (dir.sqrMagnitude < 1e-6f) dir = Vector3.up; // fallback
        _boneDefaultDir[parent] = dir.normalized;
    }
    

    private void BuildVNectMap()
    {
        _map.Clear();
        _map[J_Head] = HumanBodyBones.Head;
        _map[J_Neck] = HumanBodyBones.Neck;
        _map[J_RShoulder] = HumanBodyBones.RightShoulder;
        _map[J_RUpperArm] = HumanBodyBones.RightUpperArm;
        _map[J_RLowerArm] = HumanBodyBones.RightLowerArm;
        _map[J_RHand] = HumanBodyBones.RightHand;
        _map[J_LShoulder] = HumanBodyBones.LeftShoulder;
        _map[J_LUpperArm] = HumanBodyBones.LeftUpperArm;
        _map[J_LLowerArm] = HumanBodyBones.LeftLowerArm;
        _map[J_LHand] = HumanBodyBones.LeftHand;
        _map[J_Spine] = HumanBodyBones.Spine; // abdomen upper approx
        _map[J_Hips] = HumanBodyBones.Hips;
        _map[J_RHip] = HumanBodyBones.RightUpperLeg;
        _map[J_RKnee] = HumanBodyBones.RightLowerLeg;
        _map[J_RFoot] = HumanBodyBones.RightFoot;
        _map[J_LHip] = HumanBodyBones.LeftUpperLeg;
        _map[J_LKnee] = HumanBodyBones.LeftLowerLeg;
        _map[J_LFoot] = HumanBodyBones.LeftFoot;
    }

    private void MeasureAvatarHeight()
    {
        Transform hips = targetAvatar.GetBoneTransform(HumanBodyBones.Hips);
        Transform head = targetAvatar.GetBoneTransform(HumanBodyBones.Head);
        _avatarHeight = (hips != null && head != null) ? Vector3.Distance(hips.position, head.position) : referenceHeight;
        if (_avatarHeight <= 0.05f) _avatarHeight = referenceHeight;
    }
    #endregion

    #region API
    /// <summary>
    /// Main entry from runner.
    /// </summary>
    public void ApplyPose(SkeletonModel.JointPoint[] joints)
    {
        if (!_ready || joints == null || joints.Length < 18) return;

        // Build hip-relative, height-normalized positions in AVATAR space
        var hip = joints[J_Hips].Pos3D;
        float detectedHeight = Vector3.Distance(joints[J_Hips].Pos3D, joints[J_Head].Pos3D);
        float scale = 1f;
        if (enableHeightNormalization && detectedHeight > 1e-2f)
            scale = (_avatarHeight / detectedHeight) * poseScale;
        else
            scale = poseScale;

        // Convert VNect space (image X-right, Y-up, Z-forward) to avatar local space.
        // We assume avatar forward == +Z, right == +X, up == +Y (Unity default).
        // Also make everything hip-relative so root stays stable.
        Vector3[] P = new Vector3[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 p = (joints[i].Pos3D - hip) * scale; // hip-relative
            P[i] = p;
        }

        // Smooth joints if requested (simple exponential smoothing in hip frame)
        if (_lastJoints == null || _lastJoints.Length != P.Length)
        {
            _lastJoints = (Vector3[])P.Clone();
        }
        else if (smoothing > 0f)
        {
            float a = smoothing;
            for (int i = 0; i < P.Length; i++)
                _lastJoints[i] = Vector3.Lerp(_lastJoints[i], P[i], 1f - a);
            P = _lastJoints;
        }

        // Get current pose, we will overwrite body & muscles
        _poseHandler.GetHumanPose(ref _pose);

        // Keep body at current hips (don’t snap to pixels). Small positional nudge from VNect hip if desired.
        // Here we keep position unchanged; feel free to expose a slider to blend.
        // _pose.bodyPosition += Vector3.Lerp(Vector3.zero, P[J_Hips], 0.1f);

        // Compute body rotation from shoulders & hips to align torso facing
        Quaternion bodyQ = ComputeBodyRotation(P);
        _pose.bodyRotation = bodyQ;

        // Arms
        DriveLimb(P, J_RShoulder, J_RUpperArm, J_RElbow, J_RHand,
            HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm);
        DriveLimb(P, J_LShoulder, J_LUpperArm, J_LElbow, J_LHand,
            HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm);

        // Legs
        DriveLimb(P, J_RHip, J_RHip, J_RKnee, J_RFoot,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, isLeg:true);
        DriveLimb(P, J_LHip, J_LHip, J_LKnee, J_LFoot,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, isLeg:true);

        // Spine & head
        DriveSpineAndHead(P);

        // Commit pose
        _poseHandler.SetHumanPose(ref _pose);
        _prevPose = _pose;

        if (debugDraw) DrawDebug(P);
    }
    #endregion

    #region Body & Limbs
    private Quaternion ComputeBodyRotation(Vector3[] P)
    {
        // Torso plane defined by hips and shoulders
        Vector3 hipMid = (P[J_LHip] + P[J_RHip]) * 0.5f;
        Vector3 shMid  = (P[J_LShoulder] + P[J_RShoulder]) * 0.5f;
        Vector3 up = (shMid - hipMid); if (up.sqrMagnitude < 1e-6f) up = Vector3.up;
        Vector3 right = (P[J_RShoulder] - P[J_LShoulder]); if (right.sqrMagnitude < 1e-6f) right = Vector3.right;
        Vector3 forward = Vector3.Cross(right.normalized, up.normalized);
        if (forward.sqrMagnitude < 1e-6f) forward = Vector3.forward;

        // Make a torso rotation with forward & up
        Quaternion q = Quaternion.LookRotation(forward, up);
        return q;
    }

    private void DriveLimb(
        Vector3[] P,
        int jParent, int jUpper, int jLower, int jEnd,
        HumanBodyBones upperBone, HumanBodyBones midBone, HumanBodyBones lowerBone,
        bool isLeg = false)
    {
        // Upper segment direction (shoulder→elbow or hip→knee)
        Vector3 dirUpper = (P[jLower] - P[jUpper]);
        // Lower segment direction (elbow→hand or knee→foot)
        Vector3 dirLower = (P[jEnd] - P[jLower]);

        if (dirUpper.sqrMagnitude < 1e-6f || dirLower.sqrMagnitude < 1e-6f) return;

        // Map to muscles by comparing with default aim directions in world space
        ApplyAimToMuscles(midBone, dirUpper);
        ApplyHingeToMuscle(lowerBone, dirUpper, dirLower, isLeg);

        // Optional: shoulder/upper-leg swing from parent (clavicle/hip) vector
        if (upperBone == HumanBodyBones.RightShoulder || upperBone == HumanBodyBones.LeftShoulder)
        {
            ApplyClavicleSwing(upperBone, (P[jUpper] - P[jParent]));
        }
        else if (upperBone == HumanBodyBones.RightUpperLeg || upperBone == HumanBodyBones.LeftUpperLeg)
        {
            ApplyHipSwing(upperBone, (P[jUpper] - P[jParent]));
        }
    }

    private void ApplyAimToMuscles(HumanBodyBones bone, Vector3 worldAimDir)
    {
        if (!_boneT.TryGetValue(bone, out var t)) return;
        Vector3 defaultDir = _boneDefaultDir.TryGetValue(bone, out var d) ? d : t.forward;

        // Build rotation from default to target aim
        Quaternion fromTo = Quaternion.FromToRotation(defaultDir, worldAimDir.normalized);

        // Convert to local (muscle) rotation around the bone
        // Approximate: project into bone local space
        Quaternion local = Quaternion.Inverse(t.rotation) * (fromTo * t.rotation);
        Vector3 e = ClampEuler(local.eulerAngles);

        // Map to DOFs: 0 = swing X, 1 = swing Y, 2 = twist
        SetBoneMuscle(bone, 0, e.x / 45f);
        SetBoneMuscle(bone, 1, e.y / 45f);
        SetBoneMuscle(bone, 2, e.z / 45f);
    }

    private void ApplyHingeToMuscle(HumanBodyBones bone, Vector3 upperDir, Vector3 lowerDir, bool isLeg)
    {
        // Hinge angle from the angle between adjacent segments
        float ang = Vector3.SignedAngle(upperDir.normalized, lowerDir.normalized, Vector3.Cross(upperDir, lowerDir));
        // Elbows/knees typically flex around one DOF. Use DOF 0 as proxy.
        float normalized = Mathf.Clamp(ang / 90f, -1f, 1f);
        SetBoneMuscle(bone, 0, normalized);
    }

    private void ApplyClavicleSwing(HumanBodyBones clavicle, Vector3 worldDir)
    {
        if (!_boneT.TryGetValue(clavicle, out var t)) return;
        Quaternion local = Quaternion.Inverse(t.rotation) * Quaternion.LookRotation(worldDir.normalized, t.up);
        Vector3 e = ClampEuler(local.eulerAngles);
        SetBoneMuscle(clavicle, 0, e.x / 45f);
        SetBoneMuscle(clavicle, 1, e.y / 45f);
        // leave twist mostly alone
    }

    private void ApplyHipSwing(HumanBodyBones hip, Vector3 worldDir)
    {
        if (!_boneT.TryGetValue(hip, out var t)) return;
        Quaternion local = Quaternion.Inverse(t.rotation) * Quaternion.LookRotation(worldDir.normalized, t.up);
        Vector3 e = ClampEuler(local.eulerAngles);
        SetBoneMuscle(hip, 0, e.x / 45f);
        SetBoneMuscle(hip, 1, e.y / 45f);
    }

    private void DriveSpineAndHead(Vector3[] P)
    {
        // Spine: aim hips→chest
        Vector3 hipMid = (P[J_LHip] + P[J_RHip]) * 0.5f;
        Vector3 shMid  = (P[J_LShoulder] + P[J_RShoulder]) * 0.5f;
        Vector3 spineDir = (shMid - hipMid);
        ApplyAimToMuscles(HumanBodyBones.Spine, spineDir);
        if (_boneT.ContainsKey(HumanBodyBones.Chest)) ApplyAimToMuscles(HumanBodyBones.Chest, spineDir);
        if (_boneT.ContainsKey(HumanBodyBones.UpperChest)) ApplyAimToMuscles(HumanBodyBones.UpperChest, spineDir);

        // Head: aim neck→head; add small look from nose if available
        Vector3 neckToHead = (P[J_Head] - P[J_Neck]);
        ApplyAimToMuscles(HumanBodyBones.Neck, neckToHead);
        // For head bone, prefer keeping twist modest
        if (_boneT.ContainsKey(HumanBodyBones.Head)) ApplyAimToMuscles(HumanBodyBones.Head, neckToHead);
    }
    #endregion

    #region Muscles helpers
    private void SetBoneMuscle(HumanBodyBones bone, int dof, float value)
    {
        int m = HumanTrait.MuscleFromBone((int)bone, dof);
        if (m < 0) return;
        float v = Mathf.Clamp(value, -1f, 1f);
        // Clamp to actual muscle min/max to avoid invalid states
        if (_muscleMin != null && _muscleMax != null && m < _muscleMin.Length && m < _muscleMax.Length)
            v = Mathf.Clamp(v, _muscleMin[m], _muscleMax[m]);
        _pose.muscles[m] = v;
    }

    private static Vector3 ClampEuler(Vector3 e)
    {
        // Convert 0..360 to -180..180 and gently clamp extremes
        e.x = Mathf.DeltaAngle(0f, e.x);
        e.y = Mathf.DeltaAngle(0f, e.y);
        e.z = Mathf.DeltaAngle(0f, e.z);
        return e;
    }
    #endregion

    #region Debug
    private void DrawDebug(Vector3[] P)
    {
        if (!_boneT.TryGetValue(HumanBodyBones.Hips, out var hipsT)) return;
        Vector3 origin = hipsT.position;

        // Draw a few main segments in world space
        DrawSeg(origin, origin + targetAvatar.transform.rotation * P[J_RUpperArm], Color.cyan);
        DrawSeg(origin, origin + targetAvatar.transform.rotation * P[J_LUpperArm], Color.cyan);
        DrawSeg(origin, origin + targetAvatar.transform.rotation * P[J_RHip], Color.yellow);
        DrawSeg(origin, origin + targetAvatar.transform.rotation * P[J_LHip], Color.yellow);

        // shoulders to elbows, hips to knees
        DrawPair(P[J_RUpperArm], P[J_RElbow], Color.blue, origin);
        DrawPair(P[J_LUpperArm], P[J_LElbow], Color.blue, origin);
        DrawPair(P[J_RHip], P[J_RKnee], Color.magenta, origin);
        DrawPair(P[J_LHip], P[J_LKnee], Color.magenta, origin);
    }

    private void DrawSeg(Vector3 a, Vector3 b, Color c)
    {
        Debug.DrawLine(a, b, c, 0f, false);
    }
    private void DrawPair(Vector3 a, Vector3 b, Color c, Vector3 origin)
    {
        Debug.DrawLine(origin + targetAvatar.transform.rotation * a,
                       origin + targetAvatar.transform.rotation * b, c, 0f, false);
    }
    #endregion
}
