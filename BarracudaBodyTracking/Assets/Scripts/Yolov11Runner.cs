using System;
using System.Collections;
using System.Collections.Generic;
using PoseDetection;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Serialization;

public class Yolov11Runner : MonoBehaviour
{
    [Header("Model Settings")]
    [Tooltip("The ResNet model asset for pose estimation")]
    public ModelAsset modelAsset;

    [FormerlySerializedAs("imageSize")] public int inputImageSize = 640;
    
    // In your pose detection script
    public YOLOv11HumanDetector humanDetector;
    private RenderTexture _videoTexture; // Your camera/video input
    
    [Tooltip("Video capture component for input")]
    public VideoCapture videoCapture;

    private bool _ready;
    
    private void Awake()
    {
        _videoTexture = videoCapture.MainTexture;
    }

    private void Start()
    {
        videoCapture.Init(inputImageSize, inputImageSize);
        _ready = true;
    }

    void Update()
    {
        if (!_ready) return;
        
        //ProcessFrame();
        YOLOv11HumanDetector.DetectionResult human = humanDetector.DetectHuman(_videoTexture);
    
        if (human.isValid)
        {
            // Use bounding box to crop region for pose detection
            Rect screenBox = humanDetector.GetScreenSpaceBoundingBox(
                human, _videoTexture.width, _videoTexture.height);
        
            Debug.Log($"screen box height: {screenBox.height} width: {screenBox.width}" );
            // Feed cropped region to your ResNet pose detector
            //ProcessPoseInRegion(screenBox);
        }
    }

}
