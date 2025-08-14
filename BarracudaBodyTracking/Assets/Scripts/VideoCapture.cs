using UnityEngine;
using UnityEngine.Video;
using UnityEngine.Experimental.Rendering; // For GraphicsFormat & FormatUsage

public class VideoCapture : MonoBehaviour
{
    public GameObject VideoBackground;
    public float VideoBackgroundScale;
    public LayerMask _layer;
    public bool UseWebCam = true;
    public int WebCamIndex = 0;
    public VideoPlayer VideoPlayer;

    private WebCamTexture webCamTexture;
    private RenderTexture videoTexture;
    private GraphicsFormat mainGraphicsFormat;

    private int videoScreenWidth = 2560;
    private int bgWidth, bgHeight;

    public RenderTexture MainTexture { get; private set; }

    private void Awake()
    {
        // Detect format support only once
        GraphicsFormat preferred = GraphicsFormat.B5G6R5_UNormPack16; // maps to RGB565
        if (!SystemInfo.IsFormatSupported(preferred, FormatUsage.Render))
        {
            Debug.LogWarning($"{preferred} not supported on this platform. Falling back to R8G8B8A8_UNorm.");
            preferred = GraphicsFormat.R8G8B8A8_UNorm;
        }
        mainGraphicsFormat = preferred;
    }

    public void Init(int bgWidth, int bgHeight)
    {
        this.bgWidth = bgWidth;
        this.bgHeight = bgHeight;

        if (UseWebCam) CameraPlayStart();
        else VideoPlayStart();
    }

    public void CameraPlayStart()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length <= WebCamIndex)
        {
            WebCamIndex = 0;
        }

        webCamTexture = new WebCamTexture(devices[WebCamIndex].name);
        webCamTexture.Play();

        var aspect = (float)webCamTexture.width / webCamTexture.height;
        VideoBackground.transform.localScale = new Vector3(aspect, 1, 1) * VideoBackgroundScale;
        VideoBackground.GetComponent<Renderer>().material.mainTexture = webCamTexture;

        InitMainTexture();
    }

    public void VideoPlayStart()
    {
        var desc = new RenderTextureDescriptor(
            (int)VideoPlayer.clip.width,
            (int)VideoPlayer.clip.height
        )
        {
            depthBufferBits = 24,
            graphicsFormat = mainGraphicsFormat,
            msaaSamples = 1,
            mipCount = 1,
            sRGB = true
        };

        videoTexture = new RenderTexture(desc);

        VideoPlayer.renderMode = VideoRenderMode.RenderTexture;
        VideoPlayer.targetTexture = videoTexture;
        VideoPlayer.Play();

        var aspect = (float)videoTexture.width / videoTexture.height;
        VideoBackground.transform.localScale = new Vector3(aspect, 1, 1) * VideoBackgroundScale;
        VideoBackground.GetComponent<Renderer>().material.mainTexture = videoTexture;

        InitMainTexture();
    }

    private void InitMainTexture()
    {
        GameObject go = new GameObject("MainTextureCamera", typeof(Camera));

        go.transform.parent = VideoBackground.transform;
        go.transform.localScale = new Vector3(-1.0f, -1.0f, 1.0f);
        go.transform.localPosition = new Vector3(0.0f, 0.0f, -2.0f);
        go.transform.localEulerAngles = Vector3.zero;
        go.layer = _layer;

        var camera = go.GetComponent<Camera>();
        camera.orthographic = true;
        camera.orthographicSize = 0.5f;
        camera.depth = -5;
        camera.depthTextureMode = DepthTextureMode.None;
        camera.clearFlags = CameraClearFlags.Color;
        camera.backgroundColor = Color.black;
        camera.cullingMask = _layer;
        camera.useOcclusionCulling = false;
        camera.nearClipPlane = 1.0f;
        camera.farClipPlane = 5.0f;
        camera.allowMSAA = false;
        camera.allowHDR = false;

        var desc = new RenderTextureDescriptor(bgWidth, bgHeight)
        {
            depthBufferBits = 0,
            graphicsFormat = mainGraphicsFormat,
            msaaSamples = 1,
            mipCount = 1,
            sRGB = true
        };

        MainTexture = new RenderTexture(desc)
        {
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Point,
        };

        camera.targetTexture = MainTexture;
    }
}
