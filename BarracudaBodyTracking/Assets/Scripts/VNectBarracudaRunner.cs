using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;

/// <summary>
/// Define Joint points
/// </summary>
public class VNectBarracudaRunner : MonoBehaviour
{
    /// <summary>
    /// Neural network model
    /// </summary>
    /// 
    public NNModel NNModel;

    public WorkerFactory.Type WorkerType = WorkerFactory.Type.Auto;
    public bool Verbose = true;

    public VNectModel VNectModel;

    public VideoCapture videoCapture;

    private Model _model;
    private IWorker _worker;

    /// <summary>
    /// Coordinates of joint points
    /// </summary>
    private VNectModel.JointPoint[] _jointPoints;
    
    /// <summary>
    /// Number of joint points
    /// </summary>
    private const int JointNum = 24;

    /// <summary>
    /// input image size
    /// </summary>
    public int InputImageSize;

    /// <summary>
    /// input image size (half)
    /// </summary>
    private float InputImageSizeHalf;

    /// <summary>
    /// column number of heatmap
    /// </summary>
    public int HeatMapCol;
    private float _inputImageSizeF;

    /// <summary>
    /// Column number of heatmap in 2D image
    /// </summary>
    private int _heatMapCol_Squared;
    
    /// <summary>
    /// Column nuber of heatmap in 3D model
    /// </summary>
    private int _heatMapCol_Cube;
    private float _imageScale;

    /// <summary>
    /// Buffer memory has 2D heat map
    /// </summary>
    private float[] _heatMap2D;

    /// <summary>
    /// Buffer memory has offset 2D
    /// </summary>
    private float[] _offset2D;
    
    /// <summary>
    /// Buffer memory has 3D heat map
    /// </summary>
    private float[] _heatMap3D;
    
    /// <summary>
    /// Buffer memory hash 3D offset
    /// </summary>
    private float[] _offset3D;
    private float unit;
    
    /// <summary>
    /// Number of joints in 2D image
    /// </summary>
    private int JointNum_Squared = JointNum * 2;
    
    /// <summary>
    /// Number of joints in 3D model
    /// </summary>
    private int JointNum_Cube = JointNum * 3;

    /// <summary>
    /// HeatMapCol * JointNum
    /// </summary>
    private int HeatMapCol_JointNum;

    /// <summary>
    /// HeatMapCol * JointNum_Squared
    /// </summary>
    private int CubeOffsetLinear;

    /// <summary>
    /// HeatMapCol * JointNum_Cube
    /// </summary>
    private int CubeOffsetSquared;

    /// <summary>
    /// For Kalman filter parameter Q
    /// </summary>
    public float KalmanParamQ;

    /// <summary>
    /// For Kalman filter parameter R
    /// </summary>
    public float KalmanParamR;

    /// <summary>
    /// Lock to update VNectModel
    /// </summary>
    private bool Lock = true;

    /// <summary>
    /// Use low pass filter flag
    /// </summary>
    public bool UseLowPassFilter;

    /// <summary>
    /// For low pass filter
    /// </summary>
    public float LowPassParam;

    //public Text Msg;
    public float WaitTimeModelLoad = 10f;
    private float Countdown = 0;
    public Texture2D InitImg;

    private void Start()
    {
        // Initialize 
        _heatMapCol_Squared = HeatMapCol * HeatMapCol;
        _heatMapCol_Cube = HeatMapCol * HeatMapCol * HeatMapCol;
        HeatMapCol_JointNum = HeatMapCol * JointNum;
        CubeOffsetLinear = HeatMapCol * JointNum_Cube;
        CubeOffsetSquared = _heatMapCol_Squared * JointNum_Cube;

        _heatMap2D = new float[JointNum * _heatMapCol_Squared];
        _offset2D = new float[JointNum * _heatMapCol_Squared * 2];
        _heatMap3D = new float[JointNum * _heatMapCol_Cube];
        _offset3D = new float[JointNum * _heatMapCol_Cube * 3];
        unit = 1f / (float)HeatMapCol;
        _inputImageSizeF = InputImageSize;
        InputImageSizeHalf = _inputImageSizeF / 2f;
        _imageScale = InputImageSize / (float)HeatMapCol;// 224f / (float)InputImageSize;

        // Disabel sleep
        Screen.sleepTimeout = SleepTimeout.NeverSleep;

        // Init model
        _model = ModelLoader.Load(NNModel, Verbose);
        _worker = WorkerFactory.CreateWorker(WorkerType, _model, Verbose);

        StartCoroutine("WaitLoad");

    }

    private void Update()
    {
        if (!Lock)
        {
            UpdateVNectModel();
        }
        /*else
        {
            if (Input.GetKey(KeyCode.B))
            {
                VNectModel = GameObject.FindWithTag("jugadorPlayer").GetComponent<VNectModel>();
                StartCoroutine("WaitLoad");
            }
        }*/
    }

    private IEnumerator WaitLoad()
    {
        _inputs[inputName_1] = MakeRGBInputTensor(InitImg);
        _inputs[inputName_2] = MakeRGBInputTensor(InitImg);
        _inputs[inputName_3] = MakeRGBInputTensor(InitImg);

        // Create input and Execute model
        yield return _worker.StartManualSchedule(_inputs);

        // Get outputs
        for (var i = 2; i < _model.outputs.Count; i++)
        {
            b_outputs[i] = _worker.PeekOutput(_model.outputs[i]);
        }

        // Get data from outputs
        _offset3D = b_outputs[2].data.Download(b_outputs[2].shape);
        _heatMap3D = b_outputs[3].data.Download(b_outputs[3].shape);

        // Release outputs
        for (var i = 2; i < b_outputs.Length; i++)
        {
            b_outputs[i].Dispose();
        }

        // Init VNect model
        _jointPoints = VNectModel.Init();

        PredictPose();

        yield return new WaitForSeconds(WaitTimeModelLoad);

        // Init VideoCapture
        videoCapture.Init(InputImageSize, InputImageSize);
        Lock = false;
        //Msg.gameObject.SetActive(false);
    }
    
    bool _executing;

    static void DisposeAndNull(IDictionary<string, Tensor> dict, string key)
    {
        if (dict.TryGetValue(key, out var t) && t != null) t.Dispose();
        dict[key] = null;
    }

    static Tensor MakeRGBInputTensor(Texture tex) => new Tensor(tex, channels: 3);
    
    
    private const string inputName_1 = "input.1";
    private const string inputName_2 = "input.4";
    private const string inputName_3 = "input.7";
    /*
    private const string inputName_1 = "0";
    private const string inputName_2 = "1";
    private const string inputName_3 = "2";
    */

    private void UpdateVNectModel()
    {
        input = MakeRGBInputTensor(videoCapture.MainTexture);
        if (_inputs[inputName_1] == null)
        {
            _inputs[inputName_1] = input;
            _inputs[inputName_2] = MakeRGBInputTensor(videoCapture.MainTexture);
            _inputs[inputName_3] = MakeRGBInputTensor(videoCapture.MainTexture);
        }
        else
        {
            // dispose the one that falls out of the 3-frame ring
            DisposeAndNull(_inputs, inputName_3);
            
            _inputs[inputName_3] = _inputs[inputName_2];
            _inputs[inputName_2] = _inputs[inputName_1];
            _inputs[inputName_1] = input;
        }

        StartCoroutine(ExecuteModelAsync());
    }

    /// <summary>
    /// Tensor has input image
    /// </summary>
    /// <returns></returns>
    Tensor input = null;
    Dictionary<string, Tensor> _inputs = new Dictionary<string, Tensor>() { { inputName_1, null }, { inputName_2, null }, { inputName_3, null }, };
    Tensor[] b_outputs = new Tensor[4];

    private IEnumerator ExecuteModelAsync()
    {
        // Create input and Execute model
        yield return _worker.StartManualSchedule(_inputs);

        // Get outputs
        for (var i = 2; i < _model.outputs.Count; i++)
        {
            b_outputs[i] = _worker.PeekOutput(_model.outputs[i]);
        }

        // Get data from outputs
        _offset3D = b_outputs[2].data.Download(b_outputs[2].shape);
        _heatMap3D = b_outputs[3].data.Download(b_outputs[3].shape);
        
        // Release outputs
        for (var i = 2; i < b_outputs.Length; i++)
        {
            b_outputs[i].Dispose();
        }

        PredictPose();
    }

    /// <summary>
    /// Predict positions of each of joints based on network
    /// </summary>
    private void PredictPose()
    {
        for (var j = 0; j < JointNum; j++)
        {
            var maxXIndex = 0;
            var maxYIndex = 0;
            var maxZIndex = 0;
            _jointPoints[j].score3D = 0.0f;
            var jj = j * HeatMapCol;
            for (var z = 0; z < HeatMapCol; z++)
            {
                var zz = jj + z;
                for (var y = 0; y < HeatMapCol; y++)
                {
                    var yy = y * _heatMapCol_Squared * JointNum + zz;
                    for (var x = 0; x < HeatMapCol; x++)
                    {
                        float v = _heatMap3D[yy + x * HeatMapCol_JointNum];
                        if (v > _jointPoints[j].score3D)
                        {
                            _jointPoints[j].score3D = v;
                            maxXIndex = x;
                            maxYIndex = y;
                            maxZIndex = z;
                        }
                    }
                }
            }
           
            _jointPoints[j].Now3D.x = (_offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + j * HeatMapCol + maxZIndex] + 0.5f + (float)maxXIndex) * _imageScale - InputImageSizeHalf;
            _jointPoints[j].Now3D.y = InputImageSizeHalf - (_offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + (j + JointNum) * HeatMapCol + maxZIndex] + 0.5f + (float)maxYIndex) * _imageScale;
            _jointPoints[j].Now3D.z = (_offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + (j + JointNum_Squared) * HeatMapCol + maxZIndex] + 0.5f + (float)(maxZIndex - 14)) * _imageScale;
        }

        // Calculate hip location
        var lc = (_jointPoints[PositionIndex.rThighBend.Int()].Now3D + _jointPoints[PositionIndex.lThighBend.Int()].Now3D) / 2f;
        _jointPoints[PositionIndex.hip.Int()].Now3D = (_jointPoints[PositionIndex.abdomenUpper.Int()].Now3D + lc) / 2f;

        // Calculate neck location
        _jointPoints[PositionIndex.neck.Int()].Now3D = (_jointPoints[PositionIndex.rShldrBend.Int()].Now3D + _jointPoints[PositionIndex.lShldrBend.Int()].Now3D) / 2f;

        // Calculate head location
        var cEar = (_jointPoints[PositionIndex.rEar.Int()].Now3D + _jointPoints[PositionIndex.lEar.Int()].Now3D) / 2f;
        var hv = cEar - _jointPoints[PositionIndex.neck.Int()].Now3D;
        var nhv = Vector3.Normalize(hv);
        var nv = _jointPoints[PositionIndex.Nose.Int()].Now3D - _jointPoints[PositionIndex.neck.Int()].Now3D;
        _jointPoints[PositionIndex.head.Int()].Now3D = _jointPoints[PositionIndex.neck.Int()].Now3D + nhv * Vector3.Dot(nhv, nv);
        

        // Calculate spine location
        //jointPoints[PositionIndex.spine.Int()].Now3D = jointPoints[PositionIndex.abdomenUpper.Int()].Now3D;

        // Kalman filter
        foreach (var jp in _jointPoints)
        {
            KalmanUpdate(jp);
        }

        // Low pass filter
        if (UseLowPassFilter)
        {
            foreach (var jp in _jointPoints)
            {
                jp.PrevPos3D[0] = jp.Pos3D;
                for (var i = 1; i < jp.PrevPos3D.Length; i++)
                {
                    jp.PrevPos3D[i] = jp.PrevPos3D[i] * LowPassParam + jp.PrevPos3D[i - 1] * (1f - LowPassParam);
                }
                jp.Pos3D = jp.PrevPos3D[jp.PrevPos3D.Length - 1];
            }
        }
    }

    /// <summary>
    /// Kalman filter
    /// </summary>
    /// <param name="measurement">joint points</param>
    void KalmanUpdate(VNectModel.JointPoint measurement)
    {
        measurementUpdate(measurement);
        measurement.Pos3D.x = measurement.X.x + (measurement.Now3D.x - measurement.X.x) * measurement.K.x;
        measurement.Pos3D.y = measurement.X.y + (measurement.Now3D.y - measurement.X.y) * measurement.K.y;
        measurement.Pos3D.z = measurement.X.z + (measurement.Now3D.z - measurement.X.z) * measurement.K.z;
        measurement.X = measurement.Pos3D;
    }

	void measurementUpdate(VNectModel.JointPoint measurement)
    {
        measurement.K.x = (measurement.P.x + KalmanParamQ) / (measurement.P.x + KalmanParamQ + KalmanParamR);
        measurement.K.y = (measurement.P.y + KalmanParamQ) / (measurement.P.y + KalmanParamQ + KalmanParamR);
        measurement.K.z = (measurement.P.z + KalmanParamQ) / (measurement.P.z + KalmanParamQ + KalmanParamR);
        measurement.P.x = KalmanParamR * (measurement.P.x + KalmanParamQ) / (KalmanParamR + measurement.P.x + KalmanParamQ);
        measurement.P.y = KalmanParamR * (measurement.P.y + KalmanParamQ) / (KalmanParamR + measurement.P.y + KalmanParamQ);
        measurement.P.z = KalmanParamR * (measurement.P.z + KalmanParamQ) / (KalmanParamR + measurement.P.z + KalmanParamQ);
    }
}
