using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using System.Collections;


//Note I've added a few public variables - so they are visible in the inspector - not so they are script accessible
public class BallBalanceAgent : MonoBehaviour
{
    AgentPPO ppo;
    int batch_Capacity = 200;
    int batch_size = 32;
    int ppo_epoch = 10;
    int input_size = 8;
    int hidden_size = 80;
    int output_size = 2;
    float learning_rate_a = 1e-3f;
    float learning_rate_v = 3e-3f;
    float gamma = 0.99f;
    float clip_param = 0.2f;
    float max_grad_norm = 0.5f;
    float b = 2;// action bounds
    public bool istraining = true;
    int episodeCount;
    
    float episodeReward;
    float reward;
    public float rotationSpeed = 6f;

    public GameObject ball;
    public Text infoText;
    private Transform ballTransform;
    private Rigidbody ballRb;

    public float timePerDecision = 0.3333f;
    private float nextDecisionTime;

    private Vector3 ballStartPosition;

    private Quaternion wantedRotation;
    private Quaternion startRotation;
    private float lerpAmount = 1.01f;

    public bool isTouching = false;
    public Vector3 positionDelta;

    public Vector2 actionOut = Vector2.zero;

    public float debug;


    //cache all references at start so we're not doing expensive finds during play
    void Start() {

        ballTransform = ball.transform;
        ballRb = ball.GetComponent<Rigidbody>();
        ballStartPosition = ballTransform.position;

        nextDecisionTime = Time.fixedTime + timePerDecision;

        ppo = new AgentPPO(batch_Capacity,
            batch_size,
            ppo_epoch,
            input_size,
            hidden_size,
            output_size,
            learning_rate_a,
            learning_rate_v,
            gamma,
            b,
            clip_param,
            max_grad_norm);
        

        

        resetSimulation();

        
    }
    void Update(){

        if( lerpAmount <= 1f){

            transform.rotation = Quaternion.Lerp(startRotation, wantedRotation, lerpAmount);
            lerpAmount += Time.deltaTime * rotationSpeed;
        }

        positionDelta = ballTransform.position - transform.position;  

        //check out of bounds
		if (positionDelta.y < -1.5f || Mathf.Abs(positionDelta.x) > 3.8f || Mathf.Abs(positionDelta.z) > 3.8f){
            reward = -1f;
            resetSimulation();
            episodeCount++;
        }
    }

    void FixedUpdate(){

        if( Time.fixedTime > nextDecisionTime && isTouching){
            nextDecisionTime = Time.fixedTime + timePerDecision;
            learn();
            
        }
    }

    private void resetSimulation(){

        ballRb.velocity = new Vector3(0f, 0f, 0f);
        ballTransform.position = new Vector3(Random.Range(-1.5f, 1.5f), 0f, Random.Range(-1.5f, 1.5f)) + ballStartPosition;    

    }
    void OnCollisionStay(Collision collisionInfo) {
        if( collisionInfo.transform.tag == "Player" && !isTouching){
            isTouching = true;
        }
    }

    void OnCollisionExit(Collision collisionInfo) {
        if( collisionInfo.transform.tag == "Player" && isTouching ){
            isTouching = false;
        }
    }

    void OnCollisionEnter(Collision collisionInfo) {
        if( collisionInfo.transform.tag == "Player" ){
            isTouching = true;
        }
    }

    private void learn(){
        float losslabel = 0;
        float[,] s = new float[1, input_size];


        s[0, 0] = Mathf.Clamp(transform.rotation.x, -1f, 1f);
        s[0, 1] = Mathf.Clamp(transform.rotation.z, -1f, -1f);

        Vector3 normalisedPositionDelta = positionDelta * 0.3333f; // normalise to bounds is 3 so multiply by 1/3 = 0.3333
        s[0, 2] = Mathf.Clamp(positionDelta.x, -1f, 1f);
        s[0, 3] = Mathf.Clamp(positionDelta.y, -1f, 1f);
        s[0, 4] = Mathf.Clamp(positionDelta.z, -1f, 1f);
        
        Vector3 normalisedVelo = ballRb.velocity * 0.2f; //assume a max velo of 5

        s[0, 5] = Mathf.Clamp(normalisedVelo.x, -1f, 1f);
        s[0, 6] = Mathf.Clamp(normalisedVelo.y, -1f, 1f);
        s[0, 7] = Mathf.Clamp(normalisedVelo.z, -1f, 1f);

        float[,] alp;
        float[,] act = ppo.Predict(s, out alp, b);

        actionOut[0] = act[0,0];
        actionOut[1] = act[0,1];

		float x = Mathf.Clamp(act[0,0],-b,b);
		float z = Mathf.Clamp(act[0,1],-b,b);

        //we will set the wanted action and pick this up on the update function to act as a simple tween / lerp
        float degreeRotationRange = 8f;
        Vector3 wanterEuler = new Vector3(degreeRotationRange * x, 0, degreeRotationRange * z);
        wantedRotation.eulerAngles = wanterEuler;
        lerpAmount = 0f;
        startRotation = transform.rotation;
        
        //distance from center reward
        reward = 0.05f + Mathf.Clamp(0.1f * (3f - Mathf.Sqrt(positionDelta.x*positionDelta.x+positionDelta.z*positionDelta.z)), 0f , 0.5f);
        //reward = 0.1f;

		if (istraining == true) {
            float loss = ppo.Train(s, act, reward, alp);
            if (loss != 0){
                losslabel = loss * loss;
            }
        }
        infoText.text = "reward : "+reward + "\nloss : " + losslabel+ "\nepisode count : " + episodeCount + "\nTime : "+ Time.time;

    }
}
