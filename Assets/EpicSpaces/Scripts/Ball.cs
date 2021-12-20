using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using System.Collections;

public class Ball : MonoBehaviour
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

    public GameObject ball;
    public Text infoText;
    private Transform ballTransform;
    private Rigidbody ballRb;

    public int framesPerDecision = 30;
    private int nextDecisionFrame;


    void Start() {

        ballTransform = ball.transform;

        ballRb = ball.GetComponent<Rigidbody>();

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
        
        //ballRb.AddForce(new Vector3(1000,0,0));
        resetSimulation();

        nextDecisionFrame = Time.frameCount + framesPerDecision;
    }
    void Update(){

        if( Time.frameCount > nextDecisionFrame){
            nextDecisionFrame = Time.frameCount + framesPerDecision;
            learn();
        }
    }

    private void resetSimulation(){

        ballTransform.position = new Vector3(0,1f,0);
            //ballRb.velocity = new Vector3(0,0,0);
/*
		int a = Random.Range(0,2);
			if(a == 0)
			ballRb.AddForce(new Vector3(1000,0,0));
		    else if(a == 1)
			ballRb.AddForce(new Vector3(1000,0,1000));
*/
        ballRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 1f, Random.Range(-1.5f, 1.5f)) + gameObject.transform.position;    

    }


    private void learn(){
        float losslabel = 0;
        float[,] s = new float[1, input_size];
        reward = 0.1f;
        //Transform ball = GameObject.Find("Ball").transform;
        s[0, 0] = transform.rotation.z;
        s[0, 1] = transform.rotation.x;
        s[0, 2] = transform.position.x - ballTransform.position.x;
        s[0, 3] = transform.position.y - ballTransform.position.y;
        s[0, 4] = transform.position.z - ballTransform.position.z;
        //var rb = ball.GetComponent<Rigidbody>();
        s[0, 5] = ballRb.velocity.x;
        s[0, 6] = ballRb.velocity.y;
        s[0, 7] = ballRb.velocity.z;

        float[,] alp;
        float[,] act = ppo.Predict(s, out alp, b);

		float x = Mathf.Clamp(act[0,0],-b,b);
		float z = Mathf.Clamp(act[0,1],-b,b);
        transform.rotation = Quaternion.AngleAxis(15*x, Vector3.right) * Quaternion.AngleAxis(15*z, Vector3.forward);
        

        Vector3 positionDelta = ballTransform.position - transform.position;
        //check out of bounds
		if (positionDelta.y < -3 || Mathf.Abs(positionDelta.x) > 6 || Mathf.Abs(positionDelta.z) > 6){
            reward = -1f;
            resetSimulation();
            episodeCount++;
        }
        
       // else 
		//{
			//	reward=0.1f;
		 
        //}

        //not sure why using x & y and no z velocity... ?
        /*
		Vector3 vec= new Vector3(ballRb.velocity.x,ballRb.velocity.y,0);
		reward = 0.1f - vec.magnitude;
        */
        

		if (istraining == true) {
            float loss = ppo.Train(s, act, reward, alp);
            if (loss != 0){
                losslabel = loss * loss;
            }
        }
        infoText.text = "reward : "+reward + "\nloss : " + losslabel+ "\nepisode count : "+episodeCount;

    }
}
