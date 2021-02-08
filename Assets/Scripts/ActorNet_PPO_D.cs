﻿using UnityEngine;
using static Matr.Matrix;
    class ActorNet_PPO_D
    {
        float[,] w1;
        float[,] w2;
        float[,] b1;
        float[,] b2;

        float[,] m1;
        float[,] v1;
        float[,] m2;
        float[,] v2;
    
        float[,] mb1;
        float[,] vb1;
        float[,] mb2;
        float[,] vb2;


        int t = 0;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float lr = 1e-3f;
        public ActorNet_PPO_D(int inp, int h, int outp, float lr)
        {
            w1 = Init(inp, h);
            w2 = Init(h, outp);
            b1 = new float[1, h];
            b2 = new float[1, outp];

            m1 = new float[inp, h];
            v1 = new float[inp, h];
            m2 = new float[h, outp];
            v2 = new float[h, outp];

            mb1 = new float[1, h];
            vb1 = new float[1, h];
            mb2 = new float[1, outp];
            vb2 = new float[1, outp];

        this.lr = lr;
        }
        public float[,] Init(int inp, int outp)
        {
            float[,] u = new float[inp, outp];
            float un = Mathf.Sqrt(1.0f / (inp * outp));

            for (int i = 0; i < inp; i++)
            {
                for (int j = 0; j < outp; j++)
                {
                    u[i, j] = Random.Range(-un,un);
                }
            }
            return u;
        }float epsilon = 0.9f;
        public int Forward(float[,] s, out float ap)
        {
        float[,] z1 = Add(Dot(s, w1), b1);
        float[,] h1 = Relu(z1);
        float[,] z2 = Add(Dot(h1, w2), b2);
        float[,] softmax = new float[z2.GetLength(0),z2.GetLength(1)];
        ap = 0;
        for (int i = 0; i < z2.GetLength(0); i++)
        {
            float sumexp = 0;
            for (int j = 0; j < z2.GetLength(1); j++)
            {
                sumexp+= Mathf.Exp(z2[i, j]);
            }
            for (int j = 0; j < z2.GetLength(1); j++)
            {
                softmax[i, j] = Mathf.Exp(z2[i, j])/sumexp;
            }
        }
        /*double cumulative = 0.0;
        int selectedElement = 0;
        for (int i = 0; i < softmax.GetLength(1); i++)
        {
            cumulative += softmax[0,i];
            if (Random.Range(0f, 1f) < cumulative)
            {
                selectedElement = i;
                break;
            }
        }*/
        int act_idx = 0;
        if (Random.Range(0f, 1f) < epsilon)
        {
            act_idx = Random.Range(0, 4);
        }
        else
        {
            float max = softmax[0, 0];
            for (int i = 0; i < softmax.GetLength(1); i++)
            {
                if (max < softmax[0, i])
                {
                    max = softmax[0, i];
                    act_idx=i;
                    ap = max;
                }
            }
        }
        epsilon *= 0.999f;
        return act_idx;
        //return selectedElement;
    }

    public void Backward(float[,] s, int[] a_idx, float[] op, float[] adv, float clip_param, float max_grad_norm)
    {
        float[,] z1 = Add(Dot(s, w1), b1);
        float[,] h1 = Relu(z1);
        float[,] z2 = Add(Dot(h1, w2), b2);
        float[,] softmax = new float[z2.GetLength(0), z2.GetLength(1)];
        for (int i = 0; i < z2.GetLength(0); i++)
        {
            float sumexp = 0;
            for (int j = 0; j < z2.GetLength(1); j++)
            {
                sumexp += Mathf.Exp(z2[i, j]);
            }
            for (int j = 0; j < z2.GetLength(1); j++)
            {
                softmax[i, j] = Mathf.Exp(z2[i, j]) / sumexp;
            }
        }

        int batch_size = s.GetLength(0);

        float[] ratio = new float[batch_size];
        float[] surr1 = new float[batch_size];
        float[] surr2 = new float[batch_size];
        float[,] softmax_derv = new float[batch_size,softmax.GetLength(1)];
        for (int i = 0; i < batch_size; i++)
        {
                ratio[i] = softmax[i, a_idx[i]] / op[i];

                surr1[i] = ratio[i] * adv[i];
                surr2[i] = Mathf.Clamp(ratio[i], 1 - clip_param, 1 + clip_param) * adv[i];

                if (surr2[i] < surr1[i] && (ratio[i] < 1 - clip_param || ratio[i] > 1 + clip_param))
                {
                    softmax_derv[i,a_idx[0]] = 0;
                }
                else
                {
                softmax_derv[i, a_idx[i]] = (1 - softmax[i, a_idx[i]]) * softmax[i, a_idx[i]];
                softmax_derv[i, a_idx[i]] /= a_idx.GetLength(0);
                }
            }
        float[,] out1 = Dot(softmax_derv, Transpose(w2));
        out1 = DerRelu(out1, z1);

        float[,] dw2 = Dot(Transpose(h1), softmax_derv);
        float[,] dw1 = Dot(Transpose(s), out1);
        float[,] db2 = Sum(softmax_derv);
        float[,] db1 = Sum(out1);

        float total_norm = 0;

        total_norm += Sum_total_norm(w2, b2);
        total_norm += Sum_total_norm(w1, b1);

        total_norm = Mathf.Sqrt(total_norm);
        float clip_coef = (float)(max_grad_norm / (total_norm + 1e-6));
        if (clip_coef < 1)
        {
            dw2 = Mult_clip_coef(dw2, db2, out db2, clip_coef);
            dw1 = Mult_clip_coef(dw1, db1, out db1, clip_coef);

        }
        t++;
        Adam(w1, dw1, m1, v1, batch_size, beta1, beta2, eps, lr, t);
        Adam(w2, dw2, m2, v2, batch_size, beta1, beta2, eps, lr, t);
        Adam(b1, db1, mb1, vb1, batch_size, beta1, beta2, eps, lr, t);
        Adam(b2, db2, mb2, vb2, batch_size, beta1, beta2, eps, lr, t);
    }   
}



