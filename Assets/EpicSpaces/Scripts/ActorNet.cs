using UnityEngine;
using System.Collections.Generic;
using static Matr.Matrix;
    class ActorNet
    {
        float[,] w1;
        float[,] w2;
        float[,] w3;
        float[,] b1;
        float[,] b2;
        float[,] b3;

        float[,] m1;
        float[,] v1;
        float[,] m2;
        float[,] v2;
        float[,] m3;
        float[,] v3;
    
        float[,] mb1;
        float[,] vb1;
        float[,] mb2;
        float[,] vb2;
        float[,] mb3;
        float[,] vb3;


        int t = 0;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float lr = 1e-3f;
        public ActorNet(int inp, int h, int outp, float lr)
        {
            w1 = Init(inp, h);
            w2 = Init(h, outp);
            w3 = Init(h, outp);
            b1 = new float[1, h];
            b2 = new float[1, outp];
            b3 = new float[1, outp];

            m1 = new float[inp, h];
            v1 = new float[inp, h];
            m2 = new float[h, outp];
            v2 = new float[h, outp];
            m3 = new float[h, outp];
            v3 = new float[h, outp];

            mb1 = new float[1, h];
            vb1 = new float[1, h];
            mb2 = new float[1, outp];
            vb2 = new float[1, outp];
            mb3 = new float[1, outp];
            vb3 = new float[1, outp];

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
        }
        public float[,] Forward(float[,] s, float b, out float[,] alp)
        {
        float[,] z1 = Add(Dot(s, w1), b1);
        float[,] h1 = Relu(z1);
        float[,] z2 = Add(Dot(h1, w2), b2);

        int z2_length_0 = z2.GetLength(0);
        int z2_length_1 = z2.GetLength(1);

        float[,] mu = new float[z2_length_0,z2_length_1];
        for (int i = 0; i < z2_length_0; i++)
        {
            for (int j = 0; j < z2_length_1; j++)
            {
                mu[i,j] = b * (Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1);
            }
        }
        float[,] z3 = Add(Dot(h1, w3), b3);

        int z3_length_0 = z3.GetLength(0);
        int z3_length_1 = z3.GetLength(1);

        float[,] sigma = new float[z3_length_0, z3_length_1];
        for (int i = 0; i < z3_length_0; i++)
        {
            for (int j = 0; j < z3_length_1; j++)
            {
                sigma[i,j] = Mathf.Log(1 + Mathf.Exp(z3[i,j]));
            }
        }
        int s_length_0 = s.GetLength(0);
        int mu_length_1 = mu.GetLength(1);


        float[,]a = new float[s_length_0,mu_length_1];
        alp = new float[s_length_0, mu_length_1];
        for (int i = 0; i < s_length_0; i++)
        {
            for (int j = 0; j < mu_length_1; j++)
            {
                a[0, j] = mu[i, j] + sigma[i, j] * Mathf.Sqrt(-2.0f * Mathf.Log(Random.Range(0.0f,1.0f))) * Mathf.Sin(2.0f * Mathf.PI * Random.Range(0.0f, 1.0f));
                alp[0, j] = -((a[i, j] - mu[i, j]) * (a[i, j] - mu[i, j])) / (2 * sigma[i, j] * sigma[i, j]) - Mathf.Log(sigma[i, j]) - Mathf.Log(Mathf.Sqrt(2 * Mathf.PI));
            } 
        }

        return a;
    }

    public void Backward(float[,] s, float[,] a, float[,] olp, float[] adv, float b, float clip_param, float max_grad_norm)
    {
        Debug.Log("Train Backward");
        float[,] z1 = Add(Dot(s, w1), b1);
        float[,] h1 = Relu(z1);
        float[,] z2 = Add(Dot(h1, w2), b2);

        int z2_length_0 = z2.GetLength(0);
        int z2_length_1 = z2.GetLength(1);

        float[,] mu = new float[z2_length_0, z2_length_1];

        
        for (int i = 0; i < z2_length_0; i++)
        {
            for (int j = 0; j < z2_length_1; j++)
            {
                mu[i, j] = b * (Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1);
            }
        }
        float[,] z3 = Add(Dot(h1, w3), b3);
        int z3_length_0 = z3.GetLength(0);
        int z3_length_1 = z3.GetLength(1);

        float[,] sigma = new float[z3_length_0, z3_length_1];
        for (int i = 0; i < z3_length_0; i++)
        {
            for (int j = 0; j < z3_length_1; j++)
            {
                sigma[i, j] = Mathf.Log(1 + Mathf.Exp(z3[i, j]));
            }
        }
        int batch_size = s.GetLength(0);

        int a_length_1 = a.GetLength(1);

        float[,] alp = new float[batch_size, a_length_1];
        float[,] ratio = new float[batch_size, a_length_1];
        float[,] surr1 = new float[batch_size, a_length_1];
        float[,] surr2 = new float[batch_size, a_length_1];
        float[,] mu_derv = new float[batch_size, a_length_1];
        float[,] sigma_derv = new float[batch_size, a_length_1];
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < a_length_1; j++)
            {
                alp[i, j] = -((a[i, j] - mu[i, j]) * (a[i, j] - mu[i, j])) / (2 * sigma[i, j] * sigma[i, j]) - Mathf.Log(sigma[i, j]) - Mathf.Log(Mathf.Sqrt(2 * Mathf.PI));
                ratio[i, j] = Mathf.Exp(alp[i, j] - olp[i, j]);

                surr1[i, j] = ratio[i, j] * adv[i];
                surr2[i, j] = Mathf.Clamp(ratio[i, j], 1 - clip_param, 1 + clip_param) * adv[i];

                if (surr2[i, j] < surr1[i, j] && (ratio[i, j] < 1 - clip_param || ratio[i, j] > 1 + clip_param))
                {
                    mu_derv[i, j] = 0;
                    sigma_derv[i, j] = 0;
                }
                else
                {
                    mu_derv[i, j] = -(b * adv[i] * Mathf.Exp(-(a[i, j] - b * (Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1)) * (a[i, j] - b * (Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1)) / (2 * sigma[i, j] * sigma[i, j]) - olp[i, j]) * (1 - ((Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1)) * ((Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1))) * (a[i, j] - b * (Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1))) / (Mathf.Sqrt(2 * Mathf.PI) * sigma[i, j] * sigma[i, j] * sigma[i, j]);
                    sigma_derv[i, j] = (adv[i] * Mathf.Exp(-(a[i, j] - mu[i, j]) * (a[i, j] - mu[i, j]) / (2 * Mathf.Log(Mathf.Exp(z3[i, j]) + 1) * Mathf.Log(Mathf.Exp(z3[i, j]) + 1)) + z3[i, j] - olp[i, j]) * (Mathf.Log(Mathf.Exp(z3[i, j]) + 1) * Mathf.Log(Mathf.Exp(z3[i, j]) + 1) - mu[i, j] * mu[i, j] + 2 * a[i, j] * mu[i, j] - a[i, j] * a[i, j])) / (Mathf.Sqrt(2 * Mathf.PI) * (Mathf.Exp(z3[i, j]) + 1) * (Mathf.Log(Mathf.Exp(z3[i, j]) + 1) * Mathf.Log(Mathf.Exp(z3[i, j]) + 1) * Mathf.Log(Mathf.Exp(z3[i, j]) + 1) * Mathf.Log(Mathf.Exp(z3[i, j]) + 1)));

                    sigma_derv[i, j] /= a_length_1;
                    mu_derv[i, j] /= a_length_1;
                }

            }
        }
        float[,] out1 = Dot(sigma_derv, Transpose(w3));
        out1 = Add(out1, Dot(mu_derv, Transpose(w2)));
        out1 = DerRelu(out1, z1);

        float[,] dw3 = Dot(Transpose(h1), sigma_derv);
        float[,] dw2 = Dot(Transpose(h1), mu_derv);
        float[,] dw1 = Dot(Transpose(s), out1);
        float[,] db3 = Sum(sigma_derv);
        float[,] db2 = Sum(mu_derv);
        float[,] db1 = Sum(out1);

        float total_norm = 0;

        total_norm += Sum_total_norm(w3, b3);
        total_norm += Sum_total_norm(w2, b2);
        total_norm += Sum_total_norm(w1, b1);

        total_norm = Mathf.Sqrt(total_norm);
        float clip_coef = (float)(max_grad_norm / (total_norm + 1e-6));
        if (clip_coef < 1)
        {
            dw3 = Mult_clip_coef(dw3, db3, out db3, clip_coef);
            dw2 = Mult_clip_coef(dw2, db2, out db2, clip_coef);
            dw1 = Mult_clip_coef(dw1, db1, out db1, clip_coef);

        }
        t++;
        Adam(w1, dw1, m1, v1, batch_size, beta1, beta2, eps, lr, t);
        Adam(w2, dw2, m2, v2, batch_size, beta1, beta2, eps, lr, t);
        Adam(w3, dw3, m3, v3, batch_size, beta1, beta2, eps, lr, t);
        Adam(b1, db1, mb1, vb1, batch_size, beta1, beta2, eps, lr, t);
        Adam(b2, db2, mb2, vb2, batch_size, beta1, beta2, eps, lr, t);
        Adam(b3, db3, mb3, vb3, batch_size, beta1, beta2, eps, lr, t);
    }  

    public Dictionary<string, float[,]> GetStateDictionary() {
        Dictionary<string, float[,] > dict = new Dictionary<string,float[,]>();

        dict.Add("b1", b1);
        dict.Add("b2", b2);
        dict.Add("b3", b3);

        dict.Add("m1", m1);
        dict.Add("v1", v1);
        dict.Add("m2", m2);
        dict.Add("v2", v2);
        dict.Add("m3", m3);
        dict.Add("v3", v3);

        dict.Add("mb1", mb1);
        dict.Add("vb1", vb1);
        dict.Add("mb2", mb2);
        dict.Add("vb2", vb2);
        dict.Add("mb3", mb3);
        dict.Add("vb3", vb3);
        return dict;

    }

    public void LoadStateDictionary( Dictionary<string, float[,] > dict ){

        float[,] value = new float[0,0];

        if (dict.TryGetValue("b1", out value) ){
            b1 = value;
        }
        if (dict.TryGetValue("b2", out value) ){
           b2 = value;
        }
        if (dict.TryGetValue("b3", out value) ){
           b3 = value;
        }
 
        if (dict.TryGetValue("m1", out value) ){
           m1 = value;
        }
        if (dict.TryGetValue("v1", out value) ){
           v1 = value;
        }
        if (dict.TryGetValue("m2", out value) ){
           m2 = value;
        }
        if (dict.TryGetValue("v2", out value) ){
           v2 = value;
        }
        if (dict.TryGetValue("m3", out value) ){
           m3 = value;
        }
        if (dict.TryGetValue("v3", out value) ){
           v3 = value;
        }

        if (dict.TryGetValue("mb1", out value) ){
           mb1 = value;
        }
        if (dict.TryGetValue("vb1", out value) ){
           vb1 = value;
        }
        if (dict.TryGetValue("mb2", out value) ){
           mb2 = value;
        }
        if (dict.TryGetValue("vb2", out value) ){
           vb2 = value;
        }
        if (dict.TryGetValue("mb3", out value) ){
           mb3 = value;
        }
        if (dict.TryGetValue("vb3", out value) ){
           vb3 = value;
        }


    }


    
}



