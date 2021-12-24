using UnityEngine;
using System.Collections.Generic;
using static Matr.Matrix;
class CriticNet
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
        
        public CriticNet(int inp, int h, int outp, float lr)
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
        }
        public float[,] Forward(float[,] s)
        {
            float[,] z1 = Add(Dot(s, w1), b1);
            float[,] h1 = Relu(z1);
            return Add(Dot(h1, w2), b2);
        }

    public float Backward(float[,] s, float[] target_v, float max_grad_norm)
    {
        float[,] z1 = Add(Dot(s, w1), b1);
        float[,] h1 = Relu(z1);
        float[,] v = Add(Dot(h1, w2), b2);

        int batch_size = s.GetLength(0);

        float[,] d = new float[v.GetLength(0), v.GetLength(1)];
        for (int i = 0; i < batch_size; i++)
        {
            d[i, 0] = v[i, 0] - target_v[i];
            if (d[i, 0] < -1)
                d[i, 0] = -1;
            else if (d[i, 0] > 1)
                d[i, 0] = 1;
        }
        float[,] out1 = Dot(d, Transpose(w2));
        out1 = DerRelu(out1, z1);

        float[,] dw2 = Dot(Transpose(h1), d);
        float[,] dw1 = Dot(Transpose(s), out1);
        float[,] db2 = Sum(d);
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

        return Sum_total_norm(d, new float[1, d.GetLength(1)]) / s.GetLength(0);

    }

    public Dictionary<string, float[,]> GetStateDictionary() {
        Dictionary<string, float[,] > dict = new Dictionary<string,float[,]>();

        dict.Add("w1", w1);
        dict.Add("w2", w2);
        dict.Add("b1", b1);
        dict.Add("b2", b2);

        dict.Add("m1", m1);
        dict.Add("v1", v1);
        dict.Add("m2", m2);
        dict.Add("v2", v2);

        dict.Add("mb1", mb1);
        dict.Add("vb1", vb1);
        dict.Add("mb2", mb2);
        dict.Add("vb2", vb2);
        return dict;

    }

    public void LoadStateDictionary( Dictionary<string, float[,] > dict ){

        float[,] value = new float[0,0];
        if (dict.TryGetValue("w1", out value) ){
            w1 = value;
        }
        if (dict.TryGetValue("w2", out value) ){
            w2 = value;
        }
        if (dict.TryGetValue("b1", out value) ){
            b1 = value;
        }
        if (dict.TryGetValue("b2", out value) ){
           b2 = value;
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

    }
}



