using UnityEngine;
using UnityEngine.Profiling;
namespace MPH
{
    public static class MatrixOperations
    {
    // Adam optimiser
    // Adam realizes the benefits of both Adaptive Gradient Algorithm and Root Mean Square Propagation.
    // Adam method doesn't return anything and no variables are passed by ref - does it do anything?
    // I think x should be passed by ref

        public static void Adam( ref float[,] x,   float[,] dx,   float[,] m,   float[,] v,   int batch_size,   float beta1,   float beta2,   float eps,   float lr,  int t)
        {
            Profiler.BeginSample("Matrix:Adam"); 
            int x_length_0 = x.GetLength(0);
            int x_length_1 = x.GetLength(1);
            float d;
            float mb;
            float vb;

            // optimisation - possibly precalc all the divisors into multipliers

            for (int i = 0; i < x_length_0; i++){
                for (int j = 0; j < x_length_1; j++){

                    d = dx[i, j] / batch_size;
                    m[i, j] = m[i, j] * beta1 + (1 - beta1) * d;
                    v[i, j] = v[i, j] * beta2 + (1 - beta2) * d * d;
                    mb = m[i, j] / (1 - Mathf.Pow(beta1, t));
                    vb = v[i, j] / (1 - Mathf.Pow(beta2, t));
                    x[i, j] = x[i, j] - lr * (mb / (Mathf.Sqrt(vb) + eps));
                }
            }
            Profiler.EndSample();

        }
        public static float[,] Dot(  float[,] a,   float[,] b){
            Profiler.BeginSample("Matrix:Dot"); 
            int a_length_0 = a.GetLength(0);
            int b_length_0 = b.GetLength(0);
            int b_length_1 = b.GetLength(1);

            float[,] c = new float[a_length_0, b_length_1];

            for (int i = 0; i < a_length_0; i++)
            {
                for (int k = 0; k < b_length_1; k++)
                {
                    for (int j = 0; j < b_length_0; j++)
                    {
                        c[i, k] += a[i, j] * b[j, k];
                    }
                }
            }
            Profiler.EndSample();
            return c;
        }

        public static float[,] DotB(  float[,] a,   float[,] b){
            Profiler.BeginSample("Matrix:DotB"); 
            int a_length_0 = a.GetLength(0);
            int b_length_0 = b.GetLength(0);
            int b_length_1 = b.GetLength(1);
            float[,] c = new float[a_length_0, b_length_1];

            for (int i = 0; i < a_length_0; i++)
            {
                for (int k = 0; k < b_length_1; k++)
                {
                    for (int j = 0; j < b_length_0; j++)
                    {
                        c[i, k] += a[i, j] * b[j, k];
                    }
                }
            }
            Profiler.EndSample();
            return c;
        }

        public static float[,] Transpose(  float[,] a){
            Profiler.BeginSample("Matrix:Transpose"); 
            int a_length_0 = a.GetLength(0);
            int a_length_1 = a.GetLength(1);
            float[,] b = new float[a_length_1, a_length_0];

            for (int i = 0; i < a_length_0; i++)
            {
                for (int j = 0; j < a_length_1; j++)
                {
                    b[j, i] = a[i, j];
                }
            }
            Profiler.EndSample();
            return b;
        }
        public static float[,] Add(  float[,] a,   float[,] b)
        {
            Profiler.BeginSample("Matrix:Add"); 
            int a_length_0 = a.GetLength(0);
            int a_length_1 = a.GetLength(1);

            float[,] c = new float[a_length_0, a_length_1];
            for (int i = 0; i < a_length_0; i++)
            {
                for (int j = 0; j < a_length_1; j++)
                {
                    c[i, j] = a[i, j] + b[0, j];
                }
            }
            Profiler.EndSample();
            return c;
        }
        public static float[,] Relu(  float[,] a){
            Profiler.BeginSample("Matrix:Relu");
            int a_length_0 = a.GetLength(0);
            int a_length_1 = a.GetLength(1);
            
            for (int i = 0; i < a_length_0; i++)
            {
                for (int j = 0; j < a_length_1; j++)
                {
                    if (a[i, j] < 0)
                        a[i, j] = 0;
                }
            }
            Profiler.EndSample();
            return a;
        }
        public static float[,] DerRelu(  float[,] a,   float[,] z){
            Profiler.BeginSample("Matrix:DerRelu");
            int a_length_0 = a.GetLength(0);
            int a_length_1 = a.GetLength(1);
            
            for (int i = 0; i < a_length_0; i++)
            {
                for (int j = 0; j < a_length_1; j++)
                {
                    if (z[i, j] <= 0)
                        a[i, j] = 0;
                }
            }
            Profiler.EndSample();
            return a;
        }
        public static float[,] Sum(  float[,] a){
            Profiler.BeginSample("Matrix:Sum");
            int a_length_0 = a.GetLength(0);
            int a_length_1 = a.GetLength(1);
            float[,] b = new float[1, a_length_1];
            for (int j = 0; j < a_length_1; j++)
            {
                for (int i = 0; i < a_length_0; i++)
                {
                    b[0, j] += a[i, j];
                }
            }
            Profiler.EndSample();
            return b;
        }
        public static float Sum_total_norm(float[,] a,   float[,] b){
            Profiler.BeginSample("Matrix:Sum_total_norm");
            int a_length_0 = a.GetLength(0);
            int a_length_1 = a.GetLength(1);

            float total_norm = 0;
            for (int j = 0; j < a_length_1; j++)
            {
                total_norm += b[0, j] * b[0, j];
                for (int i = 0; i < a_length_0; i++)
                {
                    total_norm += a[i, j] * a[i, j];
                }
            }
            Profiler.EndSample();
            return total_norm;
        }
        public static float[,] Mult_clip_coef(float[,] dw, float[,] db, out float[,] rdb, float clip_coef){
            Profiler.BeginSample("Matrix:Mult_clip_coef");
            int dw_length_0 = dw.GetLength(0);
            int dw_length_1 = dw.GetLength(1);
            
            for (int j = 0; j < dw_length_1; j++)
            {
                db[0, j] *= clip_coef;
                for (int i = 0; i < dw_length_0; i++)
                {
                    dw[i, j] *= clip_coef;
                }
            }
            rdb = db;
            Profiler.EndSample();
            return dw;
        }
    }
}