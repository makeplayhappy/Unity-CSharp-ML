using UnityEngine;
namespace Matr
{
    public static class Matrix
    {
        public static void Adam(  float[,] x,   float[,] dx,   float[,] m,   float[,] v,   int batch_size,   float beta1,   float beta2,   float eps,   float lr,  int t)
        {
            
            int x_length_0 = x.GetLength(0);
            int x_length_1 = x.GetLength(1);
            for (int i = 0; i < x_length_0; i++)
            {
                for (int j = 0; j < x_length_1; j++)
                {
                    float d = dx[i, j] / batch_size;
                    m[i, j] = m[i, j] * beta1 + (1 - beta1) * d;
                    v[i, j] = v[i, j] * beta2 + (1 - beta2) * d * d;
                    float mb = m[i, j] / (1 - Mathf.Pow(beta1, t));
                    float vb = v[i, j] / (1 - Mathf.Pow(beta2, t));

                    x[i, j] = x[i, j] - lr * (mb / (Mathf.Sqrt(vb) + eps));
                }
            }

        }
        public static float[,] Dot(  float[,] a,   float[,] b)
        {
            
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
            return c;
        }

        public static float[,] DotB(  float[,] a,   float[,] b)
        {
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
            return c;
        }

        public static float[,] Transpose(  float[,] a)
        {
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
            return b;
        }
        public static float[,] Add(  float[,] a,   float[,] b)
        {
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
            return c;
        }
        public static float[,] Relu(  float[,] a)
        {
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
            return a;
        }
        public static float[,] DerRelu(  float[,] a,   float[,] z)
        {
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
            return a;
        }
        public static float[,] Sum(  float[,] a)
        {
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
            return b;
        }
        public static float Sum_total_norm(  float[,] a,   float[,] b)
        {
            
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
            return total_norm;
        }
        public static float[,] Mult_clip_coef(  float[,] dw,   float[,] db, out float[,] rdb,   float clip_coef)
        {
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
            return dw;
        }
    }
}