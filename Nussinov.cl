// Copyright (c) 2009-2011 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

// 3x3 median filter kernel based on partial sort

// Scalar version of kernel
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define N 1000

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


int can_pair(__global char* input, int a, int b) {

	return (((
		(input[a] == 'A' && input[b] == 'U') || (input[a] == 'U' && input[b] == 'A') ||
		(input[a] == 'G' && input[b] == 'C') || (input[a] == 'C' && input[b] == 'G') ||
		(input[a] == 'G' && input[b] == 'U') || (input[a] == 'U' && input[b] == 'G')
		)))/* && (a < b - 1))) */ ? 1 : 0;
}


// Tile Correction
__kernel void NussinovDeviceKernelTileCor(__global int S[N][N], __global char* RNA, __global int* lb_arr, __global int *c1_arr, __global int ST[N][N])
{
    const int x = get_global_id(0);
    const int width = get_global_size(0);
    int lb = lb_arr[0];
    int c1 = c1_arr[0];
    
int c2,c3,c5,c6,c7,c9,c11,c10,c4,c12;
   c3 = lb + x;

    for( c4 = 0; c4 <= 1; c4 += 1) {
      if (c4 == 1) {
        for( c9 = N - c1 + 129 * c3; c9 <= min(N - 1, N - c1 + 129 * c3 + 127); c9 += 1)
          for( c10 = max(0, N - c1 + 129 * c3 - c9 + 1); c10 <= 1; c10 += 1) {
            if (c10 == 1) {
              S[(N-c1+c3-1)][c9] = max(S[(N-c1+c3-1)][c9], S[(N-c1+c3-1)+1][c9-1]) + can_pair(RNA, (N-c1+c3-1), c9);
             c1=c1;
            } else
              for( c11 = 128 * c3 + 1; c11 <= -N + c1 - c3 + c9; c11 += 1)
                S[(N-c1+c3-1)][c9] = max(S[(N-c1+c3-1)][c11+(N-c1+c3-1)] + S[c11+(N-c1+c3-1)+1][c9], S[(N-c1+c3-1)][c9]);
              c1=c1;
          }
      } else
        for( c5 = 0; c5 <= 8 * c3; c5 += 1)
          for( c9 = N - c1 + 129 * c3; c9 <= min(N - 1, N - c1 + 129 * c3 + 127); c9 += 1)
            for( c11 = 16 * c5; c11 <= min(128 * c3, 16 * c5 + 15); c11 += 1)
              S[(N-c1+c3-1)][c9] = max(S[(N-c1+c3-1)][c11+(N-c1+c3-1)] + S[c11+(N-c1+c3-1)+1][c9], S[(N-c1+c3-1)][c9]);
             c1=c1;            
    }

   
}

// Space Time
__kernel void NussinovDeviceKernelTSTILE(__global int S[N][N], __global char* RNA, __global int* lb_arr, __global int *c1_arr, __global int ST[N][N])
{
    const int x = get_global_id(0);
    const int width = get_global_size(0);
    int lb = lb_arr[0];
    int c0 = c1_arr[0];
    
    int c1,c2,c3,c5,c6,c7,c9,c11,c10,c4,c12;

    c1 = lb + x;

    for( c3 = 16 * c0 - 16 * c1 + 1; c3 <= min(min(N - 1, 16 * c1 + 15), 16 * c0 - 16 * c1 + 16); c3 += 1) {
      for( c4 = 0; c4 <= c0 - c1; c4 += 1)
        for( c6 = max(-N + 16 * c1 + 1, -N + c3 + 1); c6 <= min(0, -N + 16 * c1 + 16); c6 += 1) {
          for( c10 = 16 * c4; c10 <= min(c3 - 1, 16 * c4 + 15); c10 += 1)
            S[(-c6)][(c3-c6)] = MAX(S[(-c6)][c10+(-c6)] + S[c10+(-c6)+1][(c3-c6)], S[(-c6)][(c3-c6)]);
          if (c1 + c4 == c0 && 16 * c0 + c6 + 15 >= 16 * c1 + c3)
            S[(-c6)][(c3-c6)] = MAX(S[(-c6)][(c3-c6)], S[(-c6)+1][(c3-c6)-1] + can_pair(RNA, (-c6), (c3-c6)));
        }
      for( c4 = max(c0 - c1 + 1, -c1 + (N + c3) / 16 - 1); c4 <= min((N - 1) / 16, -c1 + (N + c3 - 1) / 16); c4 += 1)
        for( c6 = max(max(-N + 16 * c1 + 1, -N + c3 + 1), c3 - 16 * c4 - 15); c6 <= min(-N + 16 * c1 + 16, c3 - 16 * c4); c6 += 1)
          S[(-c6)][(c3-c6)] = MAX(S[(-c6)][(c3-c6)], S[(-c6)+1][(c3-c6)-1] + can_pair(RNA, (-c6), (c3-c6)));
    }

   
}


// Pluto compiler
__kernel void NussinovDeviceKernelPluto(__global int S[N][N], __global char* RNA, __global int* lb_arr, __global int *c1_arr, __global int ST[N][N])
{
    const int x = get_global_id(0);
    const int width = get_global_size(0);
    int lb = lb_arr[0];
    int t2 = c1_arr[0];
   int t1, t3, t4, t5, t6, t7, t8, t9, t10; 
   t4 = lb + x;

      for (t5=max(max(-N+2,16*t2-16*t4),-16*t4-14);t5<=min(0,16*t2-16*t4+15);t5++) {
        for (t7=max(16*t4,-t5+1);t7<=min(N-1,16*t4+15);t7++) {
          for (t9=0;t9<=t5+t7-1;t9++) {
            S[-t5][t7] = MAX(S[-t5][t9+-t5] + S[t9+-t5+1][t7], S[-t5][t7]);;
          }
          S[-t5][t7] = MAX(S[-t5][t7], S[-t5+1][t7-1] + can_pair(RNA, -t5, t7));;
        }
      }
   
}

// Li method
__kernel void NussinovDeviceKernelLI(__global int S[N][N], __global char* RNA, __global int* lb_arr, __global int *diag_arr, __global int ST[N][N])
{
    const int x = get_global_id(0);
    const int width = get_global_size(0);
    int lb = lb_arr[0];
    int diag = diag_arr[0];
    
    int i,j,k;
    int row,col;
    char a,b;
    int _max,t;
    
    row = lb + x;

    col = diag + row;
    a = RNA[row];
    b = RNA[col];
    _max = S[row+1][col-1] + can_pair(RNA, row, col);
    for(k=row; k <=col-1; k++){
        t = S[row][k] + S[col][k+1];
        _max = max(_max, t);
    }
    S[row][col] = S[col][row] =  _max;
    

}

// Transpose method
__kernel void NussinovDeviceKernelTranspose(__global int S[N][N], __global char* RNA, __global int* lb_arr, __global int *diag_arr, __global int ST[N][N])
{
    const int x = get_global_id(0);
    const int width = get_global_size(0);
    int lb = lb_arr[0];
    int c0 = diag_arr[0];
    int c1 = lb + x;
    int c4;


    
    for (c4 = 0; c4 <= c0 - c1; c4 += 1) {
        S[(N - c1 - 1)][(N + c0 - 2 * c1)] = MAX(S[(N - c1 - 1)][c4 + (N - c1 - 1)] + ST[(N + c0 - 2 * c1)][c4 + (N - c1 - 1) + 1], S[(N - c1 - 1)][(N + c0 - 2 * c1)]);
        ST[(N + c0 - 2 * c1)][(N - c1 - 1)] = S[(N - c1 - 1)][(N + c0 - 2 * c1)];
    }
    S[(N - c1 - 1)][(N + c0 - 2 * c1)] = MAX(S[(N - c1 - 1)][(N + c0 - 2 * c1)], S[(N - c1 - 1) + 1][(N + c0 - 2 * c1) - 1] + can_pair(RNA, (N - c1 - 1), (N + c0 - 2 * c1)));
         

}

__kernel void NussinovDeviceKernelParallel(__global int S[N][N], __global char* RNA, __global int* lb_arr, __global int *diag_arr, __global int ST[N][N])
{

    const int x = get_global_id(0);
    const int width = get_global_size(0);
    int lb = lb_arr[0];
    int i = diag_arr[0];
    int j = lb + x;
    int k;


  //  for (i = N - 1; i >= 0; i--) {
     //   for (j = i + 1; j < N; j++) {
            for (k = 0; k < j - i; k++) {
                S[i][j] = max(S[i][k + i] + S[k + i + 1][j], S[i][j]);
            }
            for (k = 0; k < 1; k++) {
                S[i][j] = max(S[i][j], S[i + 1][j - 1] + can_pair(RNA, i, j));

            }
      //  }
  //  }
 }

