#include "model.h"
#include "trap.h"

// conv1: input 3x32x32, output 12x28x28, activation relu
// pool1: input 12x28x28, output 12x14x14
// conv2: input 12x14x14, output 32x12x12, activation relu
// pool2: input 32x12x12, output 32x6x6
// fc1: input 32*6*6=1152, output 256, activation relu
// fc2: input 256, output 64, activation relu
// fc3: input 64, output 10

// tool/model_lab2.pth 包含 int8 量化后的权重和输出激励
// 提供的脚本可以导出 int8 量化后的输入，权重，和仅用于 fc3 层的 int16 偏置
// 以及 int8 量化的缩放因子
// 到 ./data/bin 文件中

// 卷积层的激励和权重是 NCHW 格式存储的四维 tensor，NCHW 格式
// 但 lab3 告诉我们 NHWC 是访存友好的，因此本项目的makefile提供了多种选择，
// 可以修改 makefile 的 8-11 行来决定是 NCHW 还是 NHWC
// N: batch size
// H: height
// W: width
// C: channels
// 可以想象成四重循环，由外到里分别是 N, H, W, C
// 软件层面，我们直接使用 NHWC 格式
// 类似地，fc层的激励和权重是二维的，可以修改makefile来转置它们
// 中间结果会被输出到 ./data/npy 中

int8_t (*input) = (int8_t (*))ADDR_INPUT;


// nhwc
// 12x5x5x3
int8_t (*wconv1) = (int8_t (*))ADDR_WCONV1;
int8_t (*sconv1) = (int8_t *)ADDR_SCONV1;
// 32x3x3x12
int8_t (*wconv2) = (int8_t (*))ADDR_WCONV2;
int8_t (*sconv2) = (int8_t *)ADDR_SCONV2;
// 32x256x6x6
int8_t (*wfc1) = (int8_t (*))ADDR_WFC1;
int8_t (*sfc1) = (int8_t *)ADDR_SFC1;
// wfc2 和 wfc3 不转置，因为暂时不知道转置这俩有什么用
// 256x64
int8_t (*wfc2) = (int8_t (*))ADDR_WFC2;
int8_t (*sfc2) = (int8_t *)ADDR_SFC2;
// 64x10
int8_t (*wfc3) = (int8_t (*))ADDR_WFC3;
// 10x2, x2 是因为 bias 是 int16 的，要占 int8 的两倍空间，实际上只是个 1d 向量
int16_t (*bfc3) = (int16_t *)ADDR_BFC3;
int8_t (*sfc3) = (int8_t *)ADDR_SFC3;

//各层的输出可以存在下面
// outconv1: 12x28x28
int8_t (*outconv1) = (int8_t (*))ADDR_OUTCONV1;
// outpool1: 12x14x14
int8_t (*outpool1) = (int8_t (*))ADDR_OUTPOOL1;
// outconv2: 32x12x12
int8_t (*outconv2) = (int8_t (*))ADDR_OUTCONV2;
// outpool2: 32x6x6
int8_t (*outpool2) = (int8_t (*))ADDR_OUTPOOL2;
int8_t (*outpool2_1D) = (int8_t *)ADDR_OUTPOOL2;
// outfc1: 256
int8_t (*outfc1) = (int8_t (*))ADDR_OUTFC1;
// outfc2: 64
int8_t (*outfc2) = (int8_t (*))ADDR_OUTFC2;
// outfc3: 10
int8_t (*outfc3) = (int8_t (*))ADDR_OUTFC3;

int8_t relu(int8_t x) {
  return x > 0 ? x : 0;
}

// a / b, b 是 2 的幂，用移位代替除法，但周期数应该可以再优化
int8_t div_pow(int8_t a, int8_t b) {
  int8_t p = 0;
  while (b > 1) {
    p++;
    b >>= 1;
  }
  return a >> p;
}

// todo: 量化，内联 v 指令
void conv(int8_t *input, int8_t *weight, int8_t scale, int8_t *output, int N, int H, int W, int C, int K, int R, int S) {
  // input: N H W C
  // weight: K R S C
  // output: N H W K
  int8_t rw = R / 2;
  int8_t sw = S / 2;
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      for (int p = 0; p < H; p++) {
        for (int q = 0; q < W; q++) {
          int out_idx = n*H*W*K + p*W*K + q*K + k;
          for (int i = p-rw < 0 ? 0 : p-rw; i < p+rw+1 && i < H; i++) {
            for (int j = q-sw < 0 ? 0 : q-sw; j < q+sw+1 && j < W; j++) {
              for (int c = 0; c < C; c++) {
                // output[n][p][q][k] += (input[n][i][j][c] * weight[k][i-p+2][j-q+2][c]);
                int in_idx = n*H*W*C + i*W*C + j*C + c;
                int w_idx = k*R*S*C + (i-p+2)*S*C + (j-q+2)*C + c;
                output[out_idx] += (input[in_idx] * weight[w_idx]);
              }
            }
          }
          //relu
          output[out_idx] = relu(output[out_idx]);
        }
      }
    }
  }
}


// todo：pool 不用量化，但仍要内联 v 指令
void pool(int8_t *input, int8_t *output, int N, int H, int W, int C, int pool_height, int pool_width) {
  // input: N H W C
  // output: N H/pool_height W/pool_width C
  // maxpooling2D
  int pooled_height = H / pool_height;
  int pooled_width = W / pool_width;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < pooled_height; p++) {
        for (int q = 0; q < pooled_width; q++) {
          int out_idx = n*H*W*C + p*W*C + q*C + c;
          for (int i = p*pool_height; i < p*pool_height+pool_height && i < H; i++) {
            for (int j = q*pool_width; j < q*pool_width+pool_width && j < W; j++) {
              // output[n][p][q][c] = output[n][p][q][c] > input[n][i][j][c] ? output[n][p][q][c] : input[n][i][j][c];
              int in_idx = n*H*W*C + i*W*C + j*C + c;
              output[out_idx] = output[out_idx] > input[in_idx] ? output[out_idx] : input[in_idx];
            }
          }
          // relu
          output[out_idx] = relu(output[out_idx]);
        }
      }
    }
  }
}

// todo：量化，v扩展，bias
// 1D x 1D  fc1 和 fc2 没有 bias， fc3 有
void fc(int8_t *input, int8_t *weight, int8_t scale, int8_t *output, int N, int W, int R, bool has_bias, int16_t *bias) { 
  // 以 fc1 为例， fc1 是 32x6x6 的输入（reshape 到 1D），32x6x6x256 的权重，
  // 相当于 1x1152 和 1152x256 的矩阵乘法，输出 1x256

  // input:1xW
  // weight: WxR
  // output: 1xR
  for (int n = 0; n < N; n++) {
    for (int i = 0; i < R; i++) {
      int out_idx = n*R + i;
      for (int j = 0; j < W; j++) {
        // output[n][i] += (input[n][j] * weight[j][i]);
        int in_idx = n*W + j;
        int w_idx = j*R + i;
        output[out_idx] += (input[in_idx] * weight[w_idx]);
      }
      //relu
      output[out_idx] = relu(output[out_idx] + has_bias ? bias[i] : 0) ;
    }
  }
}




void nn() {
  // 先不考虑量化，量化要把除法操作优化成移位操作（编译器会干吗）
  // 默认不会，那么用 div_pow 函数代替
  conv(input, wconv1, *sconv1, outconv1, 1, 32, 32, 3, 12, 5, 5);
  pool(outconv1, outpool1, 1, 28, 28, 12, 2, 2);
  conv(outpool1, wconv2, *sconv2, outconv2, 1, 14, 14, 12, 32, 3, 3);
  pool(outconv2, outpool2, 1, 12, 12, 32, 2, 2);
  fc(outpool2_1D, wfc1, *sfc1, outfc1, 1, 1152, 256, false, NULL);
  fc(outfc1, wfc2, *sfc2, outfc2, 1, 256, 64, false, NULL);
  fc(outfc2, wfc3, *sfc3, outfc3, 1, 64, 10, true, bfc3);
}

int main () {
  nn();
  return 0;
}
