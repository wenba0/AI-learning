操作的性能瓶颈有两种：
+ compute-bound：计算密集型
+ memory-bound：存储访问密集型
之前的大部分对于attention的改进聚焦于FLOPS（计算量），比如group attention等，flash attention聚焦于后者
##### GPU硬件层级
分为SRAM（存储空间小，带宽大）、HBM（显存）、DRAM
GPU的计算流程：将数据从显存（HBM）加载至on-chip的SRAM中，然后由SM（Streaming Multiprocessors，流式多处理器）读取并进行计算。计算结果再通过SRAM返回给显存。
![[Pasted image 20260129101416.png|750]]
##### softmax与safe softmax
标准softmax
对于输入向量 $\mathbf{z} = [z_1, z_2, \dots, z_n]$，其 Softmax 输出为：
$$

\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}, \quad \text{for } i = 1, 2, \dots, n

$$

safe softmax
为避免指数运算中出现上溢（overflow）或下溢（underflow），通常在计算时减去输入向量的最大值 $M = \max(\mathbf{z})$：
$$

\text{SafeSoftmax}(z_i) = \frac{e^{z_i - M}}{\sum_{j=1}^{n} e^{z_j - M}}, \quad \text{其中 } M = \max_{k}(z_k)

$$
该变换不改变 Softmax 的输出结果（因为分子分母同除以 $e^M$），但显著提升了数值稳定性。

##### softmax分块计算
safe softmax详细计算过程如下：$$max(x) = max([x_1, x_2, ..., ])\tag{公式1}$$

直觉上，难以分块计算的主要原因是它的分母“EXP求和项”依赖于输入向量中的每一个值。
