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
输入一个B维的向量为例，safe softmax详细计算过程如下：
+ 公式1（计算最大值）：$m(x) = max([x_1, x_2, ..., x_B])$ 
+ 公式2（分子）：$f(x) = [e^{x_1 - m(x)},e^{x_2 - m(x)},..., e^{x_B - m(x)}]$
+ 公式3（分母）：$l(x) = {\sum_{i=1}^{B} f(x)_{i}}$
+ 公式4（计算结果）：$softmax(x)=\frac{f(x)}{l(x)}$
其中$f(x)$为向量，$l(x)$为标量，直觉上，难以分块计算的主要原因是它的分母“EXP求和项”依赖于输入向量中的每一个值。

输入一个2B维的向量，将其一分为二，$x = [x^{(1)}, x^{(2)}]$，先处理$x^{(1)}$，再处理$x^{(2)}$
+ 公式5（计算最大值）：$m(x^{(1)}) = max([x_1^{(1)}, x_2^{(1)}, ..., x_B^{(1)}])$ 
+ 公式6（分子）：$f(x^{(1)}) = [e^{x_1^{(1)} - m(x^{(1)})},e^{x_2^{(1)} - m(x^{(1)})},..., e^{x_B^{(1)} - m(x^{(1)})}]$
+ 公式7（分母）：$l(x^{(1)}) = {\sum_{i=1}^{B} f(x^{(1)})_{i}}$
+ 公式8（计算结果）：$softmax(x^{(1)})=\frac{f(x^{(1)})}{l(x^{(1)})}$
显然公式8的结果并不是分块1的softmax最终结果，因为分母求和不是全局的，分子的最大值也不是全局的
处理完分块1后保存$m(x^{(1)})$和$l(x^{(1)})$，相较于保存整个分块1的向量，只保存这两个标量结果的开销会小很多，此外还需要保存两个全局标量$m_{max}$和$l_{all}$，每处理了一个分块就会更新这两个全局标量，由于当前只处理了第一个分块，因此$m_{max}=m(x^{(1)})$， $l_{all}=l(x^{(1)})$；
接着用相同的方式处理分块2
+ 公式9（计算最大值）：$m(x^{(2)}) = max([x_1^{(2)}, x_2^{(2)}, ..., x_B^{(2)}])$ 
+ 公式10（分子）：$f(x^{(2)}) = [e^{x_1^{(2)} - m(x^{(2)})},e^{x_2^{(2)} - m(x^{(2)})},..., e^{x_B^{(2)} - m(x^{(2)})}]$
+ 公式11（分母）：$l(x^{(2)}) = {\sum_{i=1}^{B} f(x^{(2)})_{i}}$
+ 公式12（计算结果）：$softmax(x^{(2)})=\frac{f(x^{(2)})}{l(x^{(2)})}$
更新全局标量
+ 公式13(更新最大值）$m_{max}^{new}=max(m_{max}, m(x^{(2)}))$，不能替换旧的$m_{max}$，还会用到的；
+ 公式14(更新和）$l_{\text{all}}^{\text{new}} = e^{m_{\max} - m_{\max}^{\text{new}}} l_{\text{all}} + e^{m(x^{(2)}) - m_{\max}^{\text{new}}} l(x^{(2)})$ 
同时利用全局标量来更新分块1与分块2的softmax结果, softmax的分块计算动态更新, 数学上和全局统一计算的结果是一致的
+ 公式15(更新分块2): $softmax^{new}(x^{(2)})=\frac{softmax(x^{(2)}).l(x^{(2)}).e^{m(x^{(2)})-m_{max}^{new}}}{l_{\text{all}}^{\text{new}}}$
+ 公式16(更新分块1): $softmax^{new}(x^{(1)})=\frac{softmax(x^{(1)}).l(x^{(1)}).e^{m(x^{(1)})-m_{max}^{new}}}{l_{\text{all}}^{\text{new}}}$
##### flash attention计算流程
标准attention计算过程如下，需要在HBM和SRAM之间搬来搬去，内存读写bound严重影响模型性能
![[Pasted image 20260129143709.png|1125]]
flash attention利用分块计算的思路,将矩阵Q K V O分成很多小块逐步搬到SRAM中进行计算,减少了HBM的读写
![[Pasted image 20260129144144.png|1150]]
1. 依据特征维度d和SRAM大小选择合适的切分大小
2. 初始化O(0填充, attention最终的输出结果), l m (softmax分块动态计算过程中需要记录的每块数据的局部最大值与指数和)
3. 行切分Q, K, V(K计算的时候会被转置)
4. 切分O矩阵和l m
5. 外层循环, 遍历K V矩阵块
6. 搬到SRAM
7. 内层循环, 遍历Q O矩阵块和l m块
8. 搬到SRAM
9. 计算Q×K
10. 计算当前数据块的最大值与指数和, 并且缩放数据为saft softmax计算做准备
11. 根据传入的m l 与当前计算得到的m l 更新之
12. 这块公式太复杂了==  反正记住softmax可以动态更新的, 那O也是可以动态更新的
attention计算: $Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d_k}})V$
O分块计算的结果写入HBM中,不需要再修改了,

算子代码实现
![[Pasted image 20260129155722.png]]