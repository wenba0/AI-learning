megatron tp就是对连续的两个矩阵乘，第一个列切  第二个行切， 只需要进行一次all reduce的通信
tp只对attention内和mlp内的权重有效，
#### Megatron-LM tp
对attention和mlp两个模块的权重进行切分，这两个模块都是有矩阵乘的部分，tp的前提的矩阵乘运算的可切分性；
![[Pasted image 20260105153827.png|875]]
以MLP为例，计算过程如下
$$Z=Dropout(GeLU(XA)B)$$
MLP的tp过程如下，权重矩阵A列切，权重矩阵B行切，最终得到的结果需要进行reduce(求和)
![[Pasted image 20260105154138.png|950]]
矩阵乘可切分性如下所示：
只有一个矩阵乘的话 权重列切
![[Pasted image 20260105154315.png|725]]        ![[Pasted image 20260105154327.png|775]]



