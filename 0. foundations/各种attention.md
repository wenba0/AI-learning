参考：
[Attention进阶史（MHA, MQA, GQA, MLA） – 图神经网络公社](https://gnn.club/?p=2729)
### 原始attention
softmax对行做
每个O的输出需要当前的Q和历史所有的KV，详见[[kvcache相关#KVCache背景与attention计算]]

### MHA
MHA $n*d$的hidden_states将其按头数进行切分，头数h，每个头输入为$n*\frac{d}{h}$,每个头分别计算attention之后concat起来变成$n*d$的输出结果
### MQA
MQA 所有的头共用一个kv，只对Q按头进行切分，KV经过映射矩阵之后就是head_dim的 代码如下 （transformers中num_key_value_heads=1就是MQA 要是=num_attention_heads 那就是MHA）
![Megatron-LM sp|1125](../../images/20260204163653.png)
### GQA
GQA是MHA和MQA之间的平衡，一组head共享一个kv，要是所有的头分成g个group，linear输出维度是$g* \frac{d}{h}$，即num_key_value_heads * head_dims
### MLA
![Megatron-LM sp|1125](../../images/20260213110438.png)
deepseek v2中提出的，引入了latent特征表示，减少了kvcache的显存占用
![Megatron-LM sp|1125](../../images/20260213110717.png)
- 相较与MHA而已，MLA是再MHA之前多了一些qkv的处理，这看上去是多了一些计算，为什么会变的更高效？
- 位置编码信息RoPE为什么不直接加在$k_{t,i}^{C}$上，而要新创建一些$k_{t}^{R}$？
- 为什么对于查询向量q，也要进行潜向量的计算？

针对问题1，虽然看上去是多了一些计算，但是这种方式使得我们的KV Cache变小了。具体来说，我们只需要缓存一个较短的$c_{t}^{KV}$向量，而不是$k_{t}^{C}$和$v_{t}^{C}$。用白话说就是缓存长的变成缓存短的，缓存两个变成缓存一个。
针对问题2，是因为旋转位置编码RoPE与潜向量的计算不兼容，为了同时使用潜向量计算和旋转位置编码RoPE两个技术，只能多创建一个新的向量$k_{t}^{R}$来编码位置信息，将来通过向量合并将位置信息带入键向量。
针对问题3，将Q的输入也改为了低秩投影形式（潜向量），这与减少KV Cache无关。个人认为主要是为了特征的对齐，如果只对键 k 和值 v 进行潜在向量计算，而忽略查询 q，会导致 q 和 k、v 的特征空间不一致，影响注意力机制的效果。在注意力机制中，q、k、v是平等的输入，对它们进行相同的潜在向量计算可以保持模型的对称性和一致性。

MLA中KVcache会把压缩的latent kv和k的位置编码都存下来，两者的维度分别是kv_lora_rank 512 和qk_rope_head_dim 64，混存中间的latent kv相较于直接缓存原始的KV（MLA中latent kv经过两个上投影矩阵分别得到K和V，K与rope cat到一块，拿到QKV之后按照原始的方式计算attention）会节省很多空间
[DeepSeek MLA KV Cache占用计算 - 知乎](https://zhuanlan.zhihu.com/p/1939360765736887971)

且MLA的latent KV是所有head共享的，不受TP的切分影响，DSA同理；正常的MHA经过TP切分后每张卡上之后保留1/tp_size的KVCache
### DSA
deepseek sparse attention，会与kv进行筛选，只选择部分attention进行计算，GLM-5也用到了该方法
引入了lighting indexer 







### GDN  Linear Attention