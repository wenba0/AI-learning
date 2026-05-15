#kvcache
### KVCache背景与attention计算
==问题1：==为什么只保存KV 不缓存Q呢，每次新的一轮推理只用到新生成的Q、新生成的KV以及老的KV
从推理角度
prefill阶段：对于输入的n个token，输出结果为$n*d$ ,只选择最后一个维度的特征$1*d$用于生成一个新的token
decode阶段：每次将前一个生成的token拼接在后面继续输出一个token
==问题2==： prefill阶段我只需要计算最后一个token的Q就可以了吧  然后与全量的K和V进行计算拿到一个$1*d$的输出 用于预测下一个token，不需要全量的Q吧？
这样肯定是错误的！大模型是由很多DecoderLayer层组成的，每一层的输出结果作为下一层的输入，层层去提取特征，**KVcache会存储每一层DecoderLayer中的KV值**，因此在prefill阶段一定需要全量的QKV，QKV的映射矩阵每一层都有的，
从kvcache角度
prefill阶段：对于输入的n个token，将每一层中的KV值都保存下来
decode阶段：计算新生成的token，计算其QKV，结合之前所有的KV值计算出这个Q对应的$1*d$的输出 并预测token，同事把过程中的KV值也都保存下来 用于下一个token的生成过程

另外记住 attention是有mask的，每个$Q_n$只能和之前所有的KV值进行计算，这个地方是有mask的，对应下图中QK相乘结果的矩阵，也意味着每个token只能看到之前的所有token
再注意 QKV的映射矩阵不是同一个，有的直接初始化的三个linear 有的是一个linear，但是输出维度是$embed_{dim} * 3$，输出后再拆分成三份 
![Megatron-LM tp|1160](../../images/20260204160953.png)

