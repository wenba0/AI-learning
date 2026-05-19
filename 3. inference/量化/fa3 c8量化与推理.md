### 场景1：full attention的量化
主要分为对attn中qkv linear的量化和KVCache的量化
第一部分就是正常的linear quant，第二部分为c8量化
##### attention quant

##### KVCache quant
[KVCache Quant-MindStudio26.0.0-昇腾社区](https://www.hiascend.com/document/detail/zh/mindstudio/2600/msTT_msIT/msModelSlim/docs/zh/quantization_algorithms/quantization_algorithms/kvcache_quant.md)
静态量化生成scale与offset
