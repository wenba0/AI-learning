tolearn
+ window attn 的原理 以及KVCache管理策略
+


【模型性能优化与推理框架开发】
1 qwen2.5-vl系列A5机器上模型w8（mxfp8）a8c8(fp8)量化适配，精度掉点1%以内，xingneng 相较A2机器提升2.x倍
3 qwen3.5性能优化（kvcache连续/非连续、flashcomm、算子优化(triton算子重编译 asc替换)与融合算子接入框架）这部分我是作为一个owner的角度去拉通各个组件，来提升端到端的性能，最终在典型case场景性能达到1.3xH20
2 qwen2.5-vl模型性能优化，包括xx等手段

【大模型精度问题定位与工具链开发】
1 精度工厂  客户面精度问题
2 量化工具开发、看看绩效自述 来写
3 测评工具开发