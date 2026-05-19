
tolearn
+ window attn 的原理 以及KVCache管理策略
+

vllm推理框架
tp sp cp ep等并行方法
MHA gpa mla dsa moe等网络结构
flashattention、chunk_prefill、prefix_cache、mtp等优化特性
qwen3.5 dsv3.2  dsv4等模型结构 （这个maybe可以不要）
w8a8\kvcache fa3等常见量化方法，quarot smooth等离群值抑制方法

【模型性能优化与推理框架开发】
1 qwen2.5-vl系列A5机器上模型w8（mxfp8）a8c8(fp8)量化适配，精度掉点1%以内，性能相较基线提升2.x倍
3 qwen3.5性能优化（kvcache连续/非连续、flashcomm适配、算子优化(triton算子重编译 asc替换)与融合算子接入框架）这部分我是作为模型侧人员的角度去拉通各个组件，来提升端到端的性能，最终在典型case场景性能达到1.3xH20
2 qwen2.5-vl模型量化、性能优化与profiling分析，优化点包括：rmsnorm quant融合算子替换，add 0 bias 操作去除，

【大模型精度问题定位与工具链开发】
1 作为精度工厂组织成员，定位字节、电信等客户面精度问题，问题根因大多为配置不对齐、算子精度等问题
2 量化工具开发，适配quarot\smooth等离群值抑制方法多卡量化适配，构建modelslim多卡量化的能力，提升量化效率，开发mxfp8量化能力，补齐昇腾A5机器量化能力
3 AISBench大模型测评工具开发，开发多模态测评、多轮对话测评等特性，满足前沿模型和prefix cache等特性的测评诉求，