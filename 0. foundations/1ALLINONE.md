
tolearn
+ window attn 的原理 以及KVCache管理策略
+ lora 推理
+


**triton算子重编译问题：**
+ 为什么会触发重编译？
输入类型tl.constexpr表示编译期常量，在推理过程中传入不同的值就会重新编译
+ 为什么需要tl.constexpr?全部去掉不可以吗？
为了算子性能,它让 tile、循环、向量宽度、算法分支在编译期固定，全去掉的话在推理过程中实时的去分配资源、动态调整，性能会劣化

+ 如何解决重编译带来的整网性能波动？
	1 部分参数是不必要的，可以去掉tl.constexpr 
	2 ascendC算子替换（为什么ascendc不会有重编译：Triton 是 JIT （just in time）即时编译，AscendC 是 AOT(ahead of time) 离线编译、上线只有调用没有编译）
	3 可以在dummy_run中单独对某个triton算子写不同size的预热；
	实际业务场景中通过措施1、2和dummy_run默认的预热过程已经解决该问题了

**kvcache打满 dp同步 fused_mc2精度问题:**
[[Feature]Add xmask feature for dispatch_ffn_combine operator (only for w8a8 branch) by guanguan0308 · Pull Request #8560 · vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend/pull/8560)
问题定位：
逐渐收敛到KVCache打满+开启mc2会有精度掉点， 打点发现该场景下会有大量的0~9专家选择
问题描述：
1. KV cache 打满后，vLLM V1 会触发 preemption/recompute。
2. 某个 DP rank 上的请求被抢占后恢复，需要重新 prefill 较长上下文。
3. 其他 DP rank 可能仍然只是 decode，真实 token 数很少。
4. 在混部 / full graph / FULL_DECODE_ONLY 这类图模式下，不同 DP rank 为了 shape 对齐，会 pad 到同一个较大的 token 数。
5. 于是某些 rank 上出现大量 **padding token**。
6. 这些 padding token 进入了 fused MoE 路径。
7. 修复前的fused_mc2 路径没有 x active mask，padding token 没有被明确标记为无效 token。padding token 也参与 MoE routing。
8. 这些padding token会集中路由到少数专家，超过算子的最大处理长度，可能会把真实的token截掉，导致精度问题
问题解决：
为padding token增加mask，增加一个新的padding expert，原本的专家是0-255，padding expert的ID是256，算子内部做判断如果是padding token的话将其放到padding expert上，不参与真实专家的dispatch-ffn-combine过程

+ 为什么padding token会集中打到专家0-9？
因为选择的是topk=10，每个padding token都是0，那通过gating后每个专家的得分都是一样的，会按照默认的顺序选择0-9的专家
+ 为什么需要padding expert，直接在算子内部丢掉padding token不可以吗？
这样做是为了算子行为一致，如果在算子内删掉padding token，与token相关的一系列变量都需要改，将padding token路由到padding expert上后，算子内部对padding expert不做dispatch、ffn、combine的操作；不改变shape，只改变routing目标
+ fused mc2替换掉的是MOE中的哪部分计算逻辑？
gate--->softmax得到每个token对应的topk专家，后续的dispatch-ffn-combine交给融合算子去算
