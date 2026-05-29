
tolearn
+ window attn 的原理 以及KVCache管理策略
+


**triton算子重编译问题：**
+ 为什么会触发重编译？
输入类型tl.constexpr表示编译期常量，在推理过程中传入不同的值就会重新编译
+ 为什么需要tl.constexpr?全部去掉不可以吗？
为了算子性能
+ 如何解决重编译带来的整网性能波动？
1 部分参数是不必要的，可以去掉tl.constexpr 2  可以在dummy_run中单独对某个triton算子写不同size的预热；业务实际场景中通过措施1和dummy_run默认的预热过程已经解决该问题了

**hybrid cache问题：**
问题发现：
问题描述：vLLM中没有类似sglang的双池方案，对于linear attention的cache，也是用kvblock来存放，没有连续，会造成空间浪费
问题解决：通过连续性算子支持
