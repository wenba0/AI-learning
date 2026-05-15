```
#硬件相关：
nproc  #查看cpu核数
lscpu  #查看CPU详细信息
export ASCEND_RT_VISIBLE_DEVICES=0
```

【模型性能优化与推理框架开发】
qwen2.5-vl量化与框架适配
qwen3.5-vl模型性能优化，包括xx等手段
qwen3.5性能优化（kvcache连续/非连续  flashcomm   dcp  算子优化(triton算子重编译 asc替换)）
+ triton算子为什么会触发重编译？tl.constexpr
【大模型精度问题定位与工具链开发】
精度工厂  客户面精度问题
量化开发、看看绩效自述 来写
测评工具开发