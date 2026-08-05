整体的流程如下：
1. 读取模型配置、参数配置，并做校验
2. 在platform较严重，根据配置分别计算full和mamba的page_size，通过`Platform._align_hybrid_block_size`对两者做对齐(通过Executor调用`current_platform.update_block_size_for_backend(self.vllm_config)`), 算出最终的page_size
3. 通过EngineCore初始化KVCache(`_initialize_kv_caches`方法)，包括计算KV可用显存、KVGroups分组、KVTensors划分
4. 通过`self.model_executor.initialize_from_config(kv_cache_configs)`下发给Worker--->ModelRunner做真正的初始化：分配原始int8 buffer并reshape、通过`bind_kv_cache()`绑定到 forward context

~~当前SGLang有双池方案，且mamba block支持bf16，vLLM当前只有一种kvblock，要full attention和linear attention都要用，同时写入kv、conv_state、ssm_state，非连续----》连续，非连续对于conv_state和ssm_state没到block size时进行pad，会造成显存浪费，连续的方案就不会浪费了，需要fia等算子支持~~

#### hybrid attn 分页管理机制
减层为8层，3层linear+full 两次
**kvgroups如何分组的**：groups数量根据最小重复pattern的大小来的，共有4个group，因此`kv_cache_groups`中有4个`KVCacheGroupSpec`，每个group对应一种attn的spec，当前是`linear0, linear1, linear2, full_attn`循环两次，即4个group，要是模型中只有一种attn的话，就只有一个group，保证每个group内的attn语义是一样的，每个group会对应一个block_table
**物理KVCacheTensor怎么切**：根据最小重复pattern的数量来的，`kv_cache_groups` 是逻辑分组；`kv_cache_tensors` 是 worker 真正分配显存的物理 tensor。将物理空间切分成group_size份，即2份，对于一个10层的llama，attn都是一样的，对应有1个group，物理Tensor切成10份，因为每一层的KVCache都不一样

EngineCore初始化kv后打印的KVCacheConfig如下：
```python
 KVCacheConfig(num_blocks=13227, 
 
 kv_cache_tensors=[KVCacheTensor(size=28226629632, shared_by=['language_model.model.layers.3.self_attn.attn', 'language_model.model.layers.0.linear_attn', 'language_model.model.layers.1.linear_attn', 'language_model.model.layers.2.linear_attn']), 
					KVCacheTensor(size=28226629632, shared_by=['language_model.model.layers.7.self_attn.attn', 'language_model.model.layers.4.linear_attn', 'language_model.model.layers.5.linear_attn', 'language_model.model.layers.6.linear_attn'])], 
					
kv_cache_groups=[KVCacheGroupSpec(layer_names=['language_model.model.layers.3.self_attn.attn', 'language_model.model.layers.7.self_attn.attn'],kv_cache_spec=FullAttentionSpec(
block_size=1024,num_kv_heads=2, head_size=256, dtype=torch.bfloat16, kv_quant_mode=<KVQuantMode.NONE: 0>, page_size_padded=2134016, head_size_v=256, sliding_window=None, attention_chunk_size=None), is_eagle_group=False), 
				KVCacheGroupSpec(layer_names=['language_model.model.layers.0.linear_attn', 'language_model.model.layers.4.linear_attn'], kv_cache_spec=MambaSpec(block_size=133000, shapes=((3, 6144), (16, 128, 128)), dtypes=(torch.bfloat16, torch.float32), page_size_padded=2134016, mamba_type=<MambaAttentionBackendEnum.GDN_ATTN: 'vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend'>, mamba_cache_mode='none', num_speculative_blocks=0), is_eagle_group=False), 
				KVCacheGroupSpec(layer_names=['language_model.model.layers.1.linear_attn', 'language_model.model.layers.5.linear_attn'], kv_cache_spec=MambaSpec(block_size=133000, shapes=((3, 6144), (16, 128, 128)), dtypes=(torch.bfloat16, torch.float32), page_size_padded=2134016, mamba_type=<MambaAttentionBackendEnum.GDN_ATTN: 'vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend'>, mamba_cache_mode='none', num_speculative_blocks=0), is_eagle_group=False), 
				KVCacheGroupSpec(layer_names=['language_model.model.layers.2.linear_attn', 'language_model.model.layers.6.linear_attn'], kv_cache_spec=MambaSpec(block_size=133000, shapes=((3, 6144), (16, 128, 128)), dtypes=(torch.bfloat16, torch.float32), page_size_padded=2134016, mamba_type=<MambaAttentionBackendEnum.GDN_ATTN: 'vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend'>, mamba_cache_mode='none', num_speculative_blocks=0), is_eagle_group=False)])
```

#### 不同attn spec的page_size如何确定
==Mamba==
通过gated_delta_net_state_shape函数计算shape, 然后通过`page_size = conv_state大小+ssm_state大小 = byteofdtype(conv_state_dtype)*conv_state_shape+byteofdtype(ssm_state_dtype)*ssm_state_shape`

```python
# 计算shape
def gated_delta_net_state_shape(cls,tp_world_size: int,num_k_heads: int,num_v_heads: int,head_k_dim: int,head_v_dim: int,conv_kernel_size: int,num_spec: int = 0,):
	conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads # 128 * 16 * 2 + 128 * 16 = 6144
	conv_state_shape = cls._orient_conv_shape(
		divide(conv_dim, tp_world_size), # 没开TP
		conv_kernel_size - 1 + num_spec, # 没开MTP，conv_kernel_size=4，因此conv_state_shape=(3, 6144)
	)
	temporal_state_shape = (
		divide(num_v_heads, tp_world_size), # 16
		head_v_dim,                         # 128
		head_k_dim,                         # 128
	)
	return conv_state_shape, temporal_state_shape # (3, 6144), (16, 128, 128)

# 计算page_size
class MambaSpec(KVCacheSpec):
    ......
    
    @property
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)   # conv_state bf16, ssm_state fp32,因此page_size=3*6144*2+16*128*128*4=1085440 bytes 
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size
```
==full attn==

#### hybrid attn的page_size对齐
两个地方：1 `Platform._align_hybrid_block_size`根据mamba来调整full的block_size   2 生成KV groups时如果还是不对齐的话通过`unify_kv_cache_spec_page_size()`对齐
all align none

#### mamba如何做prefix cache