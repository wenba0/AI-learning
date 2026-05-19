### 场景1：full attention的量化
主要分为对attn中qkv linear的量化和KVCache的量化
第一部分就是正常的linear quant，第二部分为c8量化
##### attention quant

##### KVCache quant
modelslim资料（量化逻辑与量化配置）[KVCache Quant-MindStudio26.0.0-昇腾社区](https://www.hiascend.com/document/detail/zh/mindstudio/2600/msTT_msIT/msModelSlim/docs/zh/quantization_algorithms/quantization_algorithms/kvcache_quant.md)
MindIE资料（量化输出件与量化推理流程）[KV Cache int8-MindIE3.0.0-昇腾社区](https://www.hiascend.com/document/detail/zh/mindie/300/mindiellm/llmdev/user_guide/feature/kv_cache_int8.md)
![[Pasted image 20260519105314.png]]
推理时先利用量化参数将浮点的kv量化为int8，然后通过ReshapeAndCache将int8的kv保存下来，计算attention的时候将int8的kv、对应的反量化参数以及浮点的q一起送给fia算子，伪代码如下：
```python
# /vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py
class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
	def forward():
		self._prepare_c8_scales(layer, query.device)
		key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
		query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
		return self._forward_c8_decode(query, attn_metadata, output, layer)
	def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
	        raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)
        # 量化参数 切分出当前tp_rank对应的kv_head 同时reshape
        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)  # 
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        # 反量化参数 reshape成bnsd的格式，后续送给fia算子内部进行反量化与attn的计算
        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.view(bnsd).contiguous()
        layer._c8_k_aq_offset = layer._c8_k_offset.view(bnsd).contiguous()
	  def _forward_c8_decode():
		  attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            value_antiquant_scale=layer._c8_v_aq_scale,
            value_antiquant_offset=layer._c8_v_aq_offset,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            sparse_mode=0,
        )
```
fia算子文档：[torch_npu.npu_fused_infer_attention_score-Ascend Extension for PyTorch6.0.RC3-昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000768.html)
算子功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当Query矩阵的S为1，进入IncreFlashAttention分支，其余场景进入PromptFlashAttention分支。（同时支持prefill和decode，内部做判断）