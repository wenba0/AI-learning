modelslim命令：
export ASCEND_RT_VISIBLE_DEVICES=1

msmodelslim quant --model_path /data0/ljj/Qwen2.5-VL-7B-Instruct --save_path /data0/ljj/qwen_quant/  --device npu --model_type Qwen2.5-VL-7B-Instruct --config_path /usr/local/python3.11.13/lib/python3.11/site-packages/msmodelslim/lab_practice/qwen2_5_vl/qwen2.5_vl_7b_w8a8.yaml --trust_remote_code True

### qwen2.7-vl-7b模型推理信息
```
input_ids=[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655,
         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
         151655, 151653,  74785,    419,   2168,     13, 151645,    198, 151644,
          77091,    198]]
```input_ids = []