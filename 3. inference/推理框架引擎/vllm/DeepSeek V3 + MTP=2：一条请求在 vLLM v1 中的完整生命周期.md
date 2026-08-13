

  

> 本文以 **DeepSeek V3**（MLA Attention + MoE FFN）配合 **MTP（Multi-Token Prediction）num_spec_tokens=2** 为例，

> 从一条请求到达 vLLM 到输出全部 token 的完整流程逐步追踪，附带关键代码块和变量值。

>

> 基于 vLLM v1 架构（异步调度模式 `async_scheduling=True`）。

  

---

  

## 目录

  

1. [整体架构概览](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#1-整体架构概览)

2. [阶段一：前端接收请求](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#2-阶段一前端接收请求asyncllm)

3. [阶段二：IPC 传输到后端进程](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#3-阶段二ipc-传输到后端进程)

4. [阶段三：后端 EngineCore 忙循环处理请求](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#4-阶段三后端-enginecore-忙循环处理请求)

5. [阶段四：Scheduler 调度](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#5-阶段四scheduler-调度)

6. [阶段五：Worker 执行 execute_model — 前处理](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#6-阶段五worker-执行-execute_model--前处理)

7. [阶段六：GPU Forward — DeepSeek V3 模型推理](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#7-阶段六gpu-forward--deepseek-v3-模型推理)

8. [阶段七：sample_tokens — 采样 + MTP 投机解码](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#8-阶段七sample_tokens--采样--mtp-投机解码)

9. [阶段八：异步输出拷贝 + 调度下一步](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#9-阶段八异步输出拷贝--调度下一步)

10. [阶段九：Scheduler update_from_output — 状态结算](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#10-阶段九scheduler-update_from_output--状态结算)

11. [阶段十：前端输出处理与交付用户](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#11-阶段十前端输出处理与交付用户)

12. [Decode 稳态：Step N+1 的流水线重叠](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#12-decode-稳态step-n1-的流水线重叠)

13. [异步调度的核心收益总结](DeepSeek%20V3%20+%20MTP=2：一条请求在%20vLLM%20v1%20中的完整生命周期.md#13-异步调度的核心收益总结)

  

---

  

## 1. 整体架构概览

  

```

┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐

│   前端进程 (AsyncLLM)                 │     │   后端进程 (EngineCoreProc)            │

│                                      │     │                                      │

│  ┌──────────────┐ ┌───────────────┐  │ ZMQ │  ┌─────────────┐  ┌──────────────┐  │

│  │InputProcessor│ │OutputProcessor│ ◄──────► │ EngineCore  │  │AsyncScheduler│  │

│  └──────────────┘ └───────────────┘  │     │  └─────────────┘  └──────────────┘  │

│                                      │     │                                      │

│  ┌──────────────┐  ┌──────────────┐  │     │  ┌─────────────────────────────────┐ │

│  │AsyncMPClient │  │output_handler│  │     │  │  run_busy_loop()                │ │

│  └──────────────┘  └──────────────┘  │     │  │    ├─ _process_input_queue()    │ │

│                                      │     │  │    ├─ step_with_batch_queue()   │ │

│  ┌──────────────┐                    │     │  │    └─ post_step()               │ │

│  │ per-request   │                    │     │  └─────────────────────────────────┘ │

│  │ generate()    │                    │     │                                      │

│  │ coroutine     │                    │     │  ┌──────────────────────────────┐    │

│  └──────────────┘                    │     │  │ Worker (GPUModelRunner)       │    │

│                                      │     │  │  ├─ execute_model()           │    │

│                                      │     │  │  ├─ sample_tokens()           │    │

│                                      │     │  │  └─ EagleProposer (MTP)      │    │

└──────────────────────────────────────┘     └──────────────────────────────────────┘

```

  

**关键配置**（DeepSeek V3 + MTP=2）：

- `speculative_config.method = "mtp"` → `use_eagle()` 返回 `True`

- `speculative_config.num_speculative_tokens = 2`

- `async_scheduling = True`

- `drafter` 类型：`EagleProposer`（MTP 在 vLLM 中复用 Eagle 的代码路径）

- `max_concurrent_batches = 2`（batch_queue 深度为 2）

- `step_fn = step_with_batch_queue`（异步调度使用的 step 函数）

  

---

  

## 2. 阶段一：前端接收请求（AsyncLLM）

  

用户通过 OpenAI 兼容 API 发起请求，最终进入 `AsyncLLM.generate()`：

  

```555:594:vllm/v1/engine/async_llm.py

        # ... generate() 是一个 async generator

        q: RequestOutputCollector | None = None

        try:

            q = await self.add_request(

                request_id,

                prompt,

                sampling_params,

                # ...

            )

            finished = False

            while not finished:

                out = q.get_nowait() or await q.get()

                assert isinstance(out, RequestOutput)

                finished = out.finished

                if out is not STREAM_FINISHED:

                    yield out

```

  

**`add_request` 内部调用 `InputProcessor.process_inputs()`：**

  

```366:380:vllm/v1/engine/async_llm.py

            request = self.input_processor.process_inputs(

                request_id,

                prompt,

                params,

                arrival_time=arrival_time,

                lora_request=lora_request,

                tokenization_kwargs=tokenization_kwargs,

                trace_headers=trace_headers,

                priority=priority,

                data_parallel_rank=data_parallel_rank,

                supported_tasks=await self.get_supported_tasks(),

            )

            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

            self.input_processor.assign_request_id(request)

```

  

**此步骤执行的操作**：

- Tokenization：将文本转为 token ids

- 多模态特征处理（如果有）

- 参数校验、request_id 分配

- 创建 `EngineCoreRequest` 对象

  

**关键变量值示例**（假设 prompt = "Hello, how are you?"）：

```

request.prompt_token_ids = [1, 8532, 11, 574, 527, 499, 30]  # 7 tokens

request.sampling_params.max_tokens = 100

request.request_id = "req-001"

```

  

**为每个请求创建输出收集器**：

  

```388:388:vllm/v1/engine/async_llm.py

        queue = RequestOutputCollector(params.output_kind, request.request_id)

```

  

内部使用 `asyncio.Event` 来通知有新输出。

  

---

  

## 3. 阶段二：IPC 传输到后端进程

  

请求通过 `AsyncMPClient` 经 ZMQ 发送到后端进程：

  

```python

# AsyncMPClient.add_request_async()

async def add_request_async(self, request: EngineCoreRequest) -> None:

    request.client_index = self.client_index

    await self._send_input(EngineCoreRequestType.ADD, request)

```

  

**传输过程**：

1. `self.encoder.encode(request)` — msgpack 序列化 `EngineCoreRequest`

2. `self.input_socket.send_multipart(msg, copy=False)` — ZMQ 零拷贝发送

3. 这是一个 `await` 操作，不阻塞 asyncio 事件循环

  

**后端的 `input_thread`（守护线程）接收并预处理**：

  

```1239:1265:vllm/v1/engine/core.py

            while True:

                for input_socket, _ in poller.poll():

                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)

                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    if request_type == EngineCoreRequestType.ADD:

                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)

                        try:

                            request = self.preprocess_add_request(req)

                        except Exception:

                            self._handle_request_preproc_error(req)

                            continue

                    # ...

                    self.input_queue.put_nowait((request_type, request))

```

  

**`preprocess_add_request` 在 input_thread 中执行**：

  

```682:704:vllm/v1/engine/core.py

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:

        """Preprocess the request.

        This function could be directly used in input processing thread to allow

        request initialization running in parallel with Model forward

        """

        # ... multimodal feature caching ...

        req = Request.from_engine_core_request(request, self.request_block_hasher)

        if req.use_structured_output:

            self.structured_output_manager.grammar_init(req)

        return req, request.current_wave

```

  

**关键设计**：`preprocess_add_request` 在 input_thread 中执行，包括 `Request` 对象构建、block hash 计算（prefix caching）、grammar 编译等 CPU 工作。由于 input_thread 在 IO 操作时释放 GIL，这些工作可以与主线程的 GPU 计算并行。

  

---

  

## 4. 阶段三：后端 EngineCore 忙循环处理请求

  

后端进程的主循环：

  

```1041:1051:vllm/v1/engine/core.py

    def run_busy_loop(self):

        """Core busy loop of the EngineCore."""

        while True:

            # 1) Poll the input queue until there is work to do.

            self._process_input_queue()

            # 2) Step the engine core and return the outputs.

            self._process_engine_step()

            # 3) Run any per-step hooks.

            self._process_per_step_hooks()

```

  

**`_process_input_queue` 从线程安全队列取出请求**：

  

```1053:1079:vllm/v1/engine/core.py

    def _process_input_queue(self):

        waited = False

        while (

            not self.engines_running

            and not self.scheduler.has_requests()

            and not self.batch_queue

            and not self.per_step_hooks

        ):

            if self.input_queue.empty():

                with self.aborts_queue.mutex:

                    self.aborts_queue.queue.clear()

                # ...

            req = self.input_queue.get()      # 阻塞等待，不浪费 CPU

            self._handle_client_request(*req)

        # ... drain 剩余请求 ...

        while not self.input_queue.empty():

            req = self.input_queue.get_nowait()

            self._handle_client_request(*req)

```

  

**请求被分发到 `add_request()`** → 加入 Scheduler 的 waiting 队列。

  

**`_process_engine_step` 调用 step 函数**：

  

```1081:1099:vllm/v1/engine/core.py

    def _process_engine_step(self) -> bool:

        outputs, model_executed = self.step_fn()  # ← step_with_batch_queue()

        for output in outputs.items() if outputs else ():

            self.output_queue.put_nowait(output)

        self.post_step(model_executed)

        if not model_executed and self.scheduler.has_unfinished_requests():

            time.sleep(0.001)

        return model_executed

```

  

**此时 `self.step_fn` = `self.step_with_batch_queue`**（因为 `async_scheduling=True`）：

  

```207:210:vllm/v1/engine/core.py

        self.step_fn = (

            self.step if self.batch_queue is None else self.step_with_batch_queue

        )

```

  

---

  

## 5. 阶段四：Scheduler 调度

  

### 5.1 `step_with_batch_queue()` —— 异步调度的核心

  

这是 **异步调度的关键函数**，通过深度为 2 的 batch_queue 实现调度与执行的流水线重叠：

  

```416:531:vllm/v1/engine/core.py

    def step_with_batch_queue(self):

        batch_queue = self.batch_queue

        assert batch_queue is not None

        assert len(batch_queue) < self.batch_queue_size  # max = 2

  

        model_executed = False

        deferred_scheduler_output = None

        if self.scheduler.has_requests():

            # ① 调度一个新 batch

            scheduler_output = self.scheduler.schedule()

            # ② 提交 execute_model（non_block=True → 不等 GPU 完成）

            exec_future = self.model_executor.execute_model(

                scheduler_output, non_block=True

            )

            if not self.is_ec_producer:

                model_executed = scheduler_output.total_num_scheduled_tokens > 0

  

            if self.is_pooling_model or not model_executed:

                future = cast(Future[ModelRunnerOutput], exec_future)

            else:

                if not scheduler_output.pending_structured_output_tokens:

                    # ③ 提交 sample_tokens（non_block=True）

                    grammar_output = self.scheduler.get_grammar_bitmask(

                        scheduler_output

                    )

                    future = self.model_executor.sample_tokens(

                        grammar_output, non_block=True

                    )

                else:

                    deferred_scheduler_output = scheduler_output

  

            if not deferred_scheduler_output:

                # ④ 推入 batch_queue

                batch_queue.appendleft((future, scheduler_output, exec_future))

                if (

                    model_executed

                    and len(batch_queue) < self.batch_queue_size  # 队列没满

                    and not batch_queue[-1][0].done()             # 最老的还没完成

                ):

                    # ⑤ 不阻塞！立即返回 None

                    return None, True

  

        # ⑥ 队列满了或没有更多请求 → 阻塞等待最老的 batch 完成

        future, scheduler_output, exec_model_fut = batch_queue.pop()

        model_output = future.result()  # 这里才阻塞

  

        # ⑦ 用完成的结果更新 scheduler

        self._process_aborts_queue()

        engine_core_outputs = self.scheduler.update_from_output(

            scheduler_output, model_output

        )

        return engine_core_outputs, model_executed

```

  

### 5.2 AsyncScheduler 的 "乐观占位" 策略

  

异步调度使用 `AsyncScheduler`（继承自 `Scheduler`），在 `schedule()` 完成后做**乐观占位**：

  

```18:35:vllm/v1/core/sched/async_scheduler.py

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:

        super()._update_after_schedule(scheduler_output)

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id in scheduler_output.num_scheduled_tokens:

            request = self.requests[req_id]

            if request.is_prefill_chunk:

                continue

            scheduler_output.pending_structured_output_tokens |= (

                request.use_structured_output and request.num_output_placeholders > 0

            )

            # 乐观假设：这一步会产出 1(target) + num_spec_tokens(draft) 个 token

            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))

            request.num_output_placeholders += 1 + cur_num_spec_tokens

            # 用占位符 [-1, -1] 填充 draft token ids

            request.spec_token_ids = self._spec_token_placeholders

```

  

**关键变量值**（对于首次 decode step，MTP=2）：

```

cur_num_spec_tokens = 2               # MTP 产生 2 个 draft tokens

request.num_output_placeholders += 3   # 1(target) + 2(draft)

request.spec_token_ids = [-1, -1]      # 占位符，真实值在 Worker 中计算

```

  

**为什么要"乐观占位"？** 因为 `step_with_batch_queue()` 中 schedule(N+1) 在 execute(N) 完成之前就执行了。此时 Scheduler 不知道 Step N 接受了多少 draft tokens，所以假设全部接受，后续通过 `update_from_output()` 修正。

  

---

  

## 6. 阶段五：Worker 执行 execute_model — 前处理

  

Scheduler 输出的 `SchedulerOutput` 被发送到 Worker，执行 `GPUModelRunner.execute_model()`。

  

### 6.1 输入同步保护

  

```3012:3025:vllm/v1/worker/gpu_model_runner.py

    @contextmanager

    def synchronize_input_prep(self):

        if self.prepare_inputs_event is None:

            yield

            return

        # 等待上一步的 non_blocking CPU→GPU 拷贝完成

        self.prepare_inputs_event.synchronize()

        try:

            yield

        finally:

            # 在当前 compute stream 上记录新 event

            self.prepare_inputs_event.record()

```

  

**作用**：防止下一步的 `_prepare_inputs` 在 CPU pinned memory 拷贝完成前覆写这些 tensor。由于 GPU forward 时间远大于 H2D 拷贝时间，这个 `synchronize()` 通常几乎零成本。

  

### 6.2 `_update_states` — 更新 persistent batch

  

```3344:3345:vllm/v1/worker/gpu_model_runner.py

            self._update_states(scheduler_output)

```

  

对于 **首次 Prefill**（第一个 step），此函数将新请求加入 `input_batch`：

  

```878:925:vllm/v1/worker/gpu_model_runner.py

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        # 移除已完成的请求

        for req_id in scheduler_output.finished_req_ids:

            self.requests.pop(req_id, None)

        for req_id in scheduler_output.finished_req_ids:

            self.input_batch.remove_request(req_id)

        # ... 添加新请求到 persistent batch ...

```

  

对于 **后续 Decode step（含 MTP）**，关键操作是修正 `num_computed_tokens`：

  

```995:1028:vllm/v1/worker/gpu_model_runner.py

        # 等待 valid_sampled_tokens_count 的 D2H 拷贝完成

        valid_sampled_token_count = self._get_valid_sampled_token_count()

  

        for i, req_id in enumerate(req_data.req_ids):

            req_state = self.requests[req_id]

            num_computed_tokens = req_data.num_computed_tokens[i]

            # ...

            if req_state.prev_num_draft_len and self.use_async_scheduling:

                if req_index is None:

                    req_state.prev_num_draft_len = 0

                else:

                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]

                    num_accepted = valid_sampled_token_count[prev_req_index] - 1

                    num_rejected = req_state.prev_num_draft_len - num_accepted

                    num_computed_tokens -= num_rejected

                    req_state.output_token_ids.extend([-1] * num_accepted)

```

  

**关键变量值示例**（Step 3，MTP=2）：

```

# Scheduler 乐观假设全部接受，给的 num_computed_tokens = 7(prompt) + 3(step1全接受) + 3(step2全接受) = 13

# 但实际 step2 只接受了 1 个 draft token

num_computed_tokens = 13  # Scheduler 给的值

prev_num_draft_len = 2    # 上一步安排了 2 个 draft tokens

valid_sampled_token_count[prev_req_index] = 2  # 实际有效数 = 1(target) + 1(accepted draft)

num_accepted = 2 - 1 = 1

num_rejected = 2 - 1 = 1

num_computed_tokens -= 1  # 修正为 12

```

  

### 6.3 `_prepare_inputs` — CPU 密集型前处理

  

```1458:1478:vllm/v1/worker/gpu_model_runner.py

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        num_reqs = self.input_batch.num_reqs

  

        # 优化：先启动 block table 的 H2D 异步拷贝，与后续 CPU numpy 运算并行

        self.input_batch.block_table.commit_block_table(num_reqs)

```

  

**核心 numpy 计算**：

  

```1480:1522:vllm/v1/worker/gpu_model_runner.py

        # 计算 request indices: [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

  

        # 计算 positions (numpy)

        positions_np = self.positions.np[:total_num_scheduled_tokens]

        np.add(

            self.input_batch.num_computed_tokens_cpu[req_indices],

            arange,

            out=positions_np,

        )

  

        # 计算 token indices + torch.index_select 获取 input_ids

        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]

        torch.index_select(

            self.input_batch.token_ids_cpu_tensor.flatten(),

            0,

            token_indices_tensor,

            out=self.input_ids.cpu[:total_num_scheduled_tokens],

        )

```

  

**slot_mapping 计算 + non_blocking H2D 拷贝**：

  

```1571:1576:vllm/v1/worker/gpu_model_runner.py

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)

        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

  

        self.query_start_loc.np[0] = 0

        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens

```

  

所有 `copy_to_gpu()` 调用内部使用 `non_blocking=True`：

  

```130:133:vllm/v1/utils.py

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:

        if n is None:

            return self.gpu.copy_(self.cpu, non_blocking=True)

        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

```

  

### 6.4 `_prepare_input_ids` — 异步调度的 GPU→GPU scatter

  

在 Decode 阶段（非首次 step），sampled tokens 和 draft tokens 都在 GPU 上（`prev_sampled_token_ids`），通过 GPU 内 scatter 拼装 input_ids：

  

```1292:1310:vllm/v1/worker/gpu_model_runner.py

    def _prepare_input_ids(self, scheduler_output, total_num_scheduled_tokens, cu_num_tokens):

        if self.input_batch.prev_sampled_token_ids is None:

            # Prefill 路径：普通 CPU→GPU 拷贝

            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)

            return

        # Decode 异步路径：GPU→GPU scatter

```

  

**快速路径**（batch 不变时）：

  

```1364:1375:vllm/v1/worker/gpu_model_runner.py

        if indices_match and max_flattened_index == (num_commmon_tokens - 1):

            # batch 不变，直接 slice copy

            self.input_ids.gpu[:num_commmon_tokens].copy_(

                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0],

                non_blocking=True,

            )

            return

```

  

**一般路径**（scatter sampled tokens + draft tokens）：

  

```1383:1411:vllm/v1/worker/gpu_model_runner.py

        # scatter sampled tokens 到对应位置

        self.input_ids.gpu.scatter_(

            dim=0,

            index=sampled_tokens_index_tensor,

            src=self.input_batch.prev_sampled_token_ids[

                prev_common_req_indices_tensor, 0

            ],

        )

        # scatter draft tokens 到对应位置

        self.input_ids.gpu.scatter_(

            dim=0,

            index=draft_tokens_index_tensor,

            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],

        )

```

  

**关键变量值示例**（Decode step，MTP=2，1 个请求）：

```

# 假设上一步 sampled token = [42], draft tokens = [100, 200]

# 对于当前 decode step，请求被安排 1(target) + 2(draft) = 3 个 token

# input_ids 在 GPU 上的构成：

#   input_ids[0] = 42   ← prev_sampled_token_ids (GPU→GPU copy)

#   input_ids[1] = 100  ← draft_token_ids[0] (GPU→GPU scatter)

#   input_ids[2] = 200  ← draft_token_ids[1] (GPU→GPU scatter)

# 完全不经过 CPU，零 D2H/H2D！

```

  

### 6.5 `_build_attention_metadata` — 构建 MLA Attention Metadata

  

```1483:1497:vllm/v1/worker/gpu_model_runner.py

            attn_metadata, spec_decode_common_attn_metadata = (

                self._build_attention_metadata(

                    num_tokens=num_tokens_unpadded,

                    num_tokens_padded=num_tokens_padded if pad_attn else None,

                    # ...

                )

            )

```

  

`CommonAttentionMetadata` 同时持有 CPU 和 GPU tensor 引用：

  

```1748:1763:vllm/v1/worker/gpu_model_runner.py

        cm_base = CommonAttentionMetadata(

            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],

            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],

            seq_lens=self.seq_lens.gpu[:num_reqs_padded],

            _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],

            num_reqs=num_reqs_padded,

            num_actual_tokens=num_tokens_padded,

            max_query_len=max_query_len,

            max_seq_len=max_seq_len,

            block_table_tensor=block_table_gid_0,

            slot_mapping=slot_mapping_gid_0,

            causal=True,

        )

```

  

**关键点**：`.gpu` 引用指向之前 `copy_to_gpu(non_blocking=True)` 的目标 tensor。metadata 构建只创建引用，不需要等待 H2D 拷贝完成 — 真正需要数据的 GPU kernel 在同一个 compute stream 上执行，CUDA 的流内有序保证了 H2D 拷贝在 kernel 启动前完成。

  

---

  

## 7. 阶段六：GPU Forward — DeepSeek V3 模型推理

  

```3542:3548:vllm/v1/worker/gpu_model_runner.py

            model_output = self._model_forward(

                input_ids=input_ids,

                positions=positions,

                intermediate_tensors=intermediate_tensors,

                inputs_embeds=inputs_embeds,

                **model_kwargs,

            )

```

  

对于 DeepSeek V3，模型 forward 包含：

- **61 层 Transformer**，每层含：

  - **MLA (Multi-head Latent Attention)**：低秩分解的注意力机制

  - **MoE FFN**：混合专家前馈网络

- 最终输出 `hidden_states` 和 `logits`

  

**Logits 计算和状态保存**：

  

```3577:3622:vllm/v1/worker/gpu_model_runner.py

            sample_hidden_states = hidden_states[logits_indices]

            logits = self.model.compute_logits(sample_hidden_states)

  

        # 保存中间状态，供 sample_tokens 使用

        self.execute_model_state = ExecuteModelState(

            scheduler_output,

            logits,

            spec_decode_metadata,

            spec_decode_common_attn_metadata,

            hidden_states,

            sample_hidden_states,

            aux_hidden_states,

            ec_connector_output,

            cudagraph_stats,

            slot_mappings,

        )

        return None  # ← execute_model 返回 None，表示需要后续调用 sample_tokens

```

  

**关键变量值示例**（Prefill，7 个 prompt tokens）：

```

input_ids.shape = [7]       # 7 个 prompt tokens

positions = [0, 1, 2, 3, 4, 5, 6]

hidden_states.shape = [7, hidden_dim]

logits.shape = [1, vocab_size]  # 只对最后一个 token 计算 logits

logits_indices = [6]             # 最后一个 token 的索引

```

  

**关键变量值示例**（Decode + MTP=2，1 个请求，上一步安排了 1+2=3 个 token）：

```

input_ids.shape = [3]       # 1(sampled) + 2(draft)

positions = [7, 8, 9]       # 接续 prompt 的位置

hidden_states.shape = [3, hidden_dim]

logits.shape = [3, vocab_size]  # 对所有 3 个 token 计算 logits（用于验证 draft）

```

  

---

  

## 8. 阶段七：sample_tokens — 采样 + MTP 投机解码

  

`sample_tokens()` 是整个流程中最复杂的阶段，包含采样、rejection sampling、MTP drafter 推理和异步输出。

  

### 8.1 采样

  

```3669:3670:vllm/v1/worker/gpu_model_runner.py

        with record_function_or_nullcontext("gpu_model_runner: sample"):

            sampler_output = self._sample(logits, spec_decode_metadata)

```

  

**Prefill 首次采样**：

```

sampler_output.sampled_token_ids.shape = [1, 1]  # 1 个请求，1 个 token

# 假设采样得到 token_id = 42

sampler_output.sampled_token_ids = [[42]]

```

  

**Decode + MTP 验证（Rejection Sampling）**：

  

Rejection sampling 在 GPU 上通过 Triton kernel 执行：

  

```642:683:vllm/v1/sample/rejection_sampler.py

@triton.jit(do_not_specialize=["max_spec_len"])

def rejection_greedy_sample_kernel(

    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]

    cu_num_draft_tokens_ptr,

    draft_token_ids_ptr,

    target_argmax_ptr,

    bonus_token_ids_ptr,

    # ...

):

    req_idx = tl.program_id(0)

    # 全部在 GPU 上执行：逐个比对 draft token 和 target argmax

    for pos in range(num_draft_tokens):

        if not rejected:

            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)

            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)

            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,

                     target_argmax_id)

            if draft_token_id != target_argmax_id:

                rejected = True

```

  

**关键变量值示例**（Decode step，MTP=2，greedy sampling）：

```

# draft_tokens = [100, 200]

# target_argmax = [100, 300]（第 1 个 draft 接受，第 2 个拒绝）

# 输出：

sampler_output.sampled_token_ids = [[42, 100, -1]]

#                                    ^    ^    ^

#                                    |    |    └── rejected（标记为 -1）

#                                    |    └── accepted draft 1

#                                    └── target token（bonus token）

# shape = [1, 3]  (batch_size=1, max_spec_len+1=3)

```

  

### 8.2 MTP Drafter 推理

  

采样后，MTP drafter 使用 GPU 上的 sampled tokens 进行下一轮 draft 推理。

  

**关键代码路径选择**（DeepSeek V3 MTP → `use_eagle()` = True）：

  

```3705:3741:vllm/v1/worker/gpu_model_runner.py

        spec_config = self.speculative_config

        if spec_config is not None:

            # ...

            use_gpu_toks = (

                spec_config.use_eagle() or spec_config.uses_draft_model()

            ) and not spec_config.disable_padded_drafter_batch

            if use_gpu_toks:

                assert isinstance(self.drafter, EagleProposer | DraftModelProposer)

                sampled_token_ids = sampler_output.sampled_token_ids  # ← GPU tensor！

                if input_fits_in_drafter:

                    # 直接运行 drafter

                    propose_draft_token_ids(sampled_token_ids)

                elif self.valid_sampled_token_count_event is not None:

                    # batch 太大无法 fit 进 drafter，但仍在 GPU 上计算 next_token_ids

                    next_token_ids, valid_sampled_tokens_count = (

                        self.drafter.prepare_next_token_ids_padded(

                            spec_decode_common_attn_metadata,

                            sampled_token_ids,     # ← GPU tensor 直接传！

                            self.requests,

                            self.input_batch,

                            self.discard_request_mask.gpu,

                        )

                    )

                    self._copy_valid_sampled_token_count(

                        next_token_ids, valid_sampled_tokens_count

                    )

```

  

**`prepare_next_token_ids_padded` —— 关键的 Triton 后处理算子**：

  

```879:935:vllm/v1/spec_decode/eagle.py

    def prepare_next_token_ids_padded(self, ...):

        # ... 预计算 backup token ids ...

        eagle_prepare_next_token_padded_kernel[grid](

            sampled_token_ids,

            discard_request_mask,

            backup_tokens_gpu,

            next_token_ids,          # 输出：每个请求的下一个 token

            valid_sampled_tokens_count,  # 输出：每个请求有效 token 数

            # ...

        )

        return next_token_ids, valid_sampled_tokens_count

```

  

**Triton kernel 的工作**（在 GPU 上完成，零 D2H 同步）：

  

```57:117:vllm/v1/spec_decode/utils.py

@triton.jit

def eagle_prepare_next_token_padded_kernel(

    sampled_token_ids_ptr,          # [num_reqs, num_sampled_tokens_per_req]

    discard_request_mask_ptr,       # [num_reqs]

    backup_next_token_ids_ptr,      # [num_reqs]

    next_token_ids_ptr,             # [num_reqs] (输出)

    valid_sampled_tokens_count_ptr, # [num_reqs] (输出)

    # ...

):

    req_idx = tl.program_id(axis=0)

    # 从 GPU HBM 读取 sampled tokens

    token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

    # 在 GPU 上计算有效 token 数

    is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask

    valid_count = tl.sum(is_valid_mask)

    # 找到最后一个合法 token

    last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

    last_valid_token = tl.sum(tl.where(token_offs == last_valid_index, token_ids, 0))

    # 写回 GPU

    tl.store(next_token_ids_ptr + req_idx, last_valid_token)

    tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)

```

  

**关键变量值示例**（接上例）：

```

# sampled_token_ids = [[42, 100, -1]]  (GPU tensor)

# → valid_count = 2  (42 和 100 有效)

# → next_token_ids = 100  (最后一个有效 token，作为 drafter 的输入)

# → valid_sampled_tokens_count = [2]  (GPU tensor)

```

  

### 8.3 valid_sampled_token_count 的异步 D2H 拷贝

  

```3908:3925:vllm/v1/worker/gpu_model_runner.py

    def _copy_valid_sampled_token_count(

        self, next_token_ids, valid_sampled_tokens_count

    ):

        default_stream = torch.cuda.current_stream()

        # 在独立 CUDA stream 上异步拷贝

        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):

            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)

            counts_cpu = self.valid_sampled_token_count_cpu

            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)

            self.valid_sampled_token_count_event.record()

  

        # 将 next_token_ids 设为下一步的 prev_sampled_token_ids

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

```

  

**这个 D2H 拷贝与 Drafter Forward 并行执行**，只拷贝 `batch_size` 个 int32（几十 bytes），极快完成。

  

### 8.4 MTP Drafter Forward

  

Drafter 使用 target model 的 hidden states + next_token_ids 在 GPU 上执行 MTP 模型的 forward，产出新的 draft tokens：

  

```python

# propose_draft_token_ids 内部调用 self.drafter.propose()

# 输入：

#   - target_hidden_states: 来自 target model forward 的 hidden states

#   - next_token_ids: 来自 Triton kernel 计算的下一个 token（GPU tensor）

#   - common_attn_metadata: attention 元数据

# 输出：

#   - draft_token_ids: [num_reqs, num_spec_tokens] 的 GPU tensor

#     例如 [[150, 250]]  （新的 2 个 draft tokens）

```

  

### 8.5 _bookkeeping_sync — CPU 侧状态更新

  

```2885:2961:vllm/v1/worker/gpu_model_runner.py

    def _bookkeeping_sync(self, scheduler_output, sampler_output, ...):

        # ...

        if not self.use_async_scheduling:

            # 非异步路径：立即 D2H + CPU 解析

            valid_sampled_token_ids = self._to_list(sampled_token_ids)  # 触发同步！

        else:

            # 异步路径：不解析 GPU tensor

            valid_sampled_token_ids = []  # ← 空！

            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()

            invalid_req_indices_set = set(invalid_req_indices)

  

            # 缓存 sampled tokens 在 GPU 上

            if self.input_batch.prev_sampled_token_ids is None:

                assert sampled_token_ids.shape[-1] == 1

                self.input_batch.prev_sampled_token_ids = sampled_token_ids

            self.input_batch.prev_req_id_to_index = {

                req_id: i

                for i, req_id in enumerate(self.input_batch.req_ids)

                if i not in invalid_req_indices_set

            }

```

  

**异步路径 CPU 侧填占位符 -1**：

  

```2969:2994:vllm/v1/worker/gpu_model_runner.py

        for req_idx in range(num_sampled_tokens):

            if self.use_async_scheduling:

                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None

            else:

                sampled_ids = valid_sampled_token_ids[req_idx]

            # ...

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids

            req_state.output_token_ids.extend(sampled_ids)  # [-1] 占位

```

  

---

  

## 9. 阶段八：异步输出拷贝 + 调度下一步

  

### 9.1 AsyncGPUModelRunnerOutput — 异步 D2H

  

`sample_tokens()` 的最后，创建 `AsyncGPUModelRunnerOutput`，在独立 CUDA stream 上启动异步 D2H：

  

```3793:3817:vllm/v1/worker/gpu_model_runner.py

        if not self.use_async_scheduling:

            return output         # ← 同步模式：直接返回

  

        async_output = AsyncGPUModelRunnerOutput(

            model_runner_output=output,

            sampled_token_ids=sampler_output.sampled_token_ids,

            logprobs_tensors=sampler_output.logprobs_tensors,

            invalid_req_indices=invalid_req_indices,

            async_output_copy_stream=self.async_output_copy_stream,

            vocab_size=self.input_batch.vocab_size,

        )

        return async_output       # ← 异步模式：立即返回，D2H 还没完成！

```

  

**`AsyncGPUModelRunnerOutput.__init__()` 内部**：

  

```225:237:vllm/v1/worker/gpu_model_runner.py

        default_stream = torch.cuda.current_stream()

        with torch.cuda.stream(async_output_copy_stream):

            async_output_copy_stream.wait_stream(default_stream)

            self.sampled_token_ids_cpu = self._sampled_token_ids.to(

                "cpu", non_blocking=True

            )

            self._logprobs_tensors_cpu = (

                self._logprobs_tensors.to_cpu_nonblocking()

                if self._logprobs_tensors

                else None

            )

            self.async_copy_ready_event.record()

```

  

**关键**：`wait_stream(default_stream)` 确保 copy stream 等到 compute stream 上的采样完成后再开始拷贝，但**不阻塞 compute stream**。

  

### 9.2 `get_output()` — 延迟同步

  

当 EngineCore 的 `update_from_output()` 真正需要读取 token ids 时，调用 `get_output()`：

  

```239:268:vllm/v1/worker/gpu_model_runner.py

    def get_output(self) -> ModelRunnerOutput:

        max_gen_len = self.sampled_token_ids_cpu.shape[-1]

        self.async_copy_ready_event.synchronize()  # ← 此时 D2H 大概率已完成

  

        if max_gen_len == 1:

            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()

            for i in self._invalid_req_indices:

                valid_sampled_token_ids[i].clear()

        else:

            # MTP/spec decode 场景：在 CPU 上执行 parse_output

            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(

                self.sampled_token_ids_cpu,

                self.vocab_size,

                self._invalid_req_indices,

            )

        output = self._model_runner_output

        output.sampled_token_ids = valid_sampled_token_ids

        return output

```

  

**关键变量值示例**（get_output 返回的结果）：

```

# 对于上面的例子（draft 1 接受，draft 2 拒绝）：

output.sampled_token_ids = [[42, 100]]  # CPU list[list[int]]

# 42 = target token, 100 = accepted draft token

# -1 的 rejected token 已被 parse_output 过滤掉

```

  

### 9.3 `post_step` — 异步调度下跳过 draft token 传回

  

```406:414:vllm/v1/engine/core.py

    def post_step(self, model_executed: bool) -> None:

        # 异步调度下，draft token ids 直接在 Worker 内部处理

        # 不需要传回 Scheduler

        if not self.async_scheduling and self.use_spec_decode and model_executed:

            draft_token_ids = self.model_executor.take_draft_token_ids()

            if draft_token_ids is not None:

                self.scheduler.update_draft_token_ids(draft_token_ids)

```

  

**异步调度直接跳过 `take_draft_token_ids()`**，避免了 draft token ids 的 D2H + CPU 处理 + 传递开销。

  

### 9.4 batch_queue 的流水线效果

  

以稳态 decode 为例，两步迭代的调用序列：

  

```

run_busy_loop 第 K 轮:

  step_with_batch_queue():

    schedule(N)                          ← CPU 工作

    execute_model(N, non_block=True)     ← 提交给 Worker

    sample_tokens(N, non_block=True)     ← 提交给 Worker

    batch_queue: [future_N] (size=1 < 2, return None)  ← 不阻塞！

  

run_busy_loop 第 K+1 轮:

  step_with_batch_queue():

    schedule(N+1)                        ← 此时 Worker 还在执行 step N！

    execute_model(N+1, non_block=True)

    sample_tokens(N+1, non_block=True)

    batch_queue: [future_N+1, future_N] (size=2, 满了)

    pop(future_N), 等 step N 完成

    update_from_output(N)                ← 用 N 的结果修正 scheduler

    return outputs_N

```

  

---

  

## 10. 阶段九：Scheduler update_from_output — 状态结算

  

`update_from_output()` 是每一步完成后的**集中式状态结算**：

  

```1256:1377:vllm/v1/core/sched/scheduler.py

    def update_from_output(self, scheduler_output, model_runner_output):

        sampled_token_ids = model_runner_output.sampled_token_ids

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

  

        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():

            request = self.requests.get(req_id)

            if request is None or request.is_finished():

                continue

  

            req_index = model_runner_output.req_id_to_index[req_id]

            generated_token_ids = sampled_token_ids[req_index]

```

  

### 10.1 MTP Rejection 修正

  

```1319:1343:vllm/v1/core/sched/scheduler.py

            scheduled_spec_token_ids = (

                scheduler_output.scheduled_spec_decode_tokens.get(req_id)

            )

            if scheduled_spec_token_ids and generated_token_ids:

                num_draft_tokens = len(scheduled_spec_token_ids)

                num_accepted = len(generated_token_ids) - 1

                num_rejected = num_draft_tokens - num_accepted

                if request.num_computed_tokens > 0:

                    request.num_computed_tokens -= num_rejected

                if request.num_output_placeholders > 0:

                    request.num_output_placeholders -= num_rejected

```

  

**关键变量值示例**（MTP=2，draft 1 接受，draft 2 拒绝）：

```

scheduled_spec_token_ids = [-1, -1]    # 占位符（异步调度下）

generated_token_ids = [42, 100]        # target + 1 accepted draft

num_draft_tokens = 2

num_accepted = 2 - 1 = 1              # 除去 target 的数量

num_rejected = 2 - 1 = 1

request.num_computed_tokens -= 1       # 修正乐观估计

request.num_output_placeholders -= 1   # 修正乐观占位

```

  

### 10.2 AsyncScheduler 的占位修正

  

```37:60:vllm/v1/core/sched/async_scheduler.py

    def _update_request_with_output(self, request, new_token_ids):

        # ...

        new_token_ids, stopped = super()._update_request_with_output(

            request, new_token_ids

        )

        # 扣减真实消耗的占位符

        request.num_output_placeholders -= len(new_token_ids)

        assert request.num_output_placeholders >= 0

        # ...

        return new_token_ids, stopped

```

  

### 10.3 Stop 检查

  

```1352:1377:vllm/v1/core/sched/scheduler.py

            if new_token_ids:

                new_token_ids, stopped = self._update_request_with_output(

                    request, new_token_ids

                )

            # ...

            if stopped:

                finish_reason = request.get_finished_reason()

                finished = self._handle_stopped_request(request)

                if finished:

                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:

                    stopped_running_reqs.add(request)

```

  

`_update_request_with_output` 内部逐 token 检查 stop 条件（EOS、max_tokens、stop strings）：

  

如果在 MTP 接受的中间某个 token 触发了 stop，后续 token 会被截断。

  

### 10.4 构建 EngineCoreOutput

  

```1410:1428:vllm/v1/core/sched/scheduler.py

            if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:

                outputs[request.client_index].append(

                    EngineCoreOutput(

                        request_id=req_id,

                        new_token_ids=new_token_ids,

                        finish_reason=finish_reason,

                        new_logprobs=new_logprobs,

                        # ...

                    )

                )

```

  

**关键变量值示例**：

```

EngineCoreOutput(

    request_id="req-001",

    new_token_ids=[42, 100],  # target + 1 accepted draft

    finish_reason=None,        # 未结束

)

```

  

---

  

## 11. 阶段十：前端输出处理与交付用户

  

### 11.1 后端 output_thread 序列化 + ZMQ 发送

  

`_process_engine_step` 将结果放入 `output_queue`：

  

```1081:1090:vllm/v1/engine/core.py

    def _process_engine_step(self) -> bool:

        outputs, model_executed = self.step_fn()

        for output in outputs.items() if outputs else ():

            self.output_queue.put_nowait(output)

        self.post_step(model_executed)

```

  

**output_thread（守护线程）** 从队列取出、序列化、发送：

  

```python

# process_output_sockets 线程（简化）

while True:

    output = self.output_queue.get()

    client_index, outputs = output

    buffers = encoder.encode_into(outputs, buffer)

    sockets[client_index].send_multipart(buffers, copy=False, track=True)

```

  

使用 `copy=False`（零拷贝）+ buffer 复用，最大限度减少序列化和内存拷贝开销。

  

### 11.2 前端 `output_handler` asyncio Task

  

```660:711:vllm/v1/engine/async_llm.py

        async def output_handler():

            try:

                while True:

                    # 1) 从 EngineCore 拉取输出（await，不阻塞事件循环）

                    outputs = await engine_core.get_output_async()

                    num_outputs = len(outputs.outputs)

  

                    engine_core_outputs = outputs.outputs

                    for start in range(0, num_outputs, chunk_size):

                        end = start + chunk_size

                        outputs_slice = engine_core_outputs[start:end]

                        # 2) 处理输出（detokenization 等）

                        processed_outputs = output_processor.process_outputs(

                            outputs_slice, outputs.timestamp, iteration_stats

                        )

                        # 让出事件循环，防止饥饿

                        if end < num_outputs:

                            await asyncio.sleep(0)

                        # 3) 如果有因 stop string 终止的请求，发起 abort

                        if processed_outputs.reqs_to_abort:

                            await engine_core.abort_requests_async(

                                processed_outputs.reqs_to_abort

                            )

```

  

**`output_processor.process_outputs()`** 执行 detokenization，将 token ids 转为文本，并通过 `RequestOutputCollector.put()` 推送到每个请求的队列。

  

### 11.3 结果交付给用户

  

```584:594:vllm/v1/engine/async_llm.py

            finished = False

            while not finished:

                # 先尝试无等待获取（避免 Task 切换开销）

                out = q.get_nowait() or await q.get()

                assert isinstance(out, RequestOutput)

                finished = out.finished

                if out is not STREAM_FINISHED:

                    yield out

```

  

**`q.get_nowait() or await q.get()`** 的优化：在高吞吐场景下，队列中很可能已有数据，`get_nowait()` 直接返回，避免了不必要的协程挂起/恢复。

  

**关键变量值示例**（streaming 模式，收到 [42, 100]）：

```

RequestOutput(

    request_id="req-001",

    outputs=[

        CompletionOutput(

            text=" world",       # detokenized from [42, 100]

            token_ids=[42, 100],

            finish_reason=None,

        )

    ],

    finished=False,

)

```

  

---

  

## 12. Decode 稳态：Step N+1 的流水线重叠

  

在稳态 decode 阶段，以 **MultiprocExecutor** 为例，两步迭代的时间线：

  

```

                    Step N                                     Step N+1

                    ──────                                     ────────

  

EngineCore:  [schedule(N)] [submit(N)]  [schedule(N+1)] [submit(N+1)] ─── [等future_N] [update(N)]

                                │                │                              ↑

                                ▼                ▼                              │

Worker:                  [_update_states(N)]                            [_update_states(N+1)]

                         │ valid_count.sync()                           │ valid_count.sync()

                         │ ← 几乎零成本                                  │ ← 修正 num_computed

                         [_prepare_inputs(N)]                           [_prepare_inputs(N+1)]

                         │ block_table H2D(async)                       │ input_ids: GPU scatter!

                         │ positions(numpy)                             │ (prev_sampled + draft → GPU→GPU)

                         │ slot_mapping(numpy)                          │

                         [_build_attn_metadata(N)]                      [_build_attn_metadata(N+1)]

                         [═══ DeepSeek V3 Forward(N) ═══]              [═══ Forward(N+1) ═══]

                         [═══ Sample(N) + Rejection ═══]               [═══ Sample(N+1) ═══]

                         [═══ MTP Drafter Forward(N) ═══]              [═══ MTP Drafter(N+1) ═══]

                         │ sampled_token_ids(N) GPU tensor               │

                         │    ↓ 直接传给 drafter，不经 CPU                │

                         │ → prev_sampled_token_ids = next_token_ids     │

  

GPU copy stream:                    [D2H: valid_count(N)]

                                    [D2H: sampled_tokens(N)]

                                          ↑ 与 schedule(N+1) 并行

```

  

**关键流水线重叠**：

1. **schedule(N+1) 与 Worker 执行 step N 并行**（MultiprocExecutor 下跨进程）

2. **D2H 输出拷贝与 schedule(N+1) 并行**（独立 CUDA stream）

3. **Step N+1 的 prepare_inputs 中 GPU scatter 避免了 D2H+H2D 往返**

4. **block_table H2D 与 numpy 运算并行**（DMA + CPU 并行）

5. **grammar_bitmask CPU 计算与 GPU forward 并行**

  

---

  

## 13. 异步调度的核心收益总结

  

### 被隐藏/消除的开销

  

| # | 被隐藏的开销 | 机制 | 代码位置 |

|---|---|---|---|

| **1** | schedule(N+1) 的 CPU 调度时间 | `step_with_batch_queue()` 深度 2 流水线 | `core.py:416-531` |

| **2** | sampled tokens D2H 同步等待 | `AsyncGPUModelRunnerOutput` + 独立 copy stream | `gpu_model_runner.py:225-237` |

| **3** | sampled tokens 的 H2D 传回 (input_ids) | **完全消除** — `prev_sampled_token_ids` 留 GPU，GPU→GPU scatter | `gpu_model_runner.py:1364-1411` |

| **4** | draft tokens 的 D2H + H2D 往返 | **完全消除** — `_draft_token_ids` 留 GPU | `gpu_model_runner.py:3864-3873` |

| **5** | Rejection sampling 后处理 (count/next_token) | Triton kernel 在 GPU 上完成 | `spec_decode/utils.py:57-117` |

| **6** | valid_sampled_token_count D2H | 独立 stream 异步拷贝，延迟到下一步才 sync | `gpu_model_runner.py:3908-3937` |

| **7** | Scheduler 处理 draft token ids 的 CPU 往返 | **完全消除** — Worker 内部处理，`post_step()` 跳过 | `core.py:406-414` |

| **8** | prepare_inputs 中的 H2D 拷贝 | `non_blocking=True` + `prepare_inputs_event` | `gpu_model_runner.py:3012-3025` |

| **9** | 后端反序列化 + Request 构建 | input_thread 在 GPU 计算时释放 GIL 并行 | `core.py:682-704` |

| **10** | 前端 detokenization + stats | 多进程隔离 + asyncio 分块处理 | `async_llm.py:660-711` |

  

### Triton 的必要性

  

在 MTP/spec decode 场景下，Triton kernel 不是因为"算得更快"才使用，而是因为：

  

| | 能避免 CPU-GPU 同步？ | Kernel Launch 次数 | HBM 访问 |

|---|---|---|---|

| **Triton** | 能（全 GPU 执行） | 1 次 | 1 读 1 写 |

| **PyTorch ops（不调 .cpu()）** | 同样能 | ~10 次 (~30-50μs) | 多次重复读写 |

| **PyTorch ops（调 .cpu()）** | 不能（全局 stream 同步） | N/A | N/A |

  

核心价值：**Triton 将多步 GPU 计算融合为单 kernel，消除 kernel launch 开销和中间 HBM 读写。**

  

### 单请求延迟收益估算

  

对于 DeepSeek V3 decode（GPU forward ~20-30ms），每步约节省 0.5-1.5ms 的 CPU 间隙：

- schedule CPU 时间：~0.1-0.5ms

- D2H 同步等待：~0.05-0.2ms（消除）

- H2D sampled tokens：~0.01ms（消除）

- draft tokens D2H+H2D：~0.02ms（消除）

- Triton vs PyTorch ops：~0.03-0.04ms

  

总计约 **3%-10% 的 TPOT 提升**。