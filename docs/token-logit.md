# TokenLogit 使用示例与限制

`TokenLogitAdapter` 将每个候选标签编码为一个 token，`VLLMHTTPBackend` 向兼容 OpenAI 的 `/v1/completions` 发送 prompt，并从 `choices[index].logprobs.top_logprobs[0]` 按候选 `token_text` 严格读取有限 logprob。

```python
adapter = TokenLogitAdapter(tokenizer.encode)
question = adapter.compile("route", "choice", prompt, ["yes", "no"])
backend = VLLMHTTPBackend(VLLMHTTPOptions("http://127.0.0.1:8000", model="my-model"))
logits = asyncio.run(backend.execute([question]))
answer = adapter.decode(question, [logits[0][key] for key in question.candidate_ids])
```

支持一个请求中的多个 prompt、有限并发 `execute_many`、超时和任务取消。取消 async 任务不会取消标准库 `urllib` 已经运行的网络线程；本实现不声称 `to_thread` 能中止该线程。

限制：候选文本必须与服务返回的 `top_logprobs[0]` 键完全匹配，不进行空白、规范化或 token-ID 推测；每个候选必须恰好编码为不同的整数 token，响应必须包含完整且不重复的 choice 索引及有限数值。`noul` 严格要求 `true`/`false` 两个候选，score legend 输出字符串。

本阶段尚未注册 `Engine`/`ModelFamily`，也未接入服务生命周期或真实模型认证；该示例只覆盖独立的适配器与 HTTP 后端。
