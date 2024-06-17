import gc
import time
import uuid
from threading import Thread
from types import MethodType
from typing import (
    Iterable,
    Dict,
    Any,
)

import torch
from transformers import (
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from api.generation.qwen import check_is_qwen
from api.generation.utils import (
    prepare_logits_processor,
    is_partial_stop,
    apply_stopping_strings,
)


@torch.inference_mode()
def generate_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any],
):
    # 读取参数
    input_ids = params.get("inputs")  # 获取输入的token IDs
    prompt = params.get("prompt")  # 获取提示文本
    model_name = params.get("model", "llm")  # 获取模型名称，默认为"llm"
    temperature = float(params.get("temperature", 1.0))  # 获取温度参数，默认为1.0
    repetition_penalty = float(params.get("repetition_penalty", 1.0))  # 获取重复惩罚参数，默认为1.0
    top_p = float(params.get("top_p", 1.0))  # 获取top-p参数，默认为1.0
    top_k = int(params.get("top_k", -1))  # 获取top-k参数，默认为-1（表示禁用）
    max_new_tokens = int(params.get("max_tokens", 256))  # 获取最大生成token数，默认为256
    logprobs = params.get("logprobs")  # 获取是否返回log概率的标志
    echo = bool(params.get("echo", True))  # 获取是否回显输入的标志，默认为True
    stop_str = params.get("stop")  # 获取停止生成的标志字符串

    stop_token_ids = params.get("stop_token_ids") or []  # 获取停止token的ID列表，默认为空列表
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)  # 将EOS token的ID添加到停止token的ID列表中

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )  # 准备处理logits的处理器

    output_ids = list(input_ids)  # 输出token IDs初始化为输入token IDs的副本
    input_echo_len = len(input_ids)  # 计算输入的token数

    device = model.device  # 获取模型所在的设备
    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]  # 编码器的输出
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )  # 开始生成token的ID
    else:
        start_ids = torch.as_tensor([input_ids], device=device)  # 开始生成token的ID

    past_key_values, sent_interrupt = None, False  # 初始化过去的key值和中断标志
    token_logprobs = [None]  # 第一个token没有log概率

    completion_id: str = f"cmpl-{str(uuid.uuid4())}"  # 生成完成标识符
    created: int = int(time.time())  # 记录生成的时间戳
    previous_text = ""  # 初始化之前生成的文本
    for i in range(max_new_tokens):
        if i == 0:  # 第一次迭代，预先填充
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )  # 使用解码器进行解码
                logits = model.lm_head(out[0])  # LM头的logits
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits  # 输出的logits
            past_key_values = out.past_key_values  # 更新过去的key值

            if logprobs is not None:
                # 预先计算prompt的log概率
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                        shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])

        else:  # 解码阶段
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [output_ids if sent_interrupt else [token]], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=None if sent_interrupt else past_key_values,
                )  # 使用解码器进行解码
                sent_interrupt = False

                logits = model.lm_head(out[0])  # LM头的logits
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [output_ids if sent_interrupt else [token]], device=device
                    ),
                    use_cache=True,
                    past_key_values=None if sent_interrupt else past_key_values,
                )  # 模型的输出
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values  # 更新过去的key值

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # 切换到CPU，避免mps后端中的一些bug。
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # 贪婪解码
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]

        token = tokens[0]
        output_ids.append(token)

        if logprobs is not None:
            # 由于logprobs基于原始logits，因此不能使用last_token_logits。
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # 生成输出tokens
        if i % 2 == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len(prompt)
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=False if check_is_qwen(model) else True,  # 修复qwen反应的问题
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs if echo else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}] * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # 计算文本偏移量
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            partially_stopped, finish_reason = False, None
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            if each_stop == "Observation:":
                                finish_reason = "function_call"
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # 防止生成部分停止序列
            if (not partially_stopped) and output and output[-1] != "�":
                delta_text = output[len(previous_text):]
                previous_text = output

                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "delta": delta_text,
                    "text": output,
                    "logprobs": ret_logprobs,
                    "finish_reason": finish_reason,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                }

        if stopped:
            break

    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": output,
        "logprobs": ret_logprobs,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
    }

    # 清理资源
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_v2(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any],
):
    input_ids = params.get("inputs")  # 获取输入的token ids序列
    functions = params.get("functions")  # 是否包含特定功能调用标志
    model_name = params.get("model", "llm")  # 模型名称，默认为"llm"
    temperature = float(params.get("temperature", 1.0))  # 温度参数，控制生成文本的多样性，默认为1.0
    repetition_penalty = float(params.get("repetition_penalty", 1.0))  # 重复惩罚参数，控制生成文本中重复词汇的惩罚力度，默认为1.0
    top_p = float(params.get("top_p", 1.0))  # top-p采样参数，控制生成文本的多样性，默认为1.0
    top_k = int(params.get("top_k", 40))  # top-k采样参数，控制生成文本的多样性，默认为40
    max_new_tokens = int(params.get("max_tokens", 256))  # 最大生成token数，默认为256

    stop_token_ids = params.get("stop_token_ids") or []  # 停止token ids列表，用于生成文本的终止条件
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    stop_strings = params.get("stop", [])  # 停止字符串列表，用于生成文本的终止条件

    input_echo_len = len(input_ids)  # 输入文本的长度（token数）
    device = model.device  # 模型所在设备（如CPU或GPU）
    generation_kwargs = dict(
        input_ids=torch.tensor([input_ids], device=device),  # 输入的token ids张量
        do_sample=True,  # 是否进行采样生成文本
        temperature=temperature,  # 温度参数，控制生成文本的多样性
        top_p=top_p,  # top-p采样参数，控制生成文本的多样性
        top_k=top_k,  # top-k采样参数，控制生成文本的多样性
        max_new_tokens=max_new_tokens,  # 最大生成token数
        repetition_penalty=repetition_penalty,  # 重复惩罚参数，控制生成文本中重复词汇的惩罚力度
        pad_token_id=tokenizer.pad_token_id,  # pad token的id，用于生成文本时的填充
    )  # 生成文本参数字典

    # 如果温度小于等于1e-5，则关闭采样生成文本
    if temperature <= 1e-5:
        generation_kwargs["do_sample"] = False
        generation_kwargs.pop("top_k")

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )  # 创建文本迭代流对象，用于异步生成文本

    generation_kwargs["streamer"] = streamer  # 将文本迭代流对象添加到生成参数中

    # 如果模型没有GenerationMixin方法，则将其绑定到PreTrainedModel的generate方法上
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # 创建并启动异步生成文本的线程
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text, func_call_found = "", False  # 初始化生成的文本和是否发现功能调用的标志
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"  # 创建生成文本的唯一标识
    created: int = int(time.time())  # 生成文本的时间戳
    previous_text = ""  # 前一次生成的文本

    # 遍历异步生成文本流
    for i, new_text in enumerate(streamer):
        generated_text += new_text  # 将新生成的文本追加到已生成文本中
        if functions:
            _, func_call_found = apply_stopping_strings(generated_text, ["Observation:"])  # 检查是否发现特定功能调用
        generated_text, stop_found = apply_stopping_strings(generated_text, stop_strings)  # 检查是否满足停止条件

        if generated_text and generated_text[-1] != "�":  # 如果生成的文本非空且末尾不是特殊字符
            delta_text = generated_text[len(previous_text):]  # 计算增量文本
            previous_text = generated_text  # 更新前一次生成的文本

            yield {
                "id": completion_id,  # 文本生成的唯一标识
                "object": "text_completion",  # 对象类型为文本生成
                "created": created,  # 生成文本的时间戳
                "model": model_name,  # 使用的模型名称
                "delta": delta_text,  # 增量文本
                "text": generated_text,  # 完整生成的文本
                "logprobs": None,  # 日志概率，这里为空
                "finish_reason": "function_call" if func_call_found else None,  # 完成原因，如果发现功能调用则为"function_call"
                "usage": {
                    "prompt_tokens": input_echo_len,  # 输入文本的token数
                    "completion_tokens": i,  # 当前已生成的token数
                    "total_tokens": input_echo_len + i,  # 总共生成的token数
                },
            }

        if stop_found:  # 如果满足停止条件
            break  # 终止生成文本的循环

    yield {
        "id": completion_id,  # 文本生成的唯一标识
        "object": "text_completion",  # 对象类型为文本生成
        "created": created,  # 生成文本的时间戳
        "model": model_name,  # 使用的模型名称
        "delta": "",  # 增量文本为空字符串，表示这是最终输出的文本
        "text": generated_text,  # 完整生成的文本
        "logprobs": None,  # 日志概率为空，表示不记录生成文本的概率信息
        "finish_reason": "stop",  # 完成原因为"stop"，表示生成文本的停止条件已满足
        "usage": {
            "prompt_tokens": input_echo_len,  # 输入文本的token数
            "completion_tokens": i,  # 当前已生成的token数
            "total_tokens": input_echo_len + i,  # 总共生成的token数
        },
    }
