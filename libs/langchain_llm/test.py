from langchain_llm import HuggingFaceLLM, ChatHuggingFace, VLLM, ChatVLLM


def test_huggingface():
<<<<<<< HEAD
    # 测试 HuggingFaceLLM 类的功能

    # 创建 HuggingFaceLLM 实例
    llm = HuggingFaceLLM(
        model_name="qwen-7b-chat",
        model_path="data/file_storage/qwen/Qwen-7B-Chat/model-00001-of-00008.safetensors",
        load_model_kwargs={"device_map": "auto"},
    )

    # 调用 invoke 方法并打印结果
    prompt = "user\n你是谁？\nassistant\n"
    print(llm.invoke(prompt, stop=[""]))

    # 使用 Token Streaming 遍历结果并打印
    for chunk in llm.stream(prompt, stop=[""]):
        print(chunk, end="", flush=True)

    # 调用 call_as_openai 方法并打印结果
    print(llm.call_as_openai(prompt, stop=[""]))

    # 使用 Streaming 遍历结果并打印
    for chunk in llm.call_as_openai(prompt, stop=[""], stream=True):
        print(chunk.choices[0].text, end="", flush=True)

    # 创建 ChatHuggingFace 实例
    chat_llm = ChatHuggingFace(llm=llm)

    # 调用 invoke 方法并打印结果
    query = "你是谁？"
    print(chat_llm.invoke(query))

    # 使用 Token Streaming 遍历结果并打印
    for chunk in chat_llm.stream(query):
        print(chunk.content, end="", flush=True)

    # 调用 call_as_openai 方法并打印结果
    messages = [{"role": "user", "content": query}]
    print(chat_llm.call_as_openai(messages))

    # 使用 Streaming 遍历结果并打印
=======
    llm = HuggingFaceLLM(
        model_name="qwen-7b-chat",
        model_path="/data/checkpoints/Qwen-7B-Chat",
        load_model_kwargs={"device_map": "auto"},
    )

    # invoke method
    prompt = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    print(llm.invoke(prompt, stop=["<|im_end|>"]))

    # Token Streaming
    for chunk in llm.stream(prompt, stop=["<|im_end|>"]):
        print(chunk, end="", flush=True)

    # openai usage
    print(llm.call_as_openai(prompt, stop=["<|im_end|>"]))

    # Streaming
    for chunk in llm.call_as_openai(prompt, stop=["<|im_end|>"], stream=True):
        print(chunk.choices[0].text, end="", flush=True)

    chat_llm = ChatHuggingFace(llm=llm)

    # invoke method
    query = "你是谁？"
    print(chat_llm.invoke(query))

    # Token Streaming
    for chunk in chat_llm.stream(query):
        print(chunk.content, end="", flush=True)

    # openai usage
    messages = [
        {"role": "user", "content": query}
    ]
    print(chat_llm.call_as_openai(messages))

    # Streaming
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    for chunk in chat_llm.call_as_openai(messages, stream=True):
        print(chunk.choices[0].delta.content or "", end="", flush=True)


def test_vllm():
<<<<<<< HEAD
    # 测试 VLLM 类的功能

    # 创建 VLLM 实例
    llm = VLLM(
        model_name="qwen",
        model="data/file_storage/qwen/Qwen-7B-Chat/model-00001-of-00008.safetensors",
        trust_remote_code=True,
    )

    # 调用 invoke 方法并打印结果
    prompt = "user\n你是谁？\nassistant\n"
    print(llm.invoke(prompt, stop=[""]))

    # 调用 call_as_openai 方法并打印结果
    print(llm.call_as_openai(prompt, stop=[""]))

    # 创建 ChatVLLM 实例
    chat_llm = ChatVLLM(llm=llm)

    # 调用 invoke 方法并打印结果
    query = "你是谁？"
    print(chat_llm.invoke(query))

    # 调用 call_as_openai 方法并打印结果
    messages = [{"role": "user", "content": query}]
=======
    llm = VLLM(
        model_name="qwen",
        model="/data/checkpoints/Qwen-7B-Chat",
        trust_remote_code=True,
    )

    # invoke method
    prompt = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    print(llm.invoke(prompt, stop=["<|im_end|>"]))

    # openai usage
    print(llm.call_as_openai(prompt, stop=["<|im_end|>"]))

    chat_llm = ChatVLLM(llm=llm)

    # invoke method
    query = "你是谁？"
    print(chat_llm.invoke(query))

    # openai usage
    messages = [
        {"role": "user", "content": query}
    ]
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    print(chat_llm.call_as_openai(messages))


if __name__ == "__main__":
    test_huggingface()
