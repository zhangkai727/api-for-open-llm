## 环境配置

使用 `docker` 或者本地环境两种方式之一，推荐使用 `docker`

### docker

构建镜像

```shell
docker build -f docker/Dockerfile.vllm -t llm-api:vllm .
```

### 本地环境

安装依赖，确保安装顺序严格按照下面的命令：

```shell
<<<<<<< HEAD
pip install vllm==0.4.3
=======
pip install vllm>=0.4.3
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
pip install -r requirements.txt 
# pip uninstall transformer-engine -y
```

## 启动模型

### 环境变量含义


<<<<<<< HEAD
+ `MODEL_NAME`: 模型名称，如 `qwen`、`baichuan-13b-chat` 等
=======
+ `MODEL_NAME`: 模型名称，如 `chatglm4`、`qwen2`、`llama3`等


+ `PROMPT_NAME`: 使用的对话模板名称，如果不指定，则将根据 `tokenizer` 找到对应的模板
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d


+ `MODEL_PATH`: 开源大模型的文件所在路径


+ `TRUST_REMOTE_CODE`: 是否使用外部代码


+ `TOKENIZE_MODE`（可选项）: `tokenizer` 的模式，默认为 `auto`


+ `TENSOR_PARALLEL_SIZE`（可选项）: `GPU` 数量，默认为 `1`


<<<<<<< HEAD
+ `PROMPT_NAME`（可选项）: 使用的对话模板名称，如果不指定，则将根据模型名找到对应的模板


=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
+ `EMBEDDING_NAME`（可选项）: 嵌入模型的文件所在路径，推荐使用 `moka-ai/m3e-base` 或者 `BAAI/bge-large-zh`


+ `GPU_MEMORY_UTILIZATION`（可选项）: `GPU` 占用率


+ `MAX_NUM_BATCHED_TOKENS`（可选项）: 每个批处理的最大 `token` 数量


+ `MAX_NUM_SEQS`（可选项）: 批量大小


<<<<<<< HEAD
=======
+ `TASKS`（可选项）: `llm` 表示启动对话大模型，`rag` 表示启动文档文档相关接口，比如`embedding`、`rerank`


>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
### 启动方式

选择下面两种方式之一启动模型接口服务

#### docker启动

1. docker run

不同模型只需要将 [.env.vllm.example](../.env.vllm.example) 文件内容复制到 `.env` 文件中

```shell
cp .env.vllm.example .env
```

然后修改 `.env` 文件中的环境变量

```shell
docker run -it -d --gpus all --ipc=host -p 7891:8000 --name=vllm-server \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:vllm \
    python api/server.py
```

2. docker compose

```shell
docker-compose -f docker-compose.vllm.yml up -d
```

#### 本地启动

同样的，将 [.env.vllm.example](../.env.vllm.example) 文件内容复制到 `.env` 文件中

```shell
cp .env.vllm.example .env
```

然后修改 `.env` 文件中的环境变量

```shell
cp api/server.py .
python server.py
```

## 环境变量修改参考

<<<<<<< HEAD
**环境变量修改内容参考下面**

+ [internlm2](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#internlm2)    

+ [code-llama](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#code-llama) 

+ [sqlcoder](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#sqlcoder) 

+ [qwen-7b-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#qwen-7b-chat)

+ [baichuan-13b-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#baichuan-13b-chat)

+ [internlm](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#internlm)      


### Qwen-7b-chat

Qwen/Qwen-7B-Chat:


```shell
MODEL_NAME=qwen
MODEL_PATH=Qwen/Qwen-7B-Chat # 模型所在路径，若使用docker，则为在容器内的路径
ENGINE=vllm
```

### InternLM

internlm-chat-7b:

```shell
MODEL_NAME=internlm
MODEL_PATH=internlm/internlm-chat-7b
ENGINE=vllm
```

### Baichuan-13b-chat

baichuan-inc/Baichuan-13B-Chat:

```shell
MODEL_NAME=baichuan-13b-chat
MODEL_PATH=baichuan-inc/Baichuan-13B-Chat
TENSOR_PARALLEL_SIZE=2
ENGINE=vllm
```

### SQLCODER

defog/sqlcoder:

```shell
MODEL_NAME=starcode
MODEL_PATH=defog/sqlcoder
TENSOR_PARALLEL_SIZE=2
ENGINE=vllm
```

### CODE-LLAMA

```shell
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/vllm-project/vllm.git
```

codellama/CodeLlama-7b-Instruct-hf

```shell
MODEL_NAME=code-llama
MODEL_PATH=codellama/CodeLlama-7b-Instruct-hf
ENGINE=vllm
```

### Xwin-LM

Xwin-LM/Xwin-LM-7B-V0.1

```shell
MODEL_NAME=xwin-7b
MODEL_PATH=Xwin-LM/Xwin-LM-7B-V0.1
PROMPT_NAME=vicuna
ENGINE=vllm
```

### InternLM2

internlm2-chat-20b:

```shell
MODEL_NAME=internlm2
MODEL_PATH=internlm/internlm2-chat-20b
ENGINE=vllm
TENSOR_PARALLEL_SIZE=2
```
=======
### QWEN系列

| 模型      | 环境变量示例                                                                                    |
|---------|-------------------------------------------------------------------------------------------|
| qwen    | `MODEL_NAME=qwen`、`MODEL_PATH=Qwen/Qwen-7B-Chat`、`PROMPT_NAME=qwen`、 `ENGINE=vllm`        |
| qwen1.5 | `MODEL_NAME=qwen2`、`MODEL_PATH=Qwen/Qwen1.5-7B-Chat`、`PROMPT_NAME=qwen2`、 `ENGINE=vllm`   |
| qwen2   | `MODEL_NAME=qwen2`、`MODEL_PATH=Qwen/Qwen2-7B-Instruct`、`PROMPT_NAME=qwen2`、 `ENGINE=vllm` |


### GLM系列

| 模型       | 环境变量示例                                                                                       |
|----------|----------------------------------------------------------------------------------------------|
| chatglm  | `MODEL_NAME=chatglm`、`MODEL_PATH=THUDM/chatglm-6b`、`PROMPT_NAME=chatglm`、 `ENGINE=vllm`      |
| chatglm2 | `MODEL_NAME=chatglm2`、`MODEL_PATH=THUDM/chatglm2-6b`、`PROMPT_NAME=chatglm2`、 `ENGINE=vllm`   |
| chatglm3 | `MODEL_NAME=chatglm3`、`MODEL_PATH=THUDM/chatglm3-6b`、`PROMPT_NAME=chatglm3`、 `ENGINE=vllm`   |
| glm4     | `MODEL_NAME=chatglm4`、`MODEL_PATH=THUDM/glm-4-9b-chat`、`PROMPT_NAME=chatglm4`、 `ENGINE=vllm` |


### BAICHUAN系列

| 模型        | 环境变量示例                                                                                                     |
|-----------|------------------------------------------------------------------------------------------------------------|
| baichuan  | `MODEL_NAME=baichuan`、`MODEL_PATH=baichuan-inc/Baichuan-13B-Chat`、`PROMPT_NAME=baichuan`、 `ENGINE=vllm`    |
| baichuan2 | `MODEL_NAME=baichuan2`、`MODEL_PATH=baichuan-inc/Baichuan2-13B-Chat`、`PROMPT_NAME=baichuan2`、 `ENGINE=vllm` |


### INTERNLM系列

| 模型        | 环境变量示例                                                                                                 |
|-----------|--------------------------------------------------------------------------------------------------------|
| internlm  | `MODEL_NAME=internlm`、`MODEL_PATH=internlm/internlm-chat-7b`、`PROMPT_NAME=internlm`、 `ENGINE=vllm`     |
| internlm2 | `MODEL_NAME=internlm2`、`MODEL_PATH=internlm/internlm2-chat-20b`、`PROMPT_NAME=internlm2`、 `ENGINE=vllm` |


### Yi系列

| 模型    | 环境变量示例                                                                              |
|-------|-------------------------------------------------------------------------------------|
| yi    | `MODEL_NAME=yi`、`MODEL_PATH=01-ai/Yi-34B-Chat`、`PROMPT_NAME=yi`、 `ENGINE=vllm`      |
| yi1.5 | `MODEL_NAME=yi1.5`、`MODEL_PATH=01-ai/Yi1.5-9B-Chat`、`PROMPT_NAME=yi`、 `ENGINE=vllm` |


### DEEPSEEK系列

| 模型             | 环境变量示例                                                                                                                       |
|----------------|------------------------------------------------------------------------------------------------------------------------------|
| deepseek-coder | `MODEL_NAME=deepseek-coder`、`MODEL_PATH=deepseek-ai/deepseek-coder-33b-instruct`、`PROMPT_NAME=deepseek-coder`、 `ENGINE=vllm` |
| deepseek-chat  | `MODEL_NAME=deepseek`、`MODEL_PATH=deepseek-ai/deepseek-llm-67b-chat`、`PROMPT_NAME=deepseek`、 `ENGINE=vllm`                   |
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
