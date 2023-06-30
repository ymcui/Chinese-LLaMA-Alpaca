# OPENAI API DEMO

> 更加详细的OPENAI API信息：<https://platform.openai.com/docs/api-reference>

这是一个使用fastapi实现的简易的仿OPENAI API风格的服务器DEMO，您可以使用这个API DEMO来快速搭建基于中文大模型的个人网站以及其他有趣的WEB DEMO。

## 部署方式

安装依赖
``` shell
pip install fastapi uvicorn shortuuid
```

启动脚本
``` shell
python scripts/openai_server_demo/openai_api_server.py --base_model /path/to/base_model --lora_model /path/to/lora_model --gpus 0,1
```

### 参数说明

`--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录，可以是合并后的中文Alpaca或Alpaca Plus模型（此时无需提供`--lora_model`），也可以是转后HF格式后的原版LLaMA模型（需要提供`--lora_model`）

`--lora_model {lora_model}`：中文Alpaca LoRA解压后文件所在目录，也可使用🤗Model Hub模型调用名称。若不提供此参数，则只加载--base_model指定的模型

`--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与`--lora_model`相同；若也未提供`--lora_model`参数，则其默认值与--base_model相同

`--only_cpu`: 仅使用CPU进行推理

`--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2

`--load_in_8bit`: 使用8bit模型进行推理，可节省显存，但可能影响模型效果

## API文档

### 文字接龙（completion）

> 有关completion的中文翻译，李宏毅教授将其翻译为文字接龙 <https://www.youtube.com/watch?v=yiY4nPOzJEg>

最基础的API接口，输入prompt，输出语言大模型的文字接龙（completion）结果。

API DEMO内置有alpaca prompt模板，prompt将被套入alpaca instruction模板中，这里输入的prompt应更像指令而非对话。

#### 快速体验completion接口

请求command：

``` shell
curl http://localhost:19327/v1/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "prompt": "告诉我中国的首都在哪里"
  }'
```

json返回体：

``` json
{
    "id": "cmpl-3watqWsbmYgbWXupsSik7s",
    "object": "text_completion",
    "created": 1686067311,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "text": "中国的首都是北京。"
        }
    ]
}
```

#### completion接口高级参数

请求command：

``` shell
curl http://localhost:19327/v1/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "prompt": "告诉我中国和美国分别各有哪些优点缺点",
    "max_tokens": 90,
    "temperature": 0.7,
    "num_beams": 4,
    "top_k": 40
  }'
```

json返回体：

``` json
{
    "id": "cmpl-PvVwfMq2MVWHCBKiyYJfKM",
    "object": "text_completion",
    "created": 1686149471,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "text": "中国的优点是拥有丰富的文化和历史，而美国的优点是拥有先进的科技和经济体系。"
        }
    ]
}
```

#### completion接口高级参数说明

> 有关Decoding策略，更加详细的细节可以参考 <https://towardsdatascience.com/the-three-decoding-methods-for-nlp-23ca59cb1e9d> 该文章详细讲述了三种Llama会用到的Decoding策略：Greedy Decoding、Random Sampling 和 Beam Search，Decoding策略是top_k、top_p、temperature、num_beam等高级参数的基础。

`prompt`: 生成文字接龙（completion）的提示。

`max_tokens`: 新生成的句子的token长度。

`temperature`: 在0和2之间选择的采样温度。较高的值如0.8会使输出更加随机，而较低的值如0.2则会使其输出更具有确定性。temperature越高，使用随机采样最为decoding的概率越大。

`num_beams`: 当搜索策略为束搜索（beam search）时，该参数为在束搜索（beam search）中所使用的束个数，当num_beams=1时，实际上就是贪心搜索（greedy decoding）。

`top_k`: 在随机采样（random sampling）时，前top_k高概率的token将作为候选token被随机采样。

`top_p`: 在随机采样（random sampling）时，累积概率超过top_p的token将作为候选token被随机采样，越低随机性越大，举个例子，当top_p设定为0.6时，概率前5的token概率分别为{0.23, 0.20, 0.18, 0.11, 0.10}时，前三个token的累积概率为0.61，那么第4个token将被过滤掉，只有前三的token将作为候选token被随机采样。

`repetition_penalty`: 重复惩罚，具体细节可以参考这篇文章：<https://arxiv.org/pdf/1909.05858.pdf> 。

`do_sample`: 启用随机采样策略。默认为true。

### 聊天（chat completion）

聊天接口支持多轮对话

#### 快速体验聊天接口

请求command：

``` shell
curl http://localhost:19327/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "messages": [
      {"role": "user","message": "给我讲一些有关杭州的故事吧"}
    ],
    "repetition_penalty": 1.0
  }'
```

json返回体：

``` json
{
    "id": "chatcmpl-5L99pYoW2ov5ra44Ghwupt",
    "object": "chat.completion",
    "created": 1686143170,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "user",
                "content": "给我讲一些有关杭州的故事吧"
            }
        },
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": "好的，请问您对杭州有什么特别的偏好吗？"
            }
        }
    ]
}
```

#### 多轮对话

请求command：

``` shell
curl http://localhost:19327/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "messages": [
      {"role": "user","message": "给我讲一些有关杭州的故事吧"},
      {"role": "assistant","message": "好的，请问您对杭州有什么特别的偏好吗？"},
      {"role": "user","message": "我比较喜欢和西湖，可以给我讲一下西湖吗"}
    ],
    "repetition_penalty": 1.0
  }'
```

json返回体：

``` json
{
    "id": "chatcmpl-hmvrQNPGYTcLtmYruPJbv6",
    "object": "chat.completion",
    "created": 1686143439,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "user",
                "content": "给我讲一些有关杭州的故事吧"
            }
        },
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": "好的，请问您对杭州有什么特别的偏好吗？"
            }
        },
        {
            "index": 2,
            "message": {
                "role": "user",
                "content": "我比较喜欢和西湖，可以给我讲一下西湖吗"
            }
        },
        {
            "index": 3,
            "message": {
                "role": "assistant",
                "content": "是的，西湖是杭州最著名的景点之一，它被誉为“人间天堂”。 <\\s>"
            }
        }
    ]
}
```

#### 聊天接口高级参数说明

`prompt`: 生成文字接龙（completion）的提示。

`max_tokens`: 新生成的句子的token长度。

`temperature`: 在0和2之间选择的采样温度。较高的值如0.8会使输出更加随机，而较低的值如0.2则会使其输出更具有确定性。temperature越高，使用随机采样最为decoding的概率越大。

`num_beams`: 当搜索策略为束搜索（beam search）时，该参数为在束搜索（beam search）中所使用的束个数，当num_beams=1时，实际上就是贪心搜索（greedy decoding）。

`top_k`: 在随机采样（random sampling）时，前top_k高概率的token将作为候选token被随机采样。

`top_p`: 在随机采样（random sampling）时，累积概率超过top_p的token将作为候选token被随机采样，越低随机性越大，举个例子，当top_p设定为0.6时，概率前5的token概率分别为[0.23, 0.20, 0.18, 0.11, 0.10]时，前三个token的累积概率为0.61，那么第4个token将被过滤掉，只有前三的token将作为候选token被随机采样。

`repetition_penalty`: 重复惩罚，具体细节可以参考这篇文章：<https://arxiv.org/pdf/1909.05858.pdf> 。

`do_sample`: 启用随机采样策略。默认为true。

### 文本嵌入向量（text embedding）

文本嵌入向量有很多作用，包括但不限于基于大型文档问答、总结一本书中的内容、为大语言模型找到与当前用户输入最相近的记忆等等。

请求command：

``` shell
curl http://localhost:19327/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天天气真不错"
  }'
```

json返回体：

``` json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                0.003643923671916127,
                -0.0072653163224458694,
                0.0075545101426541805, 
                ....,
                0.0045851171016693115
            ],
            "index": 0
        }
    ],
    "model": "chinese-llama-alpaca"
}
```

embedding向量的长度与所使用模型hidden size相同。比如当使用7B模型时，embedding的长度为4096。
