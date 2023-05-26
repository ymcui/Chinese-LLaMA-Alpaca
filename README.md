[**🇨🇳中文**](./README.md) | [**🌐English**](./README_EN.md) | [**📖文档/Docs**](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki) | [**❓提问/Issues**](https://github.com/ymcui/Chinese-LLaMA-Alpaca/issues) | [**💬讨论/Discussions**](https://github.com/ymcui/Chinese-LLaMA-Alpaca/discussions)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="700"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ymcui/Chinese-LLaMA-Alpaca">
    <a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki"><img alt="GitHub wiki" src="https://img.shields.io/badge/Github%20Wiki-v3.2-green"></a>
</p>



以ChatGPT、GPT-4等为代表的大语言模型（Large Language Model, LLM）掀起了新一轮自然语言处理领域的研究浪潮，展现出了类通用人工智能（AGI）的能力，受到业界广泛关注。然而，由于大语言模型的训练和部署都极为昂贵，为构建透明且开放的学术研究造成了一定的阻碍。

为了促进大模型在中文NLP社区的开放研究，本项目开源了**中文LLaMA模型和指令精调的Alpaca大模型**。这些模型**在原版LLaMA的基础上扩充了中文词表**并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力。详细内容请参考技术报告[(Cui, Yang, and Yao, 2023)](https://arxiv.org/abs/2304.08177)。

**本项目主要内容：**

- 🚀 针对原版LLaMA模型扩充了中文词表，提升了中文编解码效率 
- 🚀 开源了使用中文文本数据预训练的中文LLaMA以及经过指令精调的中文Alpaca
- 🚀 开源了预训练脚本、指令精调脚本，用户可根据需要自行进一步训练
- 🚀 快速使用笔记本电脑（个人PC）的CPU/GPU本地量化和部署体验大模型
- 🚀 支持[🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LlamaChat](https://github.com/alexrozanski/LlamaChat), [LangChain](https://github.com/hwchase17/langchain), [privateGPT](https://github.com/imartinez/privateGPT)等生态
- 目前已开源的模型版本：7B（标准版、**Plus版**）、13B（标准版、**Plus版**）

💡 下图是中文Alpaca-7B模型在本地CPU量化部署后的实际体验效果（GIF未加速，M1 Max下实测）。

![](./pics/screencast.gif)

----

[多模态VLE](https://github.com/iflytek/VLE) | [中文MiniRBT](https://github.com/iflytek/MiniRBT) | [中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner)

## 新闻

**[2023/05/16] [v3.2版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.2): 添加指令精调脚本、LangChain支持、基于Gradio的本地Demo等。**

[2023/05/10] [v3.1版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.1): 发布中文LLaMA/Alpaca Plus-13B，使用了更大语料训练，相比基础版各项能力显著提升。

[2023/04/28] [v3.0版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.0): 发布中文LLaMA/Alpaca Plus版本（7B），使用了更大的语料进行训练，相比基础版各项能力显著提升。另外还进一步完善了评测流程、添加了预训练脚本等。

<details>
<summary><b>往期新闻</b></summary>

[2023/04/18] [v2.2版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.2)：添加LlamaChat支持、中文词表、LLaMA Tokenizer词表扩充脚本、添加技术报告等。

[2023/04/13] [v2.1版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.1)：添加HuggingFace推理接口、text-generation-webui接口。

[2023/04/07] [v2.0版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.0)：发布13B版本中文LLaMA、Alpaca大模型，主要升级：更强的事实性、文本问答、翻译、伦理拒答等能力全面提升。

[2023/04/03] 添加了模型合并和量化的notebook，Colab Pro(+)用户可在线合并和下载模型。请参考：[合并模型](#合并模型)

[2023/03/31] Release v1.1：简化模型合并步骤、添加指令数据爬取脚本、关于新版本llama.cpp的重要提示。请参考：[Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v1.1)

[2023/03/28] 正式开源中文LLaMA、Alpaca大模型，目前提供7B版本下载体验
</details>

## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [⏬模型下载](#模型下载)        | 中文LLaMA、Alpaca大模型下载地址                |
| [🈴合并模型](#合并模型) | （重要）介绍如何将下载的LoRA模型与原版LLaMA合并 |
| [💻本地推理与快速部署](#本地推理与快速部署) | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [💯系统效果](#系统效果) | 介绍了部分场景和任务下的使用体验效果             |
| [📝训练细节](#训练细节) | 介绍了中文LLaMA、Alpaca大模型的训练细节 |
| [❓FAQ](#FAQ) | 一些常见问题的回复 |
| [⚠️局限性](#局限性) | 本项目涉及模型的局限性 |


## 模型下载

### 用户须知（必读）

Facebook官方发布的[LLaMA模型禁止商用](https://github.com/facebookresearch/llama)，并且官方没有正式开源模型权重（虽然网上已经有很多第三方的下载地址）。为了遵循相应的许可，目前暂时无法发布完整的模型权重，敬请各位理解（目前国外也是一样）。Facebook完全开放模型权重之后，本项目会及时更新相关策略。**这里发布的是LoRA权重**，可以理解为原LLaMA模型上的一个“补丁”，两者进行合并即可获得完整版权重。以下中文LLaMA/Alpaca LoRA模型无法单独使用，需要搭配[原版LLaMA模型](https://github.com/facebookresearch/llama)。请参考本项目给出的[合并模型](#合并模型)步骤重构模型。

### 我应该选什么模型？

下面是中文LLaMA和Alpaca模型的基本对比以及建议使用场景（包括但不限于），更多内容见[训练细节](#训练细节)。

💡 **推荐使用Plus版**，模型体积与基础版相同，但使用了更多数据进行训练。

| 对比项                | 中文LLaMA                                              | 中文Alpaca                                                   |
| :-------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| 训练方式              | 传统CLM                            | 指令精调                                                     |
| 训练语料 | 无标注通用语料 | 有标注指令数据 |
| 词表大小<sup>[3]</sup> | 4995**3** | 4995**4**=49953+1（pad token） |
| 输入模板              | 不需要                                                 | 需要符合模板要求<sup>[1]</sup> |
| 适用场景 ✔️            | 文本续写：给定上文内容，让模型继续写下去，生成下文            | 1、指令理解（问答、写作、建议等）<br/>2、多轮上下文理解（聊天等） |
| 不适用场景 ❌          | 指令理解 、多轮聊天等                                  |  文本无限制自由生成                                                       |
| llama.cpp             | 使用`-p`参数指定上文                                   | 使用`-ins`参数启动指令理解+聊天模式                          |
| text-generation-webui |  不适合chat模式                              |    使用`--cpu`可在无显卡形式下运行，若生成内容不满意，建议修改prompt                                                          |
| LlamaChat             | 加载模型时选择"LLaMA"                                  | 加载模型时选择"Alpaca"                                       |
| [HF推理代码](./scripts/inference_hf.py) | 无需添加额外启动参数 | 启动时添加参数 `--with_prompt`        |
| [web-demo代码](./scripts/gradio_demo.py) | 不适用 | 直接提供Alpaca模型位置即可；支持多轮对话 |
| [LangChain示例](./scripts/langchain_demo) / privateGPT | 不适用 | 直接提供Alpaca模型位置即可 |
| 已知问题              | 如果不控制终止，则会一直写下去，直到达到输出长度上限。<sup>[2]</sup> | 目前版本模型生成的文本长度相对短一些，比较惜字如金。可在指令中要求详细回答。<sup>[2]</sup> |

*[1] llama.cpp/LlamaChat/[HF推理代码](./scripts/inference_hf.py)/[web-demo代码](./scripts/gradio_demo.py)/[LangChain示例](./scripts/langchain_demo)等已内嵌，无需手动添加模板。*<br/>
*[2] 如果出现了模型回答质量特别低、胡言乱语、不理解问题等情况，请检查是否针对场景使用了正确的模型和正确的启动参数。*<br/>
*[3] 经过指令精调的Alpaca会比LLaMA多一个pad token，**因此请勿混用LLaMA/Alpaca词表**。*


### 中文LLaMA模型

中文LLaMA模型在原版的基础上扩充了中文词表，使用了中文通用纯文本数据进行二次预训练。

| 模型名称                 | 训练数据 | 重构模型<sup>[1]</sup> | 大小<sup>[2]</sup> |                    LoRA下载<sup>[3]</sup>                    |
| :----------------------- | :------: | :--------------------: | :----------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-7B         | 通用20G  |      原版LLaMA-7B      |        770M        | [[百度网盘]](https://pan.baidu.com/s/1oORTdpr2TvlkxjpyWtb5Sw?pwd=33hb)</br>[[Google Drive]](https://drive.google.com/file/d/1iQp9T-BHjBjIrFWXq_kIm_cyNmpvv5WN/view?usp=sharing) |
| Chinese-LLaMA-Plus-7B ⭐️  | 通用120G |      原版LLaMA-7B      |        790M        | [[百度网盘]](https://pan.baidu.com/s/1zvyX9FN-WSRDdrtMARxxfw?pwd=2gtr)</br>[[Google Drive]](https://drive.google.com/file/d/1N97m3rBj-rp-J1X8rgRfluyomEscfAq0/view?usp=sharing) |
| Chinese-LLaMA-13B        | 通用20G  |     原版LLaMA-13B      |         1G         | [[百度网盘]](https://pan.baidu.com/s/1BxFhYhDMipW7LwI58cGmQQ?pwd=ef3t)<br/>[[Google Drive]](https://drive.google.com/file/d/12q9EH4mfKRnoKlbkkhzv1xDwWnroo9VS/view?usp=sharing) |
| Chinese-LLaMA-Plus-13B ⭐️ | 通用120G |     原版LLaMA-13B      |         1G         | [[百度网盘]](https://pan.baidu.com/s/1VGpNlrLx5zHuNzLOcTG-xw?pwd=8cvd)<br/>[[Google Drive]](https://drive.google.com/file/d/1q0L5Me_1j_9iiRRNfuEFUt3SOjQo3-g3/view?usp=share_link) |


### 中文Alpaca模型

中文Alpaca模型在上述中文LLaMA模型的基础上进一步使用了指令数据进行精调。**如希望体验类ChatGPT对话交互，请使用Alpaca模型，而不是LLaMA模型。**

| 模型名称                  | 训练数据 |                   重构模型<sup>[1]</sup>                   | 大小<sup>[2]</sup> |                    LoRA下载<sup>[3]</sup>                    |
| :------------------------ | :------: | :--------------------------------------------------------: | :----------------: | :----------------------------------------------------------: |
| Chinese-Alpaca-7B         |  指令2M  |                        原版LLaMA-7B                        |        790M        | [[百度网盘]](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e)</br>[[Google Drive]](https://drive.google.com/file/d/1JvFhBpekYiueWiUL3AF1TtaWDb3clY5D/view?usp=sharing) |
| Chinese-Alpaca-Plus-7B ⭐️  |  指令4M  |  *原版LLaMA-7B &<br/>Chinese-LLaMA-Plus-7B*<sup>[4]</sup>  |        1.1G        | [[百度网盘]](https://pan.baidu.com/s/12tjjxmDWwLBM8Tj_7FAjHg?pwd=32hc)</br>[[Google Drive]](https://drive.google.com/file/d/1EDcTmq6tDmRxqarpapdyDGBE9opY0zrB/view?usp=share_link) |
| Chinese-Alpaca-13B        |  指令3M  |                       原版LLaMA-13B                        |        1.1G        | [[百度网盘]](https://pan.baidu.com/s/1wYoSF58SnU9k0Lndd5VEYg?pwd=mm8i)<br/>[[Google Drive]](https://drive.google.com/file/d/1gzMc0xMCpXsXmU1uxFlgQ8VRnWNtDjD8/view?usp=share_link) |
| Chinese-Alpaca-Plus-13B ⭐️ | 指令4.3M | *原版LLaMA-13B &<br/>Chinese-LLaMA-Plus-13B<sup>[4]</sup>* |        1.3G        | [[百度网盘]](https://pan.baidu.com/s/1Mew4EjBlejWBBB6_WW6vig?pwd=mf5w)<br/> [[Google Drive]](https://drive.google.com/file/d/1CcLJvY7XsFAOjfSIqCpDI7jf3EEPDcEF/view?usp=share_link) |

### Model Hub

可以在🤗Model Hub下载以上所有模型，并且使用[transformers](https://github.com/huggingface/transformers)和[PEFT](https://github.com/huggingface/peft)调用中文LLaMA或Alpaca LoRA模型。以下模型调用名称指的是使用`.from_pretrained()`中指定的模型名称。

| 模型名                  | 模型调用名称                            |                             链接                             |
| ----------------------- | :-------------------------------------- | :----------------------------------------------------------: |
| Chinese-LLaMA-7B        | ziqingyang/chinese-llama-lora-7b        | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-7b) |
| Chinese-LLaMA-Plus-7B   | ziqingyang/chinese-llama-plus-lora-7b   | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b) |
| Chinese-LLaMA-13B       | ziqingyang/chinese-llama-lora-13b       | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-13b) |
| Chinese-LLaMA-Plus-13B  | ziqingyang/chinese-llama-plus-lora-13b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-13b) |
| Chinese-Alpaca-7B       | ziqingyang/chinese-alpaca-lora-7b       | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b) |
| Chinese-Alpaca-Plus-7B  | ziqingyang/chinese-alpaca-plus-lora-7b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b) |
| Chinese-Alpaca-13B      | ziqingyang/chinese-alpaca-lora-13b      | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b) |
| Chinese-Alpaca-Plus-13B | ziqingyang/chinese-alpaca-plus-lora-13b | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-13b) |

### 脚注及其他说明

**[1]** 重构需要原版LLaMA模型，[去LLaMA项目申请使用](https://github.com/facebookresearch/llama)或参考这个[PR](https://github.com/facebookresearch/llama/pull/73/files)。因版权问题本项目无法提供下载链接。

**[2]** 经过重构后的模型大小比同等量级的原版LLaMA大一些（主要因为扩充了词表）。

**[3]** 下载后务必检查压缩包中模型文件的SHA256是否一致，请查看[SHA256.md](./SHA256.md)。

**[4]** Alpaca-Plus模型的合并方法略有不同，请参考[合并教程](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/手动模型合并与转换#多lora权重合并适用于chinese-alpaca-plus)。

压缩包内文件目录如下（以Chinese-LLaMA-7B为例）：

```
chinese_llama_lora_7b/
  - adapter_config.json		# LoRA权重配置文件
  - adapter_model.bin		# LoRA权重文件
  - special_tokens_map.json	# special_tokens_map文件
  - tokenizer_config.json	# tokenizer配置文件
  - tokenizer.model		# tokenizer文件 
```

以下是各原模型和4-bit量化后的大小，转换相应模型时确保本机有足够的内存和磁盘空间（最低要求）：

| 模型版本            |   7B   |   13B   |   33B   |   65B   |
| :------------------ | :----: | :-----: | :-----: | :-----: |
| 原模型大小（FP16）  | 13 GB  |  24 GB  |  60 GB  | 120 GB  |
| 量化后大小（8-bit） | 7.8 GB | 14.9 GB |    -    |    -    |
| 量化后大小（4-bit） | 3.9 GB | 7.8 GB  | 19.5 GB | 38.5 GB |

## 合并模型

前面提到LoRA模型无法单独使用，必须与原版LLaMA进行合并才能转为完整模型，以便进行模型推理、量化或者进一步训练。请选择以下方法对模型进行转换合并。

| 方式         | 适用场景                                                   |                             教程                             |
| :----------- | :--------------------------------------------------------- | :----------------------------------------------------------: |
| **在线转换** | Colab用户可利用本项目提供的notebook进行在线转换并量化模型  | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/在线模型合并与转换) |
| **手动转换** | 离线方式转换，生成不同格式的模型，以便进行量化或进一步精调 | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/手动模型合并与转换) |

具体内容请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/模型合并与转换)

## 本地推理与快速部署

本项目中的模型主要支持以下量化、推理和部署方式。

| 推理和部署方式                                               | 特点                                         | 平台  | CPU  | GPU  | 量化加载 | 图形界面 |                             教程                             |
| :----------------------------------------------------------- | -------------------------------------------- | :---: | :--: | :--: | :------: | :------: | :----------------------------------------------------------: |
| [**llama.cpp**](https://github.com/ggerganov/llama.cp)       | 丰富的量化选项和高效本地推理                 | 通用  |  ✅   |  ✅   |    ✅     |    ❌     | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/llama.cpp量化部署) |
| [**🤗Transformers**](https://github.com/huggingface/transformers) | 原生transformers推理接口                     | 通用  |  ✅   |  ✅   |    ✅     |    ✅     | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/使用Transformers推理) |
| [**text-generation-webui**](https://github.com/oobabooga/text-generation-webui) | 前端Web UI界面的部署方式                     | 通用  |  ✅   |  ✅   |    ✅     |    ✅     | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/使用text-generation-webui搭建界面) |
| [**LlamaChat**](https://github.com/alexrozanski/LlamaChat)   | macOS下的图形交互界面（需搭配llama.cpp模型） | MacOS |  ✅   |  ❌   |    ✅     |    ✅     | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/使用LlamaChat图形界面（macOS）) |
| [**LangChain**](https://github.com/hwchase17/langchain)      | LLM应用开发框架，适用于进行二次开发          | 通用  | ✅<sup>†</sup> |  ✅   | ✅<sup>†</sup> |    ❌     | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/与LangChain进行集成) |
| [**privateGPT**](https://github.com/imartinez/privateGPT) | 基于LangChain的多文档本地问答框架 | 通用 | ✅ | ✅ | ✅ | ❌ | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/使用privateGPT进行多文档问答) |

<sup>†</sup>: LangChain框架支持，但教程中未实现；详细说明请参考LangChain官方文档。

具体内容请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/模型推理与部署)


## 系统效果

为了快速评测相关模型的实际表现，本项目在给定相同的prompt的情况下，在一些常见任务上对比测试了本项目的中文Alpaca-7B、中文Alpaca-13B、中文Alpaca-Plus-7B、中文Alpaca-Plus-13B的效果。生成回复具有随机性，受解码超参、随机种子等因素影响。以下相关评测并非绝对严谨，测试结果仅供晾晒参考，欢迎自行体验。详细评测结果请查看[examples目录](./examples)。

| 测试任务         | 样例数 | Alpaca-13B | Alpaca-Plus-7B | Alpaca-Plus-13B |
| ---------------- | :----: | :--------: | :------------: | :-------------: |
| **💯总平均分**    |  200   |    74.3    |      78.2      |   **👍🏻80.8**    |
| 知识问答         |   20   |     70     |       74       |    **👍🏻79**     |
| 开放式问答       |   20   |     77     |       77       |       77        |
| 数值计算、推理   |   20   |     61     |       61       |       60        |
| 诗词、文学、哲学 |   20   |     65     |    **👍🏻76**    |    **👍🏻76**     |
| 音乐、体育、娱乐 |   20   |     68     |       73       |    **👍🏻80**     |
| 写信、写文章     |   20   |     83     |       82       |    **👍🏻87**     |
| 文本翻译         |   20   |     84     |       87       |    **👍🏻90**     |
| 多轮交互         |   20   |     88     |       89       |       89        |
| 代码编程         |   20   |     65     |       64       |    **👍🏻70**     |
| 伦理、拒答       |   20   |     82     |    **👍🏻99**    |    **👍🏻100**    |


## 训练细节

整个训练流程包括词表扩充、预训练和指令精调三部分。

- 本项目的模型均在原LLaMA词表的基础上扩充了中文单词，代码请参考[merge_tokenizers.py](scripts/merge_tokenizers.py)
- 预训练和指令精调代码参考了🤗transformers中的[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)和[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)项目中数据集处理的相关部分
- 已开源用于预训练和指令精调的训练脚本：
  - 预训练脚本：[scripts/run_clm_pt_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/run_clm_pt_with_peft.py)，相关使用教程请参考[预训练脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/预训练脚本)
  - 指令精调脚本：[scripts/run_clm_sft_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/run_clm_sft_with_peft.py)，相关使用教程请参考[指令精调脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/指令精调脚本)


具体内容请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/训练细节)

## FAQ

FAQ中给出了常见问题的解答，请在提Issue前务必先查看FAQ。

```
问题1：为什么不能放出完整版本权重？
问题2：后面会有33B、65B的版本吗？
问题3：一些任务上效果不好！
问题4：为什么要扩充词表？直接在原版LLaMA上用中文预训练不行吗？
问题5：回复内容很短
问题6：Windows下，模型无法理解中文、生成速度很慢等问题
问题7：Chinese-LLaMA 13B模型没法用llama.cpp启动，提示维度不一致
问题8：Chinese-Alpaca-Plus效果很差
问题9：模型在NLU类任务（文本分类等）上效果不好
```

具体问题和解答请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/常见问题)


## 局限性

虽然本项目中的模型具备一定的中文理解和生成能力，但也存在局限性，包括但不限于：

- 可能会产生不可预测的有害内容以及不符合人类偏好和价值观的内容
- 由于算力和数据问题，相关模型的训练并不充分，中文理解能力有待进一步提升
- 暂时没有在线可互动的demo（注：用户仍然可以自行在本地部署）


## 引用

如果您觉得本项目对您的研究有所帮助或使用了本项目的代码或数据，请参考引用本项目的技术报告：https://arxiv.org/abs/2304.08177
```
@article{chinese-llama-alpaca,
      title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca}, 
      author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
      journal={arXiv preprint arXiv:2304.08177},
      url={https://arxiv.org/abs/2304.08177},
      year={2023}
}
```

## 致谢

本项目基于以下开源项目二次开发，在此对相关项目和研究开发人员表示感谢。

|                        基础模型、代码                        |                       量化、推理、部署                       |                             数据                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [LLaMA by Facebook](https://github.com/facebookresearch/llama)<br/>[Alpaca by Stanford](https://github.com/tatsu-lab/stanford_alpaca)<br/>[alpaca-lora by @tloen](https://github.com/tloen/alpaca-lora) | [llama.cpp by @ggerganov](https://github.com/ggerganov/llama.cpp)<br/>[LlamaChat by @alexrozanski]( https://github.com/alexrozanski/LlamaChat)<br/>[text-generation-webui by @oobabooga](https://github.com/oobabooga/text-generation-webui) | [pCLUE and MT data by @brightmart](https://github.com/brightmart/nlp_chinese_corpus) |

Episode: The current logo is automatically generated by GPT-4 with the DALL·E plugin (previously generated by midjourney).

## 免责声明

**本项目相关资源仅供学术研究之用，严禁用于商业用途。** 使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

本项目由个人及协作者业余时间发起并维护，因此无法保证能及时回复解决相应问题。


## 问题反馈
如有问题，请在GitHub Issue中提交。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 提交问题请使用本项目设置的Issue模板，以帮助快速定位具体问题。
- 重复以及与本项目无关的issue会被[stable-bot](https://github.com/marketplace/stale)处理，敬请谅解。
- 礼貌地提出问题，构建和谐的讨论社区。
