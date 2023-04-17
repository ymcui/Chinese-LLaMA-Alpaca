[**中文**](./README.md) | [**English**](./README_EN.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="600"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ymcui/Chinese-LLaMA-Alpaca">
</p>


以ChatGPT、GPT-4等为代表的大语言模型（Large Language Model, LLM）掀起了新一轮自然语言处理领域的研究浪潮，展现出了类通用人工智能（AGI）的能力，受到业界广泛关注。然而，由于大语言模型的训练和部署都极为昂贵，为构建透明且开放的学术研究造成了一定的阻碍。

为了促进大模型在中文NLP社区的开放研究，本项目开源了**中文LLaMA模型和指令精调的Alpaca大模型**。这些模型**在原版LLaMA的基础上扩充了中文词表**并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力。

***声明：本项目相关资源仅供学术研究使用。项目文档： [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki))***

**本项目主要内容：**

- 🚀 针对原版LLaMA模型扩充了中文词表，提升了中文编解码效率 
- 🚀 开源了使用中文文本数据预训练的中文LLaMA大模型（7B、13B）
- 🚀 开源了进一步经过指令精调的中文Alpaca大模型（7B、13B）
- 🚀 快速使用笔记本电脑（个人PC）的CPU/GPU本地部署和体验大模型

💡 下图给出了7B版本模型本地CPU部署后的实际体验效果（动画未经加速，Apple M1 Max下实测）。

![](./pics/screencast.gif)

----

[多模态VLE](https://github.com/iflytek/VLE) | [中文MiniRBT](https://github.com/iflytek/MiniRBT) | [中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner)

## 新闻

**[2023/04/17] Release v2.2：添加LlamaChat支持（macOS图形界面）、文档移至GitHub Wiki。请参考：[Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.2)**

[2023/04/13] Release v2.1：添加HuggingFace推理接口、text-generation-webui接口。请参考：[Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.1)

[2023/04/07] Release v2.0：发布13B版本中文LLaMA、Alpaca大模型，主要升级：更强的事实性、文本问答、翻译、伦理拒答等能力全面提升！更多更新内容请参考：[Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.0)

[2023/04/03] 添加了模型合并和量化的notebook，Colab Pro(+)用户可在线合并和下载模型。请参考：[合并模型](#合并模型)

[2023/03/31] Release v1.1：简化模型合并步骤、添加指令数据爬取脚本、关于新版本llama.cpp的重要提示。请参考：[Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v1.1)

[2023/03/28] 正式开源中文LLaMA、Alpaca大模型，目前提供7B版本下载体验

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

Facebook官方发布的[LLaMA模型禁止商用](https://github.com/facebookresearch/llama)，并且官方没有正式开源模型权重（虽然网上已经有很多第三方的下载地址）。为了遵循相应的许可，目前暂时无法发布完整的模型权重，敬请各位理解（目前国外也是一样）。Facebook完全开放模型权重之后，本项目会及时更新相关策略。**这里发布的是LoRA权重**，可以理解为原LLaMA模型上的一个“补丁”，两者进行合并即可获得完整版权重。

提醒：以下中文LLaMA/Alpaca LoRA模型无法单独使用，需要搭配原版LLaMA模型<sup>[1]</sup>。请参考本项目给出的[合并模型](#合并模型)步骤重构模型。

### 中文LLaMA模型

中文LLaMA模型在原版的基础上扩充了中文词表，使用了中文纯文本数据进行二次预训练，具体见[训练细节](#训练细节)一节。

| 模型名称          | 类型 |        重构所需模型         | 大小<sup>[2]</sup> |                         LoRA下载地址                         | SHA256<sup>[3]</sup> |
| :---------------- | :--: | :-------------------------: | :----------------: | :----------------------------------------------------------: | :------------------: |
| Chinese-LLaMA-7B  | 通用 | 原版LLaMA-7B<sup>[1]</sup>  |        770M        | [[百度网盘]](https://pan.baidu.com/s/1oORTdpr2TvlkxjpyWtb5Sw?pwd=33hb)</br>[[Google Drive]](https://drive.google.com/file/d/1iQp9T-BHjBjIrFWXq_kIm_cyNmpvv5WN/view?usp=sharing) |  39b86b......fe0e60  |
| Chinese-LLaMA-13B | 通用 | 原版LLaMA-13B<sup>[1]</sup> |         1G         | [[百度网盘]](https://pan.baidu.com/s/1BxFhYhDMipW7LwI58cGmQQ?pwd=ef3t)<br/>[[Google Drive]](https://drive.google.com/file/d/12q9EH4mfKRnoKlbkkhzv1xDwWnroo9VS/view?usp=sharing) |  3d6dee......e5199b  |


### 中文Alpaca模型

中文Alpaca模型在上述中文LLaMA模型的基础上进一步使用了指令数据进行精调，具体见[训练细节](#训练细节)一节。如希望体验类ChatGPT对话交互，请使用Alpaca模型，而不是LLaMA模型。

| 模型名称           |   类型   |        重构所需模型         | 大小<sup>[2]</sup> |                         LoRA下载地址                         | SHA256<sup>[3]</sup> |
| :----------------- | :------: | :-------------------------: | :----------------: | :----------------------------------------------------------: | :------------------: |
| Chinese-Alpaca-7B  | 指令精调 | 原版LLaMA-7B<sup>[1]</sup>  |        790M        | [[百度网盘]](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e)</br>[[Google Drive]](https://drive.google.com/file/d/1JvFhBpekYiueWiUL3AF1TtaWDb3clY5D/view?usp=sharing) |  9bb5b6......ce2d87  |
| Chinese-Alpaca-13B | 指令精调 | 原版LLaMA-13B<sup>[1]</sup> |        1.1G        | [[百度网盘]](https://pan.baidu.com/s/1wYoSF58SnU9k0Lndd5VEYg?pwd=mm8i)<br/>[[Google Drive]](https://drive.google.com/file/d/1gzMc0xMCpXsXmU1uxFlgQ8VRnWNtDjD8/view?usp=share_link) |  45c92e......682d91  |

### Model Hub

可以在🤗Model Hub下载以上所有模型，并且使用[transformers](https://github.com/huggingface/transformers)和[PEFT](https://github.com/huggingface/peft)调用中文LLaMA或Alpaca LoRA模型。以下模型调用名称指的是使用`.from_pretrained()`中指定的模型名称。

| 模型名             |            模型调用名称            |                             链接                             |
| ------------------ | :--------------------------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-7B   |  ziqingyang/chinese-llama-lora-7b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-7b) |
| Chinese-LLaMA-13B  | ziqingyang/chinese-llama-lora-13b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-13b) |
| Chinese-Alpaca-7B  | ziqingyang/chinese-alpaca-lora-7b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b) |
| Chinese-Alpaca-13B | ziqingyang/chinese-alpaca-lora-13b | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b) |

### 脚注及其他说明

**[1]** 原版LLaMA模型需要[去LLaMA项目申请使用](https://github.com/facebookresearch/llama)或参考这个[PR](https://github.com/facebookresearch/llama/pull/73/files)。因版权问题本项目无法提供下载链接。

**[2]** 经过重构后的模型大小比同等量级的原版LLaMA大一些（主要因为扩充了词表）。

**[3]** 下载后务必检查压缩包的SHA256是否一致，完整值请查看[SHA256.md](./SHA256.md)。

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

|                     |   7B   |  13B   |   33B   |   65B   |
| :------------------ | :----: | :----: | :-----: | :-----: |
| 原模型大小（FP16）  | 13 GB  | 24 GB  |  60 GB  | 120 GB  |
| 量化后大小（4-bit） | 3.9 GB | 7.8 GB | 19.5 GB | 38.5 GB |

## 合并模型

为了将LoRA模型与原版LLaMA进行合并以便进行推理或继续训练，目前提供了两种方式：

- 在线转换：适合Google Colab用户，可利用notebook进行在线转换并量化模型

- 手动转换：适合离线方式转换，生成不同格式的模型，以便进行量化或进一步精调

具体步骤请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/模型合并与转换)

## 本地推理与快速部署

本项目中的模型主要支持以下四种推理和部署方式：

- llama.cpp：提供了一种模型量化和在本地CPU上部署方式
- 🤗Transformers：提供原生transformers推理接口，支持CPU/GPU上进行模型推理
- text-generation-webui：提供了一种可实现前端UI界面的部署方式
- LlamaChat：提供了一种macOS下的图形交互界面

相关文档已移至本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/模型推理与部署)


## 系统效果

为了快速评测相关模型的实际表现，本项目在给定相同的prompt的情况下，在一些常见任务上对比测试了本项目的中文Alpaca-7B和中文Alpaca-13B的效果。生成回复具有随机性，受解码超参、随机种子等因素影响。以下相关评测并非绝对严谨，测试结果仅供晾晒参考，欢迎自行体验。详细评测结果请查看[examples/README.md](./examples/README.md)。

*以下测试结果均基于**4-bit量化模型**，理论效果比非量化版本差一些。*

| 测试任务         |                详细样例                | 样例数 | 中文Alpaca-7B | 中文Alpaca-13B |
| ---------------- | :------------------------------------: | :----: | :-----------: | :------------: |
| **💯总平均分**    |                   -                    |  160   |    **49**     |    **👍🏻71**    |
| 知识问答         |            [QA.md](./examples/QA.md)            |   20   |      53       |    **👍🏻77**    |
| 开放式问答       |           [OQA.md](./examples/OQA.md)           |   20   |      64       |    **👍🏻73**    |
| 数值计算、推理   |     [REASONING.md](./examples/REASONING.md)     |   20   |      23       |    **👍🏻50**    |
| 诗词、文学、哲学 |    [LITERATURE.md](./examples/LITERATURE.md)    |   20   |      31       |    **👍🏻54**    |
| 音乐、体育、娱乐 | [ENTERTAINMENT.md](./examples/ENTERTAINMENT.md) |   20   |      36       |    **👍🏻65**    |
| 写信、写文章     |    [GENERATION.md](./examples/GENERATION.md)    |   15   |      65       |    **👍🏻78**    |
| 文本翻译         |   [TRANSLATION.md](./examples/TRANSLATION.md)   |   15   |      63       |    **👍🏻79**    |
| 多轮交互         |      [DIALOGUE.md](./examples/DIALOGUE.md)      |   10   |      80       |    **👍🏻83**    |
| 代码编程         |          [CODE.md](./examples/CODE.md)          |   10   |      27       |    **👍🏻49**    |
| 伦理、拒答       |        [ETHICS.md](./examples/ETHICS.md)        |   10   |      50       |   **👍🏻100**    |


## 训练细节

整个训练流程包括词表扩充、预训练和指令精调三部分，其中训练代码参考了🤗transformers中的[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)和[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)项目中数据集处理的相关部分。

关于相关模型的训练细节，请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/训练细节)

## FAQ

常见问题的解答已移至本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/常见问题)


## 局限性

虽然本项目中的模型相比原版LLaMA和Alpaca在中文理解和生成能力上得到显著提升，但也存在以下局限性：

- 可能会产生不可预测的有害内容以及不符合人类偏好和价值观的内容
- 由于算力和数据问题，相关模型的训练并不充分，中文理解能力有待进一步提升
- 暂时没有在线可互动的demo（注：用户仍然可以自行在本地部署）


## 引用

如果您觉得本项目对您的研究有所帮助或使用了本项目的代码或数据，请参考以下引用（临时）：
```
@misc{chinese-llama-alpaca,
  author = {Yiming Cui and Ziqing Yang},
  title = {Chinese LLaMA and Alpaca LLMs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ymcui/Chinese-LLaMA-Alpaca}},
}
```

## 致谢

本项目基于以下开源项目二次开发，在此对相关项目和研究开发人员表示感谢。

- Facebook LLaMA: https://github.com/facebookresearch/llama
- Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- alpaca-lora by @tloen: https://github.com/tloen/alpaca-lora
- llama.cpp by @ggerganov: https://github.com/ggerganov/llama.cpp
- pCLUE and translation data by @brightmart: https://github.com/brightmart/nlp_chinese_corpus
- LlamaChat by @alexrozanski: https://github.com/alexrozanski/LlamaChat

Episode: Logo中的小羊驼是由[midjourney](http://midjourney.com)自动生成，并由Mac自带的预览工具自动抠出来的。

## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

本项目由个人及协作者业余时间发起并维护，因此无法保证能及时回复解决相应问题。


## 问题反馈
如有问题，请在GitHub Issue中提交。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 重复以及与本项目无关的issue会被[stable-bot](https://github.com/marketplace/stale)处理，敬请谅解。
- 礼貌地提出问题，构建和谐的讨论社区。
