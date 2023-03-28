[**中文**](./README.md) | [**English**](./README_EN.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="600"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ymcui/Chinese-LLaMA-Alpaca">
</p>


以ChatGPT、GPT-4等为代表的大语言模型（Large Language Model, LLM）掀起了新一轮自然语言处理领域的研究浪潮，展现出了类通用人工智能（AGI）的能力，受到业界广泛关注。然而，由于大语言模型的训练和部署都极为昂贵，为构建透明且开放的学术研究造成了一定的阻碍。

为了促进大模型在中文NLP社区的开放研究，本项目开源了**中文LLaMA模型和经过指令精调的Alpaca大模型**。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，在中文LLaMA的基础上，本项目使用了中文指令数据进行指令精调，显著提升了模型对指令的理解和执行能力。

***声明：本项目相关资源仅供学术研究使用。***

**本项目主要内容：**

- 🚀 开源了经过中文文本数据预训练的中文LLaMA大模型
- 🚀 开源了进一步经过指令精调的中文Alpaca大模型
- 🚀 快速地使用笔记本电脑（个人PC）本地部署和体验量化版大模型

💡 下图给出了7B版本模型本地化部署后的实际体验效果（动画未经加速，Apple M1 Max下实测）。

![](./pics/screencast.gif)

----

[多模态VLE](https://github.com/iflytek/VLE) | [中文MiniRBT](https://github.com/iflytek/MiniRBT) | [中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner)

## 新闻

**2023/3/28 正式开源中文LLaMA、Alpaca大模型，目前提供7B版本下载体验。**

## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [模型下载](#模型下载)         | 中文LLaMA、Alpaca大模型下载地址                |
| [合并模型](#合并模型) | （重要）介绍如何将下载的LoRA模型与原版LLaMA合并 |
| [本地快速部署](#本地快速部署)       | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [系统效果](#系统效果) | 介绍了部分场景和任务下的使用体验效果             |
| [训练细节](#训练细节) | 介绍了中文LLaMA、Alpaca大模型的训练细节 |
| [局限性](#局限性) | 本项目涉及模型的局限性 |


## 模型下载

### ⚠️ 用户须知（必读）

Facebook官方发布的[LLaMA模型禁止商用](https://github.com/facebookresearch/llama)，并且官方没有正式开源模型权重（虽然网上已经有很多第三方的下载地址）。为了遵循相应的许可，目前暂时无法发布完整的模型权重，敬请各位理解（目前国外也是一样）。Facebook完全开放模型权重之后，本项目会及时更新相关策略。**这里发布的是LoRA权重**，可以理解为原LLaMA模型上的一个“补丁”，两者进行合并即可获得完整版权重。

### 下载地址

注意：以下模型无法直接使用，必须按照本项目给出的[合并模型](#合并模型)步骤重构模型。

| 模型名称          |   类型   |       重构所需基模型       | 大小<sup>[2]</sup> |                         LoRA下载地址                         | SHA256<sup>[3]</sup> |
| :---------------- | :------: | :------------------------: | :----------------: | :----------------------------------------------------------: | :------------------: |
| Chinese-LLaMA-7B  |   通用   | 原版LLaMA-7B<sup>[1]</sup> |        770M        | [[网盘地址]](https://pan.baidu.com/s/1oORTdpr2TvlkxjpyWtb5Sw?pwd=33hb)</br>（密码: 33hb） |  39b86b......fe0e60  |
| Chinese-Alpaca-7B | 指令精调 | 原版LLaMA-7B<sup>[1]</sup> |        790M        | [[网盘地址]](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e)</br>（密码：923e） |  9bb5b6......ce2d87  |

**[1]** 原版LLaMA模型需要在[Facebook-LLaMA](https://github.com/facebookresearch/llama)中申请使用，或参考这个[PR](https://github.com/facebookresearch/llama/pull/73/files)。由于版权问题本项目无法提供下载，敬请谅解。

**[2]** 经过重构后的模型大小比原版LLaMA稍大（因为扩充了词表），7B模型约为13G+。

**[3]** 下载后务必检查压缩包的SHA256是否一致，完整值请查看[SHA256.md](./SHA256.md)。

压缩包内文件目录如下（以Chinese-LLaMA为例）：

```
chinese_llama_lora_7b/
  - adapter_config.json		# LoRA权重配置文件
  - adapter_model.bin		# LoRA权重文件
  - special_tokens_map.json	# special_tokens_map文件
  - tokenizer_config.json	# tokenizer配置文件
  - tokenizer.model		# tokenizer文件
```

## 合并模型

### 准备工作

合并前务必确认基模型和LoRA模型补丁的SHA256是否和下表所列SHA256值一致，否则无法进行合并操作。

1. 原版LLaMA包含以下文件：`tokenizer.model`、`tokenizer_checklist.chk`、`consolidated.00.pth`、`params.json`
2. 其中，权重文件`consolidated.00.pth`的SHA256: `700df0d3013b703a806d2ae7f1bfb8e59814e3d06ae78be0c66368a50059f33d`


### Step 1: 将原版LLaMA模型转换为HF格式

请使用[最新版🤗transformers](https://huggingface.co/docs/transformers/installation#install-from-source)提供的脚本[convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)，将原版LLaMA模型转换为HuggingFace格式。*本项目不对使用第三方（非Facebook官方）权重的合规性和正确性负责，例如HuggingFace模型库中的`decapoda-research/llama-7b-hf`（use at your own risk）。*

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

### Step 2: 对模型进行中文词表扩充

使用本项目中的`scripts/extend_llama_with_zh_vocab.py`对原版LLaMA模型扩充中文词表，请执行以下命令：

```bash
python scripts/extend_llama_with_zh_vocab.py \
    --llama_model path_to_original_llama_model_hf \ 
    --tokenizer path_to_chinese_tokenzier \
    --output_dir output_dir
```

其中：

- `--llama_model`参数：存放HF格式的LLaMA模型权重和配置文件的目录
- `--tokenizer`参数：Chinese LLaMA或者Alpaca模型的`tokenizer.model`文件所在目录，请指向在[上一节](#下载地址)里下载的LoRA模型压缩包解压后文件所在目录
- `--output_dir`参数：扩充词表后的模型存放位置


### Step 3: 合并LoRA权重，生成全量模型权重

使用`scripts/export_state_dict_checkpoint.py`脚本，将Step 2生成的中文词表扩充的模型和LoRA权重进行合并，生成全量模型权重。请执行以下命令：

```bash
python scripts/export_state_dict_ckeckpoint.py \
    --base_model path_to_zh_vocab_extended_model_hf \
    --lora_model path_to_chinese_lora \
    --model_type pretrained
```

其中：

- `--base_model`参数：经过中文词表扩充的模型（Step 2生成）
- `--lora_model`参数：在[上一节](#下载地址)里下载的LoRA模型压缩包解压后文件所在目录
- `--model_type`参数：指定`pretrained`表示转换LLaMA，指定`finetuned`表示转换指令精调的Alpaca

*（可选）如有需要，可自行按照Step 1中的脚本将`.pth`文件转换为HuggingFace格式。*

## 本地快速部署

研究社区已经有很多优秀的模型量化和部署工具帮助用户**很方便地将大模型在自己的电脑上进行本地部署**。接下来以[llama.cpp工具](https://github.com/ggerganov/llama.cpp)为例，介绍MacOS和Linux系统中，将模型进行量化并部署的详细步骤。Windows则可能需要cmake等编译工具的安装，可参考[alpaca.cpp](https://github.com/antimatter15/alpaca.cpp#building-from-source-windows)中的步骤。**本地快速部署体验推荐使用经过指令精调的Alpaca模型。**

运行前请确保：

1. 模型量化过程需要将未量化模型全部载入内存，请确保有足够可用内存（7B版本需要13G以上）
2. 加载使用量化后的模型时（例如7B版本），确保本机可用内存大于4-6G（受上下文长度影响）
3. 系统应有`make`（MacOS/Linux自带）或`cmake`（Windows需自行安装）编译工具
4. 推荐使用Python 3.9或3.10编译运行[llama.cpp工具](https://github.com/ggerganov/llama.cpp)（因为`sentencepiece`还不支持3.11）

### Step 1: 克隆和编译llama.cpp

运行以下命令对llama.cpp项目进行编译，生成`./main`和`./quantize`二进制文件。

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

###  Step 2: 生成量化版本模型

根据需要转换的模型类型（LLaMA或Alpaca），将下载的LoRA模型压缩包中的`tokenizer.*`文件放入`zh-models`目录下，将本项目根目录中的`params.json`和[合并模型](#合并模型)中最后一步获取的`.pth`模型文件放入`zh-models/7B`目录下。请注意`.pth`模型文件和`tokenizer.model`是对应的，LLaMA和Alpaca的`tokenizer.model`不可混用。目录结构类似：

```
llama.cpp/zh-models/
   - 7B/
     - consolidated.00.pth
     - params.json
   - tokenizer.model
```

将上述`.pth`模型权重转换为ggml的FP16格式，生成文件路径为`zh-models/7B/ggml-model-f16.bin`。

```
python convert-pth-to-ggml.py zh-models/7B/ 1
```

进一步对FP16模型进行Q4量化，生成量化模型文件路径为`zh-models/7B/ggml-model-q4_0.bin`。

```
python quantize.py 7B -m zh-models
```

### Step 3: 加载并启动模型

运行`./main`二进制文件，`-m`命令指定Q4量化模型（也可加载ggml-FP16的模型）。以下是解码参数示例：

```
./main -m zh-models/7B/ggml-model-q4_0.bin --color -f ./prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
```
在提示符 `>` 之后输入你的prompt，`command+c`中断输出。如需查看帮助和参数说明，请执行`./main -h`命令。

简要介绍几个重要参数：

```
-c 控制上下文的长度，值越大越能参考更长的对话历史
-ins 启动类ChatGPT的对话交流模式
-n 控制回复生成的最大长度
--repeat_penalty 控制生成回复中对重复文本的惩罚力度
--temp 温度系数，值越低回复的随机性越小，反之越大
--top_p, top_k 控制采样的相关参数
```

## 系统效果

TBA

## 训练细节

整个训练流程包括词表扩充、预训练和指令精调三部分，其中训练代码参考了🤗transformers中的[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)和[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)项目中数据集处理的相关部分。

### 准备工作：词表扩充

在通用中文语料上训练了基于[sentencepiece](https://github.com/google/sentencepiece)的20K词表并与原版LLaMA的32K词表进行合并，排除重复的token后，得到的扩充后词表大小为49953。

### 预训练

在预训练阶段，使用通用中文语料（与[中文BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)、[MacBERT](https://github.com/ymcui/MacBERT)、[LERT](https://github.com/ymcui/PERT)、[PERT](https://github.com/ymcui/PERT)中使用的语料一致）在原版LLaMA权重的基础上进一步进行预训练。该过程又分为两个阶段：

1. 第一阶段：固定模型transformer部分的参数，仅训练embedding，在尽量不干扰原模型的情况下适配新增的中文词向量。

2. 第二阶段：使用LoRA技术，为模型添加LoRA权重（adapter），训练embedding的同时也更新LoRA参数。

### 指令精调

指令精调阶段的任务形式基本与[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)相同。训练方案同样采用了LoRA进行高效精调，并进一步增加了可训练参数数量。

### 训练数据与超参

在指令精调阶段使用了约200万条数据，其中包括约50万条中英翻译数据、30万条的经过清洗的pCLUE的数据、10万条的原版Stanford Alpaca数据与其中文翻译版本，以及从多种渠道爬取的self-instruct数据。

训练过程的主要超参如下：

| Hyperparameters          | 预训练-第一阶段 | 预训练-第二阶段 | 指令精调 |
| :----------------------- | :-------------: | :-------------: | :------: |
| Batch Size               |      1024       |      1024       |   512    |
| Initial Learning Rate    |      2e-4       |      1e-4       |   1e-4   |
| Training Steps           |       3K        |       6K        |  6K-10K  |
| Max Length               |       512       |       512       |   512    |
| Trainable Parameters (%) |      2.97%      |      6.06%      |  6.22%   |

## 局限性

虽然本项目中的模型相比原版LLaMA和Alpaca在中文理解和生成能力上得到显著提升，但也存在以下局限性：

- 可能会产生不可预测的有害内容以及不符合人类偏好和价值观的内容
- 由于算力和数据问题，相关模型的训练并不充分，中文理解能力有待进一步提升
- 暂时没有在线可互动的demo（注：用户仍然可以自行在本地部署）


## 致谢

本项目基于以下开源项目二次开发，在此对相关项目和研究开发人员表示感谢。

- Facebook LLaMA: https://github.com/facebookresearch/llama
- Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- alpaca-lora by @tloen: https://github.com/tloen/alpaca-lora
- llama.cpp by @ggerganov: https://github.com/ggerganov/llama.cpp

## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目无法对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

本项目由个人及协作者业余时间发起并维护，因此无法保证能及时回复解决相应问题。


## 问题反馈
如有问题，请在GitHub Issue中提交。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 重复以及与本项目无关的issue会被[stable-bot](stale · GitHub Marketplace)处理，敬请谅解。
- 礼貌地提出问题，构建和谐的讨论社区。
