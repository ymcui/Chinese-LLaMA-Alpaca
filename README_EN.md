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
    <a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki"><img alt="GitHub wiki" src="https://img.shields.io/badge/Github%20Wiki-v2.2-green"></a>
</p>


Large Language Models (LLM), represented by ChatGPT and GPT-4, have sparked a new wave of research in the field of natural language processing, demonstrating capabilities of Artificial General Intelligence (AGI) and attracting widespread attention from the industry. However, the expensive training and deployment of large language models have posed certain obstacles to building transparent and open academic research.

To promote open research of large models in the Chinese NLP community, this project has open-sourced the **Chinese LLaMA model and the Alpaca large model with instruction fine-tuning**. These models expand the Chinese vocabulary based on the original LLaMA and use Chinese data for secondary pre-training, further enhancing Chinese basic semantic understanding. Additionally, the project uses Chinese instruction data for fine-tuning on the basis of the Chinese LLaMA, significantly improving the model's understanding and execution of instructions. Please refer to our technical report for further details [(Cui, Yang, and Yao, 2023)](https://arxiv.org/abs/2304.08177).

**Main contents of this project:**

- 🚀 Extended Chinese vocabulary on top of original LLaMA with significant encode/decode efficiency
- 🚀 Open-sourced the Chinese LLaMA (general purpose) and Alpaca (instruction-tuned)   (7B, 13B)
- 🚀 Quickly deploy and experience the quantized version of the large model on CPU/GPU of your laptop (personal PC) 
- 🚀 Support [🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LlamaChat](https://github.com/alexrozanski/LlamaChat), etc.

💡 The following image shows the actual experience effect of the 7B version model after local deployment (animation unaccelerated, tested on Apple M1 Max).

![](./pics/screencast.gif)

----

[Multi-modal VLE](https://github.com/iflytek/VLE) | [Chinese MiniRBT](https://github.com/iflytek/MiniRBT) | [Chinese LERT](https://github.com/ymcui/LERT) | [Chinese-English PERT](https://github.com/ymcui/PERT) | [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [Knowledge distillation tool TextBrewer](https://github.com/airaria/TextBrewer) | [Model pruning tool TextPruner](https://github.com/airaria/TextPruner)

## News

**[2023/04/18] Release v2.2: Add LlamaChat support (macOS UI), tokenizer merging scripts, documentations are migrated to GitHub Wiki. Refer to [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.2)**

[2023/04/13] Release v2.1: Add HuggingFace-transformers and text-generation-webui interfances. Refer to [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.1)

[2023/04/07] Release v2.0: Release 13B versions of Chinese LLaMA and Alpaca model. Main upgrades: stronger factuality, better performance on QA, translation and more. Refer to [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.0)

2023/3/31 Release v1.1, major updates: simplification of model merging steps, addition of instruction data crawling script, and important notes about the new version of llama.cpp. See [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v1.1).

2023/3/28  Open-sourcing Chinese LLaMA and Alpaca, currently offering the 7B version for download and experience 

## Content Navigation

| Chapter                                       | Description                                                  |
| --------------------------------------------- | ------------------------------------------------------------ |
| [Download](#Download)                         | Download links for Chinese LLaMA and Alpaca                  |
| [Model Reconstruction](#Model-Reconstruction) | (Important) Explains how to merge downloaded LoRA models with the original LLaMA |
| [Quick Deployment](#Quick=Deployment)         | Steps for quantize and deploy LLMs on personal computers     |
| [Example Results](#Example-Results)           | Examples of the system output                                |
| [Training Details](#Training-Details)         | Introduces the training details of Chinese LLaMA and Alpaca  |
| [FAQ](#FAQ)                                   | Replies to some common questions                             |
| [Limitations](Limitations)                    | Limitations of the models involved in this project           |

## Model Download

### ⚠️ User Notice (Must Read)

The official [LLaMA models released by Facebook prohibits commercial use](https://github.com/facebookresearch/llama), and the official model weights have not been open-sourced (although there are many third-party download links available online). In order to comply with the relevant licenses, it is currently not possible to release the complete model weights. We appreciate your understanding. After Facebook fully opens up the model weights, this project will update its policies accordingly. **What is released here are the LoRA weights**, which can be seen as a "patch" for the original LLaMA model, and the complete weights can be obtained by merging the two.

### Which model should I use?

The following table provides a basic comparison of the Chinese LLaMA and Alpaca models, as well as recommended usage scenarios (including, but not limited to):

| Comparison Item        | Chinese LLaMA                                                | Chinese Alpaca                                               |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Training Method        | Traditional CLM (trained on general corpus)                  | Instruction Fine-tuning (trained on instruction data)        |
| Input Template         | Not required                                                 | Must meet template requirements (llama.cpp/LlamaChat/[inference_hf.py](./scripts/inference_hf.py) are built-in) |
| Suitable Scenarios ✔️   | Text continuation: Given a context, let the model continue writing | 1. Instruction understanding (Q&A, writing, advice, etc.)<br/>2. Multi-turn context understanding (chat, etc.) |
| Unsuitable Scenarios ❌ | Instruction understanding, multi-turn chat, etc.             | Unrestricted free text generation                            |
| llama.cpp              | Use `-p` parameter to specify context                        | Use `-ins` parameter to enable instruction understanding + chat mode |
| text-generation-webui  | Not suitable for chat mode                                   | Use `--cpu` to run without a GPU; if not satisfied with generated content, consider modifying prompt |
| LlamaChat              | Choose "LLaMA" when loading the model                        | Choose "Alpaca" when loading the model                       |
| inference_hf.py        | No additional startup parameters required                    | Add `--with_prompt` parameter when launching                 |
| Known Issues           | If not controlled for termination, it will continue writing until reaching the output length limit. | Current version of the model generates relatively shorter texts, being more concise. |

*Note: If you encounter issues such as low-quality model responses, nonsensical answers, or failure to understand questions, please check whether you are using the correct model and startup parameters for the scenario.*


### Chinese LLaMA

The Chinese LLaMA model has expanded the Chinese vocabulary on the basis of the original version, and used Chinese plain text data for secondary pre-training. For details, see the [Training Details](#Training-Details) section.

| Model             |  Type   | Required Original Model | Size<sup>[2]</sup> |                        Download Links                        | SHA256<sup>[3]</sup> |
| :---------------- | :-----: | :---------------------: | :----------------: | :----------------------------------------------------------: | :------------------: |
| Chinese-LLaMA-7B  | general | LLaMA-7B<sup>[1]</sup>  |        770M        | [[BaiduDisk]](https://pan.baidu.com/s/1oORTdpr2TvlkxjpyWtb5Sw?pwd=33hb)</br>[[Google Drive]](https://drive.google.com/file/d/1iQp9T-BHjBjIrFWXq_kIm_cyNmpvv5WN/view?usp=sharing) |  39b86b......fe0e60  |
| Chinese-LLaMA-13B | general | LLaMA-13B<sup>[1]</sup> |         1G         | [[BaiduDisk]](https://pan.baidu.com/s/1BxFhYhDMipW7LwI58cGmQQ?pwd=ef3t)<br/>[[Google Drive]](https://drive.google.com/file/d/12q9EH4mfKRnoKlbkkhzv1xDwWnroo9VS/view?usp=sharing) |  3d6dee......e5199b  |

### Chinese Alpaca

The Chinese Alpaca model further uses instruction data for fine-tuning on the basis of the above-mentioned Chinese LLaMA model. For details, see the [Training Details](#Training-Details) section.

**⚠️ Please use Alpaca model if you want to try ChatGPT-like model.**

| Model              |        Type        | Required Original Model | Size<sup>[2]</sup> |                        Download Links                        | SHA256<sup>[3]</sup> |
| :----------------- | :----------------: | :---------------------: | :----------------: | :----------------------------------------------------------: | :------------------: |
| Chinese-Alpaca-7B  | Instruction Tuning | LLaMA-7B<sup>[1]</sup>  |        790M        | [[BaiduDisk]](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e)</br>[[Google Drive]](https://drive.google.com/file/d/1JvFhBpekYiueWiUL3AF1TtaWDb3clY5D/view?usp=sharing) |  9bb5b6......ce2d87  |
| Chinese-Alpaca-13B | Instruction Tuning | LLaMA-13B<sup>[1]</sup> |        1.1G        | [[BaiduDisk]](https://pan.baidu.com/s/1wYoSF58SnU9k0Lndd5VEYg?pwd=mm8i)<br/>[[Google Drive]](https://drive.google.com/file/d/1gzMc0xMCpXsXmU1uxFlgQ8VRnWNtDjD8/view?usp=share_link) |  45c92e......682d91  |

### Model Hub

You can download all the above models in 🤗Model Hub, and use [🤗transformers](https://github.com/huggingface/transformers) and [🤗PEFT](https://github.com/huggingface/peft) to call Chinese LLaMA or the Alpaca LoRA model.

| Model              |             MODEL_NAME             |                             Link                             |
| ------------------ | :--------------------------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-7B   |  ziqingyang/chinese-llama-lora-7b  | [Link](https://huggingface.co/ziqingyang/chinese-llama-lora-7b) |
| Chinese-LLaMA-13B  | ziqingyang/chinese-llama-lora-13b  | [Link](https://huggingface.co/ziqingyang/chinese-llama-lora-13b) |
| Chinese-Alpaca-7B  | ziqingyang/chinese-alpaca-lora-7b  | [Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b) |
| Chinese-Alpaca-13B | ziqingyang/chinese-alpaca-lora-13b | [Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b) |

### Footnote and Others

**[1]** The original LLaMA model needs to be applied for use in [Facebook-LLaMA](https://github.com/facebookresearch/llama) or refer to this [PR](https://github.com/facebookresearch/llama/pull/73/files). Due to copyright issues, this project cannot provide downloads, and we ask for your understanding.

**[2]** The reconstructed model is slightly larger than the original LLaMA (due to the expanded vocabulary); the 7B model is about 13G+.

**[3]** After downloading, be sure to check whether the SHA256 of the ZIP file is consistent; for the full value, please see [SHA256.md](./SHA256.md).

The file directory inside the ZIP file is as follows (using Chinese-LLaMA as an example):

```
chinese_llama_lora_7b/
  - adapter_config.json       # LoRA weight configuration file
  - adapter_model.bin         # LoRA weight file
  - special_tokens_map.json   # special_tokens_map file
  - tokenizer_config.json     # tokenizer configuration file
  - tokenizer.model           # tokenizer file
```

The following is the size of each original model and 4-bit quantization. When converting the corresponding model, make sure that the machine has enough memory and disk space (minimum requirements):

|                    |   7B   |  13B   |   33B   |   65B   |
| :----------------- | :----: | :----: | :-----: | :-----: |
| Original（FP16）   | 13 GB  | 24 GB  |  60 GB  | 120 GB  |
| Quantized（4-bit） | 3.9 GB | 7.8 GB | 19.5 GB | 38.5 GB |

## Model Reconstruction

In order to merge the LoRA model with the original LLaMA for further tuning or inference, two methods are currently provided:

- Online conversion: suitable for Google Colab users, can use notebook for online conversion and model quantization.
- Manual conversion: suitable for offline conversion, generates models in different formats for quantization or further fine-tuning.

Related documentation has been moved to the project's >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Model-Reconstruction).

## Quick Deployment

We mainly provide the following three ways for inference and local deployment.

- llama.cpp：a tool for quantizing model and deploying on local CPU
- 🤗Transformers：original transformers inference method, support CPU/GPU
- text-generation-webui：a tool for deploying model as a web UI
- LlamaChat: a macOS app that allows you to chat with LLaMA, Alpaca, etc.

Related documentation has been moved to the project's >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Model-Inference-and-Deployment).


## System Performance

In order to quickly evaluate the actual performance of related models, this project compared the effects of Chinese Alpaca-7B and Chinese Alpaca-13B on some common tasks given the same prompt. The test models are all **4-bit quantized models**, and the theoretical effect is worse than the non-quantized version. Reply generation is random and is affected by factors such as decoding hyperparameters and random seeds. The following related evaluations are not absolutely rigorous, and the test results are for reference only. Welcome to experience it yourself. For detailed evaluation results, please see [examples/README.md](./examples/README.md)

| Task                           |                Samples                 |  #   | Chinese Alpaca-7B | Chinese Alpaca-13B |
| ------------------------------ | :------------------------------------: | :--: | :---------------: | :----------------: |
| **💯 Overall**                  |                   -                    | 160  |      **49**       |      **👍🏻71**      |
| Question Answering             |       [QA.md](./examples/QA.md)        |  20  |        53         |      **👍🏻77**      |
| Open QA                        |           [OQA.md](./examples/OQA.md)           |  20  |        64         |      **👍🏻73**      |
| Computation, Reasoning         |     [REASONING.md](./examples/REASONING.md)     |  20  |        23         |      **👍🏻50**      |
| Poetry, Literature, Philosophy |    [LITERATURE.md](./examples/LITERATURE.md)    |  20  |        31         |      **👍🏻54**      |
| Music, Sports, Entertainment   | [ENTERTAINMENT.md](./examples/ENTERTAINMENT.md) |  20  |        36         |      **👍🏻65**      |
| Letters and Articles           |    [GENERATION.md](./examples/GENERATION.md)    |  15  |        65         |      **👍🏻78**      |
| Translation                    |   [TRANSLATION.md](./examples/TRANSLATION.md)   |  15  |        63         |      **👍🏻79**      |
| Multi-turn Dialogue            |      [DIALOGUE.md](./examples/DIALOGUE.md)      |  10  |        80         |      **👍🏻83**      |
| Coding                         |          [CODE.md](./examples/CODE.md)          |  10  |        27         |      **👍🏻49**      |
| Ethics                         |        [ETHICS.md](./examples/ETHICS.md)        |  10  |        50         |     **👍🏻100**      |


## Training Details

The entire training process includes three parts: vocabulary expansion, pre-training, and instruction fine-tuning. Please refer to [merge_tokenizers.py](scripts/merge_tokenizers.py) for vocabulary expansion; refer to [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) in 🤗transformers and the relevant parts of dataset processing in the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) project for pre-training and self-instruct fine-tuning.

Please refer to our  >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Training-Details).


## FAQ

Please refer to our  >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/FAQ).

## Limitations

Although the models in this project have significantly improved Chinese understanding and generation capabilities compared to the original LLaMA and Alpaca, there are also the following limitations:

- It may produce unpredictable harmful content and content that does not conform to human preferences and values.
- Due to computing power and data issues, the training of the related models is not sufficient, and the Chinese understanding ability needs to be further improved.
- There is no online interactive demo available for now (Note: users can still deploy it locally themselves).

## Citation

If you find the model, data, code in our project useful, please consider cite our work as follows: 

```
@article{chinese-llama-alpaca,
      title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca}, 
      author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
      journal={arXiv preprint arXiv:2304.08177},
      url={https://arxiv.org/abs/2304.08177},
      year={2023}
}
```

## Acknowledgements

This project is based on the following open-source projects for secondary development, and we would like to express our gratitude to the related projects and research and development personnel.

- Facebook LLaMA: https://github.com/facebookresearch/llama
- Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- alpaca-lora by @tloen: https://github.com/tloen/alpaca-lora
- llama.cpp by @ggerganov: https://github.com/ggerganov/llama.cpp
- pCLUE and translation data by @brightmart: https://github.com/brightmart/nlp_chinese_corpus
- LlamaChat by @alexrozanski: https://github.com/alexrozanski/LlamaChat

Episode: The Alpaca Logo is generated by [midjourney](http://midjourney.com) and is automatically extracted by Preview in MacOS.

## Disclaimer

**The resources related to this project are for academic research purposes only and are strictly prohibited for commercial use.** When using parts involving third-party code, please strictly follow the corresponding open-source agreements. The content generated by the model is affected by factors such as model calculation, randomness, and quantization accuracy loss. This project cannot guarantee its accuracy. For any content output by the model, this project does not assume any legal responsibility and does not assume responsibility for any losses that may result from the use of related resources and output results.

This project is initiated and maintained by individuals and collaborators in their spare time, so we cannot guarantee a timely response to resolving relevant issues.

## Feedback

If you have any questions, please submit them in GitHub Issues.

- Before submitting a question, please check if the FAQ can solve the problem and consult past issues to see if they can help.
- Duplicate and unrelated issues will be handled by [stable-bot](https://github.com/marketplace/stale); please understand.
- Raise questions politely and help build a harmonious discussion community.
