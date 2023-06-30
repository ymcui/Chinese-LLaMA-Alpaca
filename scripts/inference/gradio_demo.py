import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)
import gradio as gr
import argparse
import os


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_model',
    default=None,
    type=str,
    required=True,
    help='Base model path')
parser.add_argument('--lora_model', default=None, type=str,
                    help="If None, perform inference on the base model")
parser.add_argument(
    '--tokenizer_path',
    default=None,
    type=str,
    help='If None, lora model path or base model path will be used')
parser.add_argument(
    '--gpus',
    default="0",
    type=str,
    help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
parser.add_argument('--share', default=True, help='Share gradio domain name')
parser.add_argument('--port', default=19324, type=int, help='Port of gradio demo')
parser.add_argument(
    '--max_memory',
    default=256,
    type=int,
    help='Maximum input prompt length, if exceeded model will receive prompt[-max_memory:]')
parser.add_argument(
    '--load_in_8bit',
    action='store_true',
    help='Use 8 bit quantified model')
parser.add_argument(
    '--only_cpu',
    action='store_true',
    help='Only use CPU for inference')
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""


# Set CUDA devices if available
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# Peft library can only import after setting CUDA devices
from peft import PeftModel


# Set up the required components: model and tokenizer

def setup():
    global tokenizer, model, device, share, port, max_memory
    max_memory = args.max_memory
    port = args.port
    share = args.share
    load_in_8bit = args.load_in_8bit
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_model,
            torch_dtype=load_type,
            device_map='auto',
        )
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()

    model.eval()


# Reset the user input
def reset_user_input():
    return gr.update(value='')


# Reset the state
def reset_state():
    return []


# Generate the prompt for the input of LM model
def generate_prompt(instruction):
    return f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
{instruction}
 """


# User interaction function for chat
def user(user_message, history):
    return gr.update(value="", interactive=False), history + \
        [[user_message, None]]


# Perform prediction based on the user input and history
@torch.no_grad()
def predict(
    history,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.0
):
    history[-1][1] = ""
    if len(history) != 0:
        input = "".join(["### Instruction:\n" +
                         i[0] +
                         "\n\n" +
                         "### Response: " +
                         i[1] +
                         ("\n\n" if i[1] != "" else "") for i in history])
        if len(input) > max_memory:
            input = input[-max_memory:]
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    original_size = len(input_ids[0])
    logits_processor = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=temperature),
            RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty)),
            TopPLogitsWarper(top_p=top_p),
            TopKLogitsWarper(top_k=top_k)
        ])
    eos_token_id = tokenizer.eos_token_id
    while True:
        logits = model(input_ids).logits
        logits = logits[:, -1, :]
        logits = logits_processor(input_ids, logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1) \
            if do_sample else torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        if next_token_id == eos_token_id:
            break
        tokens_previous  = tokenizer.decode(
            input_ids[0], skip_special_tokens=True)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
        tokens = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_tokens = tokens[len(tokens_previous) :]
        history[-1][1] += new_tokens
        yield history
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
        if len(input_ids[0]) >= original_size + max_new_tokens:
            break


# Call the setup function to initialize the components
setup()


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Chinese LLaMA & Alpaca LLM</h1>""")
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    local_banner_path = f'{current_file_path}/../../pics/small_banner.png'
    github_banner_path = 'https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/pics/small_banner.png'
    banner = ''
    if os.path.exists(local_banner_path):
        banner = local_banner_path
    else:
        banner = github_banner_path
    gr.Image(banner, label='Chinese LLaMA & Alpaca LLM')
    gr.Markdown("> 为了促进大模型在中文NLP社区的开放研究，本项目开源了中文LLaMA模型和指令精调的Alpaca大模型。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力")
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Shift + Enter发送消息...",
                    lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_token = gr.Slider(
                0,
                4096,
                value=128,
                step=1.0,
                label="Maximum New Token Length",
                interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0,
                1,
                value=0.7,
                step=0.01,
                label="Temperature",
                interactive=True)
            top_k = gr.Slider(1, 40, value=40, step=1,
                              label="Top K", interactive=True)
            do_sample = gr.Checkbox(
                value=True,
                label="Do Sample",
                info="use random sample strategy",
                interactive=True)
            repetition_penalty = gr.Slider(
                1.0,
                3.0,
                value=1.1,
                step=0.1,
                label="Repetition Penalty",
                interactive=True)

    params = [user_input, chatbot]
    predict_params = [
        chatbot,
        max_new_token,
        top_p,
        temperature,
        top_k,
        do_sample,
        repetition_penalty]

    submitBtn.click(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    user_input.submit(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)


# Launch the Gradio interface
demo.queue().launch(
    share=share,
    inbrowser=True,
    server_name='0.0.0.0',
    server_port=port)
