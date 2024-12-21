from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from utils import *
"""
split can be: “train”, “val”, “test”
You can use the “streaming” argument to avoid downloading whole data
dataset: 30GB,  https://huggingface.co/datasets/ntudlcv/dlcv_2024_final1
how to use?:    https://huggingface.co/docs/datasets/en/stream 
streaming or local?: https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable
load then preprocess: https://huggingface.co/docs/diffusers/en/training/unconditional_training
"""

def inference(args):

    """
    Fine-tune a pretrained model: https://huggingface.co/docs/transformers/en/training
    example1: https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/fine_tune_VLM_LlaVa.ipynb#scrollTo=ee1IMXszqWyU
    
    """
    model_id = args.model
    save_json_path=args.save_json_path
    print("using ", model_id)
    print("save at ", save_json_path)

    # step: load and prepare model, https://huggingface.co/llava-hf/llava-1.5-7b-hf
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=bnb_config
        # torch_dtype=torch.float16, 
        # low_cpu_mem_usage=True,
        # load_in_4bit=True
    )
    model.eval()
    # processor = AutoProcessor.from_pretrained(model_id)

    # step: load and prepare dataset
    dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='test')
    dataset_test= dataset_test.with_format("torch")
    
    preprocess= preprocess_data if args.use_prompt_tuning == False else preprocess_data_prompt_tuning
    dataloader_test=DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=preprocess, shuffle=False)
    print(f"{len(dataloader_test)=}")


    # step: inference
    caption_dict={}
    # TODO: case, if json file not exist
    if os.path.exists(save_json_path):
        with open(save_json_path, "r") as f:  # reading a file
            caption_dict = json.load(f)  # deserialization

    for _, (inputs, image_names) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        # print(inputs)
        # step: model 
        inputs=inputs.to(0)
        cap_output = model.generate(**inputs, max_new_tokens=400, do_sample=False)

        # step: saving
        for j, image_name in enumerate(image_names):
            caption=processor.decode(cap_output[j][2:], skip_special_tokens=True) # step: decode and parse output 
            image_name=image_names[j]
            caption_dict[image_name]=caption.split(':')[-1]
            with open(save_json_path, 'w') as f:
                json.dump(caption_dict, f, indent=2)
    print(f"save at: {save_json_path=}")

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="Image root")
    parser.add_argument("--save_json_path", default="inference/baseline.json", help="output json file")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--use_prompt_tuning", action='store_true', help="use_prompt_tuning or not")
    args = parser.parse_args()
    
    inference(args)
    