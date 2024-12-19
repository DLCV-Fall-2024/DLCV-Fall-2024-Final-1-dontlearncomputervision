from transformers import TrainingArguments
from transformers import Trainer

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import json
import timm
import open_clip
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoImageProcessor
from transformers import TrainingArguments
from utils import *
# import loralib as lora
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
"""
split can be: “train”, “val”, “test”
You can use the “streaming” argument to avoid downloading whole data
dataset: 30GB,  https://huggingface.co/datasets/ntudlcv/dlcv_2024_final1
how to use?:    https://huggingface.co/docs/datasets/en/stream 
streaming or local?: https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable
load then preprocess: https://huggingface.co/docs/diffusers/en/training/unconditional_training
"""

# model_id = "llava-hf/llava-1.5-7b-hf"
# processor = AutoProcessor.from_pretrained(model_id)
# def preprocess_data(batch):
#     def conver_to_template(task, conversation):
#         """    
#         from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.

#         to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
#         """    
#         prompt=" "
#         # TODO: process three type of template 
#         if 'general' in task:
#             # retrieved from RAG
#             RAG_example_text="'In the traffic scene under observation, multiple road users and traffic elements are present, playing a crucial role in guiding the ego car's driving behavior.\n\nFirstly, there's a red delivery truck identified in the right lane, traveling in the same direction as the ego car. The vehicle seems to be in motion. This indicates that the ego car must keep an eye on the truck since it could potentially stop unexpectedly for a delivery or change its movement pattern, necessitating the ego car to adjust its speed or lane position accordingly.\n\nFurther ahead, in the left lane, a blue public bus is noticed, appearing stationary or moving slowly near a bus stop, also heading in the same direction as the ego car. The stationary nature of the bus suggests that it might either merge into traffic soon or halt to pick up passengers, which could block the lane. Hence, the ego car might need to slow down or change lanes to avoid any disruption.\n\nIn addition to the vehicles, pedestrians are spotted near the bus stop on the sidewalk to the left. Considering their proximity to the street and a potential desire to cross, the ego car must be ready to yield or stop to ensure their safety.\n\nAn important traffic sign is also in view, an overhead directional sign indicating the lane functionalities, specifically designating the leftmost lane for turning left and the other lanes for going straight. This piece of information is vital for the ego car to choose the correct lane based on its intended route.\n\nAlthough the scene includes no visible traffic lights, indicating that at this moment, the ego car's behavior won't be influenced by signal phases like red, green, or yellow lights, this detail simplifies the decision-making process in this specific scenario.\n\nLastly, the presence of plastic water-filled barriers lining the median strip is notable. These barriers serve a significant purpose by ensuring that vehicles stay in their lanes and preventing any unsafe turns across the median, maintaining orderly lane usage and enhancing road safety.\n\nThe absence of traffic cones and other miscellaneous objects in the provided data implies a relatively uncluttered road environment, allowing for straightforward navigation by the ego car, given it considers the movements and potential actions of the described road users and elements.'"
            
#             # origianl prompt
#             prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
#             prompt_context="Here are some similar example: "+ RAG_example_text+" "
#             prompt_command="Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
            
#             prompt=prompt_constrain+prompt_context+prompt_command
#         elif 'suggestion' in task:
#             pass
#         elif 'regional' in task:
#             pass

#         template=f'USER: {prompt} ASSISTANT:'
#         return template
            
#     image=[]
#     text=[]
#     for data in batch:
#         process_text=conver_to_template(data[id], data['conversations'][0]['value'])
#         print(process_text)
#         image.append(data['image'])
#         text.append(process_text)
#     inputs = processor(images=image, text=text,  padding=True, return_tensors='pt')        
#     return inputs

def inference(args):

    """
    Fine-tune a pretrained model: https://huggingface.co/docs/transformers/en/training
    example1: https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/fine_tune_VLM_LlaVa.ipynb#scrollTo=ee1IMXszqWyU
    
    """
    # step: load and prepare model, https://huggingface.co/llava-hf/llava-1.5-7b-hf
    # model_id = "llava-hf/llava-1.5-7b-hf"
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_id, 
    #     torch_dtype=torch.float16, 
    #     low_cpu_mem_usage=True,
    #     load_in_4bit=True
    # )

    # processor = AutoProcessor.from_pretrained(model_id)

    # step: load and prepare dataset
    dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='test')
    dataset_test= dataset_test.with_format("torch")
    dataloader_test=DataLoader(dataset_test, batch_size=1, collate_fn=preprocess_data_prompt_tuning, shuffle=True)
    print(f"{len(dataloader_test)=}")


    # step: inference
    save_json_path="inference/prompt_tuning.json"
    caption_dict={}
    for i, inputs in tqdm(enumerate(dataloader_test)):
        if i==10: break
        print("=======================================")
        # step: model 
        # cap_output = model.generate(**inputs, max_new_tokens=300, do_sample=False)

        # step: parse output 
        # caption=processor.decode(cap_output[0][2:], skip_special_tokens=True)
    #     print(caption)
    #     image_name=str(i)
    #     caption_dict[image_name]=caption.split(':')[-1]
        
    #     # step: saving
    #     if i%3==0 or i==len(dataloader_test)-1:
    #         with open(save_json_path, 'w') as f:
    #             json.dump(caption_dict, f, indent=2)
    # print(f"save at: {save_json_path=}")
    
if __name__=="__main__":
    args=None
    inference(args)
    