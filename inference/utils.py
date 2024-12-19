

from transformers import AutoProcessor
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
def preprocess_data(batch):
    def conver_to_template(conversation):
        """    
        from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
        to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
        """    
        template=f'USER: {conversation} ASSISTANT:'
        return template
    image=[]
    text=[]
    image_names=[]
    for data in batch:
        process_text=conver_to_template(data['conversations'][0]['value'])
        image.append(data['image'])
        text.append(process_text)
        image_names.append(data['id'])
    inputs = processor(images=image, text=text,  padding=True, return_tensors='pt')        
    # inputs['ids']=image_name
    # print(inputs['pixel_values'].shape)
    return inputs, image_names

def preprocess_data_prompt_tuning(batch):
    def conver_to_template(task, conversation, retrieve_conversation):
        """    
        from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
        to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
        """    
        prompt=""
        RAG_example_text=retrieve_conversation            
        prompt_context="Here is a similar example for the following task:\n"+ RAG_example_text
        # TODO: process three type of template 
        if 'general' in task:
            # retrieved from RAG
            # origianl prompt
            prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above.\n"
            prompt_command="\nPlease describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
            prompt=prompt_constrain+prompt_context+prompt_command
            
        elif 'suggestion' in task:
            """
            There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
            """
            prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above.\n"
            prompt_command= "\nPlease provide driving suggestions for the ego car based on the current scene."
            prompt=prompt_constrain+prompt_context+prompt_command
            
        elif 'regional' in task:
            """
            Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
            """
            # prompt_context="Here is a similar example for the following task:\n"+ RAG_example_text
            prompt_command="\nPlease describe the object inside the red rectangle in the image and explain why it affect ego car driving."
            prompt=prompt_context+prompt_command
            
        template=f'USER: <image>\n{prompt} ASSISTANT:'
        return template
            
    image=[]
    text=[]
    # TODO: RAG retrieve data
    RAG=""

    for data in batch:
        process_text=conver_to_template(data['id'], data['conversations'][0]['value'],"RAG top 1 conversation")
        print(process_text)
        image.append(data['image'])
        text.append(process_text)
    inputs = processor(images=image, text=text,  padding=True, return_tensors='pt')        
    return inputs
