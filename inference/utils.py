from transformers import AutoProcessor
import numpy as np
import faiss
import json


def retrieve_similar_images(query_embedding, index, top_k=3):
    """
    input: a list of query_embeddings
    output: 
        - distances: similarity of images 
        - indices: 2D array, n*k, n= number of query_embeddings, k= top_k values
    """
    # query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
    query_vectors=np.array(query_embedding)
    # query_embedding=query_embedding.detach().numpy().astype(np.float32)
    # query_vectors = np.array([query_embedding])
    
    distances, indices = index.search(query_vectors, top_k)
    
    # retrieved_images = [image_paths[int(idx)] for idx in indices[0]]
    return distances, indices 



model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
def preprocess_data(batch):
    def conver_to_template(conversation):
        """    
        from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
        to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
        """    
        template=f'USER: You are a traffic analysis assistant.  {conversation} ASSISTANT:'
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

# TODO: adding RAG into data processing 
# vector_db_path="test/RAG/vector.index"
# metadata_path="test/RAG/vector.index.json"

# RAG_index=faiss.read_index("")
# with open(metadata_path,'r') as f:
#     metadata= json.load(f)

def preprocess_data_prompt_tuning(batch):
    def conver_to_template(task, conversation, retrieve_conversation):
        """    
        from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
        to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
        """    
        prompt=""
        RAG_example_text=retrieve_conversation            
        prompt_charactor="You are a traffic analysis assistant. "
        prompt_context="Here is a similar example for the following task:\n"+ RAG_example_text
        # TODO: process three type of template 
        if 'general' in task:
            # retrieved from RAG
            # origianl prompt
            prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above.\n"
            prompt_command="\nPlease describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
            prompt=prompt_charactor+prompt_constrain+prompt_context+prompt_command
            
        elif 'suggestion' in task:
            """
            There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
            """
            prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above.\n"
            prompt_command= "\nPlease provide driving suggestions for the ego car based on the current scene."
            prompt=prompt_charactor+prompt_constrain+prompt_context+prompt_command
            
        elif 'regional' in task:
            """
            Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
            """
            # prompt_context="Here is a similar example for the following task:\n"+ RAG_example_text
            prompt_command="\nPlease describe the object inside the red rectangle in the image and explain why it affect ego car driving."
            prompt=prompt_charactor+prompt_context+prompt_command
            
        template=f'USER: <image>\n{prompt} ASSISTANT:'
        return template
    # ==========================================================================================================================================================================================
    image=[]
    text=[]
    image_names=[]
    RAG=''
    # TODO: RAG retrieve data
    # step RAG : preprocess images into concatenated embeddings using Semantic SAM and Depth anything, one image at a time
    # concated_embedding_list=[embed_fn(image) for image in data['image']]
    # distance, indices= retrieve_similar_images(concated_embedding_list, RAG_index, top_k=3)
    # retrieved_data_list=[metadata[str(i[0])]['conversations'] for i in indices]
    
    for data in batch:
        process_text=conver_to_template(data['id'], data['conversations'][0]['value'], )
        # print(process_text)
        image.append(data['image'])
        text.append(process_text)
        image_names.append(data['id'])
    inputs = processor(images=image, text=text,  padding=True, return_tensors='pt')        
    # inputs['ids']=image_name
    # print(inputs['pixel_values'].shape)
    return inputs, image_names

if __name__=='__main__':
    from datasets import load_dataset
    dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='test')
    