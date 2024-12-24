from transformers import AutoProcessor
import numpy as np
import faiss
import json

# TODO: embed a pil image into a concated embedding using other model(sam, clip, ...) 
def embed_fn(input_image):
    """
    input: a pil image
    output: a embedding
    """
    pass

def retrieve_similar_images(query_embedding, index, top_k=3):
    """
    input: one query_embedding (assume it is a tensor on gpu)
    output: 
        - distances: similarity of images 
        - indices: return the top-k indice of the 2D array, n*k, n= number of query_embeddings, k= top_k values
    """
    # query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
    query_embedding=query_embedding.detach().numpy().astype(np.float32)
    query_vectors = np.array([query_embedding])
    
    distances, indices = index.search(query_vectors, top_k)
    # print(indices)
    # retrieved_images = [image_paths[int(idx)] for idx in indices[0]]
    return distances[0], indices[0] 


class DataPreprocess():
    def __init__(self, use_RAG=False, model_id="llava-hf/llava-1.5-7b-hf", vector_db_path=None, metadata_path=None):

        self.task_names=['general', 'regional', 'suggestion']
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.use_RAG=use_RAG
        self.vector_db_path=vector_db_path
        self.metadata_path=metadata_path
        
        self.RAG_index={}
        self.metadata={}

        # TODO: adding RAG into data processing 
        if self.use_RAG: 
            for task_name in self.task_names:
                assert task_name in vector_db_path and task_name in metadata_path, "typo for input_path key"
                self.RAG_index[task_name]=faiss.read_index(vector_db_path[task_name])
                with open(metadata_path[task_name], 'r') as f:
                    self.metadata[task_name]= json.load(f)

    def preprocess_data(self, batch):
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
            # print("================")
            # print(process_text)
            image.append(data['image'])
            text.append(process_text)
            image_names.append(data['id'])
        inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt')        
        # inputs['ids']=image_name
        # print(inputs['pixel_values'].shape)
        return inputs, image_names


    def preprocess_data_prompt_tuning(self, batch):
        def conver_to_template(task, conversation):
            """    
            from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
            to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
            """    
            prompt=""
            prompt_charactor="You are a traffic analysis assistant. "
            if 'general' in task:
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command="Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                
            elif 'suggestion' in task:
                """
                There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command= "Please provide driving suggestions for the ego car based on the current scene."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                
            elif 'regional' in task:
                """
                Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. "
                prompt_command="Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                # print(prompt)
            
            template=f'USER: <image>\n{prompt} ASSISTANT:'
            return template
        # ==========================================================================================================================================================================================
        image=[]
        text=[]
        image_names=[]
        
        for data in batch:
            process_text=conver_to_template(data['id'], data['conversations'][0]['value'], )
            # print(process_text)
            image.append(data['image'])
            text.append(process_text)
            image_names.append(data['id'])
        inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt')        
        return inputs, image_names

    def preprocess_data_RAG_prompt_tuning(self, batch):
        def conver_to_template(task, retrieve_conversation):
            """    
            from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
            to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
            """    
            prompt=""
            prompt_charactor="You are a traffic analysis assistant. "
            prompt_context="Here is a similar example for the following task: "+ retrieve_conversation
            # TODO: process three type of template 
            if 'general' in task:
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command="Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
                prompt=prompt_charactor+prompt_constrain+prompt_context+prompt_command
                
            elif 'suggestion' in task:
                """
                There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command= "Please provide driving suggestions for the ego car based on the current scene."
                prompt=prompt_charactor+prompt_constrain+prompt_context+prompt_command
                
            elif 'regional' in task:
                """
                Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
                """
                prompt_command="Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
                prompt=prompt_charactor+prompt_context+prompt_command
                
            template=f'USER: <image>\n{prompt} ASSISTANT:'
            return template
        # ==========================================================================================================================================================================================
        image=[]
        text=[]
        image_names=[]
        for data in batch:
            task=data['id'].split('_')[1]

            # TODO: RAG retrieve data
            # step: preprocess test images into concatenated embeddings using Semantic SAM and Depth anything
            concated_embedding= embed_fn(data['image'].copy())  # TODO: embed_fn, embed test image into a concated embedding 
            
            # step: To get the most similiar image in the database
            distance, indice= retrieve_similar_images(concated_embedding, self.RAG_index[task], top_k=3)
            
            # step: To get the meta data of the image 
            top_1=indice[0]
            retrieved_data= self.metadata[task][str(top_1)]['conversations']
            
            # step: Prompt tuning 
            process_text=conver_to_template(data['id'], retrieved_data )
            
            # step: append data into a list for processor to tokenize and transform
            image.append(data['image'])
            text.append(process_text)
            image_names.append(data['id']) # for output .json format
        
        # step: Using processor to tokenize and transform
        inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt')        
        return inputs, image_names

# if __name__=='__main__':
    # from datasets import load_dataset
    # dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='test')
    