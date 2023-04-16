from knowledgegpt.extractors.base_extractor import BaseExtractor
from knowledgegpt.extractors.helpers import check_embedding_extractor, check_model_lang, check_index_type
from knowledgegpt.utils.utils_task_selection import get_task_type, get_task_template
from knowledgegpt.utils.utils_wiki_fetcher import wiki_fetcher

class BaseAgent(BaseExtractor):
    """_summary_

    Args:
        BaseExtractor (_type_): _description_
    """
    
    def __init__(self, dataframe=None, embedding_extractor="hf", model_lang="en", is_turbo=False, index_type="basic",
                 verbose=False, index_path=None, is_gpt4=False,  hf_token=None, info_source=None, info_topic=None):
        super().__init__(dataframe=dataframe, embedding_extractor=embedding_extractor, model_lang=model_lang,
                         is_turbo=is_turbo, index_type=index_type, verbose=verbose, index_path=index_path, is_gpt4=is_gpt4,
                        )
        
        self.hf_token = hf_token
        self.task_type = "image_generation"
        self.agent_first_load = True
        self.info_source = info_source
        self.info_topic = info_topic
        
    def agent_run(self, query, max_tokens, load_index=False):
        
        if load_index:
            self.agent_first_load = True
        
        if self.agent_first_load:
            self.agent_first_load = False
            task_type = get_task_type(query)
            self.task_type = task_type
            print("Task Type: ", task_type)
            self.prompt_template = get_task_template(task_type)
            if self.info_source == "wikipedia":
                self.df  = wiki_fetcher(self.info_topic)
                
        
        if self.task_type == "image_generation":
            print("Image Generation Task")
                  
            import requests
            import os
            from PIL import Image
            import io
            import uuid


            filename = str(uuid.uuid4()) + ".jpg"
            working_dir = "output_images"

            # Create target Directory if don't exist
            if not os.path.exists(working_dir):
                os.mkdir(working_dir)
                print("Directory ", working_dir, " Created ")

            answer, prompt, messages = self.extract(query, max_tokens, load_index=load_index)


            API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
            headers = {"Authorization": "Bearer " + self.hf_token}

            response = requests.post(API_URL, headers=headers, json={
                "inputs": answer,
            })
            
            if response.status_code != 200:
                raise Exception("Request failed: {} - {}".format(response.status_code, response.text))
            
            image = Image.open(io.BytesIO(response.content))
            print("Image Generated for prompt:" + answer)

            image.save(os.path.join(working_dir, filename))

            return "Saved to disk:" + filename
        
    
        if self.task_type == "image_captioning":
            print("Image Captioning")