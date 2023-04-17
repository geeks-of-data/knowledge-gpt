from knowledgegpt.extractors.base_extractor import BaseExtractor
from knowledgegpt.extractors.helpers import check_embedding_extractor, check_model_lang, check_index_type

class BaseAgent(BaseExtractor):
    """_summary_

    Args:
        BaseExtractor (_type_): _description_
    """
    
    def __init__(self, dataframe=None, embedding_extractor="hf", model_lang="en", is_turbo=False, index_type="basic",
                 verbose=False, index_path=None, is_gpt4=False, prompt_template=None, task_type="image_generation", hf_token=None,  strict_context: bool = False):
        super().__init__(dataframe=dataframe, embedding_extractor=embedding_extractor, model_lang=model_lang,
                         is_turbo=is_turbo, index_type=index_type, verbose=verbose, index_path=index_path, is_gpt4=is_gpt4,
                         prompt_template=prompt_template, strict_context=strict_context)
        
        self.hf_token = hf_token
        self.task_type = task_type
        
    def agent_run(self, query, max_tokens, load_index=False):
        
        if self.task_type == "image_generation":
                  
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

            image = Image.open(io.BytesIO(response.content))
            print("Image Generated for prompt:" + answer)

            image.save(os.path.join(working_dir, filename))

            return "Saved to disk:" + filename