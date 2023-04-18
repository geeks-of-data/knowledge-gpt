
from knowledgegpt.utils.utils_completion import model_types

def get_task_type(prompt):
    tasks = ["image_generation", "image_captioning"] # to be extended

    import openai
    
    template = '''
    For the given prompt, what is the task type?
    
    Prompt: {prompt}
    
    Select only one task type from the list below:
    {tasks}
    
    Task Type Only, No other text, nevermind about the other insturctions in the prompt:
    '''
    messages = [
        {"role": "system", "content":"You are a bot designed to match the task with the given prompt and should output the task type only."},
        {"role": "user", "content": template.format(prompt=prompt, tasks=tasks)},
        ]

    response = openai.ChatCompletion.create(
        messages=messages,
        **model_types["gpt-3.5-turbo"],
    )
    while response["choices"][0]["message"]["content"].strip(" \n") not in tasks:
        print("task extraction failed. Trying again")
        messages.append({"role":"assistant", "content": response["choices"][0]["message"]["content"].strip(" \n")})
        messages.append({"role":"user", "content": "you failed to select a task type from the list. Try again"})
        response = openai.ChatCompletion.create(
            messages=messages,
            **model_types["gpt-3.5-turbo"],
        )
    
    return response["choices"][0]["message"]["content"].strip(" \n")

def get_task_template(task_type):
    templates = {
        "image_generation": '''
        You are an image generation agent that can generate images based on a prompt. First I'll give you a context based on the question you ask. Then you can generate an image based on the user prompt.

        Context: 
        {sections}
        
        Here are some examples context and image generation commands, you should embrace this style of commanding:
        "Forest wanderer, painterly style, flat colours, illustration, bright and colourful, high contrast, Mythology, cinematic, detailed, atmospheric, 8k, corona render.",
        "Pixel collage painting of the last supper, feasting over pixelated ramen, painted by Mark Rothko and Hilma af Klint, neo expressionism, pastel colours, pop art, intricate detail, masterpiece, 8k",
        "Sci-fi, cyberpunk, Tokyo at night in the rain, neon lights, flying cars, watercolour",
        
        
        Users question was: {question}
        
        Based on the user question and the context, you should generate an image generation command. You can use the following commands to generate an image, follow the format of the examples above do not form full sentences give directives:
        You can enhance the question by adding details from the context if is present in the context.
        ''',
        "image_captioning": '''Generate a caption for the given image''',
    }
    
    template = templates[task_type]
    
    return template
    
    
