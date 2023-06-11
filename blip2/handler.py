
#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

import os
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import runpod

# Load models into VRAM here so they can be warm between requests
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device,
)
model.to(device)


def predict(image_path, caption=False, question="What is this a picture of?", context=None, use_nucleus_sampling=False, temperature=1.0):
    """Run a single prediction on the model"""
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    if caption or question == "":
        print("captioning")
        response = model.generate({"image": image})
        return response[0]

    q = f"Question: {question} Answer:"
    if context:
        q = " ".join([context, q])
    print(f"input for question answering: {q}")
    if use_nucleus_sampling:
        response = model.generate(
            {"image": image, "prompt": q},
            use_nucleus_sampling=use_nucleus_sampling,
            temperature=temperature,
        )
    else:
        response = model.generate({"image": image, "prompt": q})

    return response[0]


def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    '''
    {
    'delayTime': 2534,
    'id': '2a16b881-830f-4d14-af5b-f7db7c0a96fc',
    'input': {
        'prompt': 'A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.'
        },
    'status': 'IN_PROGRESS'
}
    '''
    print(event)

    # Parse input parameters from event
    image_path = event['input']['image_path']
    print(image_path)   
    caption = event['input']['caption']
    question = event['input']['question']
    context = event['input']['context']
    use_nucleus_sampling = event['input']['use_nucleus_sampling']
    temperature = event['input']['temperature']


    # Call the predict function with the input parameters
    response = predict(
        image_path=image_path,
        caption=caption,
        question=question,
        context=context,
        use_nucleus_sampling=use_nucleus_sampling,
        temperature=temperature,
    )

    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return response

runpod.serverless.start({
    "handler": handler
})