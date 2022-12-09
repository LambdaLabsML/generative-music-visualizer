import gradio as gr

from visualize import visualize

network_choices = [
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl'
]

description = \
    """
    Generate visualizations on an input audio file using [StyleGAN3](https://nvlabs.github.io/stylegan3/) (Karras, Tero, et al. "Alias-free generative adversarial networks." Advances in Neural Information Processing Systems 34 (2021): 852-863.).
    Inspired by [Deep Music Visualizer](https://github.com/msieg/deep-music-visualizer), which used BigGAN (Brock et al., 2018) 
    Developed by Jeremy Hummel at [Lambda](https://lambdalabs.com/)
    """

article = \
    """
    ## How does this work?
    The audio is transformed to a spectral representation by using Short-time Fourier transform (STFT). [librosa]() 
    Starting with an initial noise vector, we perform a random walk, adjusting the length of each step with the power gradient.
    This pushes the noise vector to move around more when the sound changes.
    
    ## Parameter info:
    *Network*: various pre-trained models from NVIDIA, "afhqv2" is animals, "ffhq" is faces, "metfaces" is artwork.
    
    *Truncation*: controls how far the noise vector can be from the origin. `0.7` will generate more realistic, but less diverse samples,
    while `1.2` will can yield more interesting but less realistic images.
    
    *Tempo Sensitivity*: controls the how the size of each step scales with the audio features
    
    *Jitter*: prevents the same exact noise vectors from cycling repetitively, if set to `0`, the images will repeat during
    repetitive parts of the audio
    
    *Frame Length*: controls the number of audio frames per video frame in the output. 
    If you want a higher frame rate for visualizing very rapid music, lower the frame length. 
    If you want a lower frame rate (which will complete the job faster), raise the frame length
    
    *Max Duration*: controls the max length of the visualization, in seconds. Use a shorter value here to get output
    more quickly, especially for testing different combinations of parameters.
    
    Media sources:
    [Maple Leaf Rag - Scott Joplin (1916, public domain)](https://commons.wikimedia.org/wiki/File:Maple_leaf_rag_-_played_by_Scott_Joplin_1916_V2.ogg)
    [Moonlight Sonata Opus 27. no 2. - movement 3 - Ludwig van Beethoven, played by Muriel Nguyen Xuan (2008, CC BY-SA 3.0)](https://commons.wikimedia.org/wiki/File:Muriel-Nguyen-Xuan-Beethovens-Moonlight-Sonata-mvt-3.oga)
    """

examples = [
    ["examples/Maple_leaf_rag_-_played_by_Scott_Joplin_1916_V2.ogg", network_choices[0], 1.0, 0.25, 0.5, 512, 600],
    ["examples/Muriel-Nguyen-Xuan-Beethovens-Moonlight-Sonata-mvt-3.ogx", network_choices[4], 1.2, 0.3, 0.5, 384, 600],
]

demo = gr.Interface(
    fn=visualize,
    title="Generative Music Visualizer",
    description=description,
    article=article,
    inputs=[
            gr.Audio(label="Audio File", type="filepath"),
            gr.Dropdown(choices=network_choices, value=network_choices[0], label="Network"),
            gr.Slider(minimum=0.0, value=1.0, maximum=2.0, label="Truncation"),
            gr.Slider(minimum=0.0, value=0.25, maximum=2.0, label="Tempo Sensitivity"),
            gr.Slider(minimum=0.0, value=0.5, maximum=2.0, label="Jitter"),
            gr.Slider(minimum=64, value=512, maximum=1024, step=64, label="Frame Length (samples)"),
            gr.Slider(minimum=1, value=300, maximum=600, step=1, label="Max Duration (seconds)"),
            ],
    # examples=examples,
    outputs=gr.Video(),
    # cache_examples=True,
)
demo.launch()
