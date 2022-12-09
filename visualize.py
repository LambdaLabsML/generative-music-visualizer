import librosa
import numpy as np
import moviepy.editor as mpy
import random
import torch
from tqdm import tqdm
import dnnlib
import legacy

target_sr = 22050

def visualize(audio_file,
              network,
              truncation,
              tempo_sensitivity,
              jitter,
              frame_length,
              duration,
              ):
    print(audio_file)

    if audio_file:
        print('\nReading audio \n')
        audio, sr = librosa.load(audio_file, duration=duration)
    else:
        raise ValueError("you must enter an audio file name in the --song argument")

    # print(sr)
    # print(audio.dtype)
    # print(audio.shape)
    # if audio.shape[0] < duration * sr:
    #     duration = None
    # else:
    #     frames = duration * sr
    #     audio = audio[:frames]
    #
    # print(audio.dtype)
    # print(audio.shape)
    # if audio.dtype == np.int16:
    #     print(f'min: {np.min(audio)}, max: {np.max(audio)}')
    #     audio = audio.astype(np.float32, order='C') / 2**15
    # elif audio.dtype == np.int32:
    #     print(f'min: {np.min(audio)}, max: {np.max(audio)}')
    #     audio = audio.astype(np.float32, order='C') / 2**31
    # audio = audio.T
    # audio = librosa.to_mono(audio)
    # audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
    # print(audio.dtype)
    # print(audio.shape)



    # TODO:
    batch_size = 1
    resolution = 512
    outfile="output.mp4"

    tempo_sensitivity = tempo_sensitivity * frame_length / 512

    # Load pre-trained model
    device = torch.device('cuda')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G.eval()

    with torch.no_grad():
        z = torch.randn([1, G.z_dim]).cuda()    # latent codes
        c = None                                # class labels (not used in this example)
        img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #create spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=target_sr, n_mels=512,fmax=8000, hop_length=frame_length)

    #get mean power at each time point
    specm=np.mean(spec,axis=0)

    #compute power gradient across time points
    gradm=np.gradient(specm)

    #set max to 1
    gradm=gradm/np.max(gradm)

    #set negative gradient time points to zero
    gradm = gradm.clip(min=0)

    #normalize mean power between 0-1
    specm=(specm-np.min(specm))/np.ptp(specm)

    #initialize first noise vector
    nv1 = torch.randn([G.z_dim]).cuda()

    #initialize list of class and noise vectors
    noise_vectors=[nv1]

    #initialize previous vectors (will be used to track the previous frame)
    nvlast=nv1

    #initialize the direction of noise vector unit updates
    update_dir=np.zeros(512)
    print(len(nv1))
    for ni,n in enumerate(nv1):
        if n<0:
            update_dir[ni] = 1
        else:
            update_dir[ni] = -1

    #initialize noise unit update
    update_last=np.zeros(512)

    #get new jitters
    def new_jitters(jitter):
        jitters=np.zeros(512)
        for j in range(512):
            if random.uniform(0,1)<0.5:
                jitters[j]=1
            else:
                jitters[j]=1-jitter
        return jitters


    #get new update directions
    def new_update_dir(nv2,update_dir):
        for ni,n in enumerate(nv2):
            if n >= 2*truncation - tempo_sensitivity:
                update_dir[ni] = -1

            elif n < -2*truncation + tempo_sensitivity:
                update_dir[ni] = 1
        return update_dir

    print('\nGenerating input vectors \n')
    for i in tqdm(range(len(gradm))):

        #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
        if i%200==0:
            jitters=new_jitters(jitter)

        #get last noise vector
        nv1=nvlast

        #set noise vector update based on direction, sensitivity, jitter, and combination of overall power and gradient of power
        update = np.array([tempo_sensitivity for k in range(512)]) * (gradm[i]+specm[i]) * update_dir * jitters

        #smooth the update with the previous update (to avoid overly sharp frame transitions)
        update=(update+update_last*3)/4

        #set last update
        update_last=update

        #update noise vector
        nv2=nv1.cpu()+update

        #append to noise vectors
        noise_vectors.append(nv2)

        #set last noise vector
        nvlast=nv2

        #update the direction of noise units
        update_dir=new_update_dir(nv2,update_dir)

    noise_vectors = torch.stack([nv.cuda() for nv in noise_vectors])


    print('\n\nGenerating frames \n')
    frames = []
    for i in tqdm(range(noise_vectors.shape[0] // batch_size)):

        noise_vector=noise_vectors[i*batch_size:(i+1)*batch_size]

        c = None  # class labels (not used in this example)
        with torch.no_grad():
            img = np.array(G(noise_vector, c, truncation_psi=truncation, noise_mode='const').cpu())            # NCHW, float32, dynamic range [-1, +1], no truncation
            img = np.transpose(img, (0,2,3,1)) #CHW -> HWC
            img = np.clip((img * 127.5 + 128), 0, 255).astype(np.uint8)

        # add to frames
        for im in img:
            frames.append(im)


    #Save video
    aud = mpy.AudioFileClip(audio_file)

    if duration < aud.duration:
        aud.duration = duration

    fps = target_sr / frame_length
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip = clip.set_audio(aud)
    clip.write_videofile(outfile, audio_codec='aac', ffmpeg_params=[
        # "-vf", "scale=-1:2160:flags=lanczos",
        "-bf", "2",
        "-g", f"{fps/2}",
        "-crf", "18",
        "-movflags", "faststart"
    ])

    return outfile