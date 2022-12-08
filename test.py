from stylegan3 import dnnlib, legacy

network = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl'
f = dnnlib.util.open_url(network)
G = legacy.load_network_pkl(f)