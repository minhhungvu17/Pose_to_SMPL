import matplotlib.pyplot as plt
import imageio, os

images = []
dataset = 'HAA4D'
action = 'baseball_swing_000'
filenames = sorted(fn for fn in os.listdir(f'fit/output/{dataset}/picture/{action}'))

for filename in filenames:
    images.append(imageio.imread(f'fit/output/{dataset}/picture/{action}/'+filename))
imageio.mimsave(f'{dataset}_{action}.gif', images, duration=0.2)