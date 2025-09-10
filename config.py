import os

home = os.path.expanduser("")
root_dirs = {
    'bird': home + '../datasets/CUB_200_2011/',
    'car':  home + '../datasets/StanfordCars',
    'air':  home + '../datasets/fgvc-aircraft-2013b',
    'algae':  home + '../datasets/ALGAE',
}

class_nums = {
    'bird': 200,
    'car': 196,
    'air': 100,
    'dog': 120,
    'algae': 32,
}

HyperParams = {
    'alpha': 0.5,
    'beta':  0.5,
    'gamma': 1,
    'kind': 'bird',
    'bs': 10,
    'start_epoch': 0,
    'epoch': 200,
    'arch': 'resnet50',
    'part': 2,
    'restart': False,
    'InfoNCE': 0.3,
    'kd_temp': 5,
    'type': 'full',
}
