import Augmentor

PATH_DATASET="/home/dataset/"
PATH_IMAGES=os.path.join(PATH_DATASET,'images')
PATH_SALIENCY=os.path.join(PATH_DATASET,'maps')

p = Augmentor.Pipeline(PATH_IMAGES)
p.ground_truth(PATH_SALIENCY)

p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.process()
