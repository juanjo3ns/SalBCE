import os
from IPython import embed
from tensorboard_logger import configure, log_value, log_histogram
import numpy as np
from evaluation.video_metrics import video_metrics

configure("/home/code/evaluation/video_metrics/", flush_secs=5)

videos = video_metrics()
for v in videos:
	for id_epoch in range(0,4):
		embed()
		log_value("Metric/{}".format("AUCJ"),videos[v]['AUC-J'][id_epoch] , id_epoch)
