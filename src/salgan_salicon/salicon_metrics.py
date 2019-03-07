import pysaliency
import datetime
from IPython import embed
dataset_location = 'datasets'
model_location = 'models'

start = datetime.datetime.now().replace(microsecond=0)
maps = pysaliency.external_datasets.get_SALICON(edition='2017',location=dataset_location)
saliency_maps = maps[1]
fixations_maps = maps[4]
print("The metrics are:")
embed()
my_model = pysaliency.SaliencyMapModelFromDirectory(saliency_maps, '/home/saliency_maps/salgan_salicon_baseline/')
auc = my_model.AUC(saliency_maps, fixations_maps)
print("AUC-JUDD is {}".format(auc))
# sauc = my_model.AUCs(saliency_maps, fixations_maps)
# print("AUC-SHUFFLED is {}".format(sauc))
nss = my_model.NSS(saliency_maps, fixations_maps)
print("NSS is {}".format(nss))
cc = my_model.CC(saliency_maps,my_model)
print("CC is {}".format(cc))
sim = my_model.SIM(saliency_maps,my_model)
print("SIM is {}".format(sim))

print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
