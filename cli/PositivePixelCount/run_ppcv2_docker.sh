##  this will run the latest ppc, with some default params, for the file E127-127_10_AB.svs
##  it will output a tiff file and an annotation json file
## 
#python PositivePixelCountV2.py "PositivePixelCount2.0" E127-127_10_AB.svs 0.05 0.15 0.05 0.95 0.65 0.35 0.05 
#--outputLabelImage bigPPCTiff.tiff --image_annotation outputAnnotationJson.anot


docker run -v "$(pwd):/data" gutmanlab/neurotk_docker:latest PositivePixelCountV2  "PPC_2.0_Test" /data/E127-127_10_AB.svs 0.05 0.15 0.05 0.95 0.65 0.35 0.05  --outputLabelImage bigPPCTiff.tiff --image_annotation outputAnnotationJson.t2.anot --num_workers 16
