##  this will run the latest ppc, with some default params, for the file E127-127_10_AB.svs
##  it will output a tiff file and an annotation json file
## 
python PositivePixelCountV2.py "PositivePixelCount2.0" E127-127_10_AB.svs 0.05 0.15 0.05 0.95 0.65 0.35 0.05 --outputLabelImage bigPPCTiff.tiff --image_annotation outputAnnotationJson.anot --outputLabelImage SampleImage1234.tiff --output_form pixelmap --region 10000,10000,30000,30000
