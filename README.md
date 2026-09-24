# Scene classification with a spatial-pyramid bag of visual words

Six-class scene recognition (buildings, forest, glacier, mountain, sea, street) using hand-crafted
features only, with no neural network anywhere in the pipeline.

MSc Artificial Intelligence coursework, COMP6223 Computer Vision, University of Southampton,
spring 2026. Group coursework for three students, published with my teammates' agreement. I led the
design and integration, and wrote the traditional-feature module (Canny edges and Harris corners),
the feature fusion, the cross-validation and the evaluation.

## The data

The public [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification):
14,034 training images and 3,000 test images across six scene classes. **Not included here** -
download it into `./intel_image_classification`.

## The pipeline

1. **Traditional features** - Canny edge and Harris corner statistics per image.
2. **Dense SIFT** descriptors sampled on a grid.
3. **Codebook** - Mini-Batch K-Means over the descriptors, 500 visual words.
4. **Spatial pyramid encoding** - three levels, so the histogram keeps some spatial layout rather
   than treating the image as an unordered bag.
5. **PCA whitening**, then fusion with the traditional features.
6. **Linear SVM**, one-vs-rest, tuned with 5-fold cross-validation.

## Results

| Metric | Score |
|---|---|
| Top-1 accuracy | 0.7173 |
| mAP | 0.7805 |

The confusion matrix shows where it fails: glacier and mountain are the pair it mixes up most, which
makes sense for a bag-of-words model, since both are dominated by the same textures and edges and the
difference is largely in context rather than local appearance.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook scene_classification.ipynb
```

Dense SIFT over 14,000 images is the slow step.

## Note

This is coursework code, published with my tutor's confirmation that the code I wrote is mine to
share and with my teammates' agreement for the parts we wrote together. The assignment brief, the
report and the dataset are not included.
