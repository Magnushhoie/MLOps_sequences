torch-model-archiver \
    --model-name pep_cnn \
    --version 1.0 \
    --serialized-file deployable_cnn.pt \
    --export-path model_store \
    --handler image_classifier