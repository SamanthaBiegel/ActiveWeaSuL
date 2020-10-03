#!/bin/bash
set -euxo pipefail

# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "snorkel" ]; then
    echo "Script must be run from snorkel directory" >&2
    exit 1
fi

ANNOTATIONS_URL="https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip"
IMAGES_URL="http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip"
EMBEDDINGS_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

if [ ! -d "data" ]; then
    mkdir -p data
fi

if [ ! "${PWD##*/}" = "data" ]; then
    cd data
fi

if [ ! -d "data/annotations" ]; then
    # Download and unzip metadata and annotations
    echo "Downloading annotations..."
    wget $ANNOTATIONS_URL
    unzip vrd.zip
    rm vrd.zip
    mv VRD annotations
fi

if [ ! -d "data/images" ]; then
    # Download and unzip all images
    echo "Downloading images..."
    wget $IMAGES_URL
    unzip sg_dataset.zip
    rm sg_dataset.zip
    mv sg_dataset images
    cd images
    mv sg_train_images train_images
    mv sg_test_images test_images
fi

if [ ! -d "data/word_embeddings" ]; then
# Download and unzip glove word embeddings
    mkdir -p word_embeddings
    cd word_embeddings

    wget $EMBEDDINGS_URL
    unzip glove.6B.zip
    rm  glove.6B.zip
    cd ../..
fi

