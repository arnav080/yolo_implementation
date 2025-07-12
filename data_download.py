from torchvision.datasets import VOCDetection

# Automatically downloads VOC 2007 train data to ./data/VOCdevkit/
VOCDetection(
    root="./data",
    year="2007",
    image_set="train",
    download=True
)
