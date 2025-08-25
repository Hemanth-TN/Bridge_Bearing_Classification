This project is about classifying the images of structural bridge bearings. The bridge bearings can be classified as one of "Good", "Fair", "Poor" and "Severe".
The dataset was taken from [University Libraries of Virginia Tech](https://data.lib.vt.edu/articles/dataset/Bearing_Condition_State_Classification_Dataset/16624642)

The original dataset is highly imbalanced. The distribution of the images are as follows:

| Condition State | # Images in training dataset | # Images in test dataset |
| :-------------- | :--------------------------: | -----------------------: |
| Good (1)        |         124                  |            13            |
| Fair (2)        |         215                  |            23            |
| Poor (3)        |         450                  |            50            |
| Severe (4)      |         90                   |            9             |

**Problem Statement:**

To predict the condition state of a bridge bearing given its image

**Solution Approach:**

* Leverage transfer learning to use the [EfficientNet Models](https://docs.pytorch.org/vision/0.21/models/efficientnet.html)


* The training set is split into training and validation datasets. These two datasets were used to train the models, while the test dataset was only used to report the accuracy

* The classifier head of the EfficientNet models was modified to suit the problem of 4 classes.

* The data augmentation transformations were used to create more number of images with comparable number of samples for each class.

* The last layers of features of EfficientNet models were unfrozen to increase the accuracy

* The accuracy, F1 Score and other metrics were monitored through tensor board.

* The accuracy of the ensemle model was 70% on the test dataset.

* The ensemble models is hosted on HugginFace and can be accessed [here](https://huggingface.co/spaces/Hemanth-TN/Bearing-Classification?logs=container)
