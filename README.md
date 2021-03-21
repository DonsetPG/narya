# Narya

The Narya API allows you track soccer player from camera inputs, and evaluate them with an Expected Discounted Goal (EDG) Agent. This repository contains the implementation of the [following paper](https://arxiv.org/abs/2101.05388). We also make available all of our pretrained agents, and the datasets we used as well. 

The goal of this repository is to allow anyone without any access to soccer data to produce its own and to analyse them with powerfull tools. We also hope that by releasing our training procedures and datasets, better models will emerge and make this tool better iteratively. 

We also built 4 notebooks to explain how to use our models and a colab:

- [Colab](https://colab.research.google.com/drive/1VC3yd_M4va86N0q9NsYT0ajhCZ-k1sBO?usp=sharing)
- [Each tracking model](https://github.com/DonsetPG/narya/blob/master/models_examples.ipynb)
- [Complete Tracker](https://github.com/DonsetPG/narya/blob/master/full-tracking.ipynb)
- [EDG DRL Agent](https://github.com/DonsetPG/narya/blob/master/data-analysis.ipynb)
- [Datasets and Training](https://github.com/DonsetPG/narya/blob/master/training.ipynb)

and released of blog post version of these notebooks [here](https://donsetpg.github.io/blog/2020/12/24/Narya/).

We tried to make everything easy to reuse, we hope anyone will be able to:

* Use our datasets to train other models
* Finetune some of our trained models
* Use our trackers
* Evaluate players with our EDG Agent
* and much more

You can find at the bottom of the readme links to our models and datasets, but also to tools and models trained by the community. 

# Installation  

You can either install narya from source:

```git clone && cd narya && pip3 install -r requirements.txt```

### Google Football: 

Google Football needs to be installed differently. Please see their repo to take care of it.

[Google Football Repo](https://github.com/google-research/football)

## Player tracking: 

The installed version is directly compatible with the player tracking models. However, it seems that some errors might occur with ```keras.load_model``` when the architecture of the model is contained in the .h5 file. In doubt, Python 3.7 is always working with our installation.

## EDG: 

As Google Football API is currently not supporting Tensorflow 2, you need to manually downgrade its version in order to use our EDG agent: 

```pip3 install tensorflow==1.13.1```
```pip3 install tensorflow_probability==0.5.0```

### Models & Datasets:

The models will be downloaded automatically with the library. If needed, they can be access at the end of the readme. The datasets are also available below.

# Tracking Players Models:

Each model can be accessed on its own, or you can use the full tracking itself.

## Single Model 

Each pretrained model is built on the same architecture to allow for the easier utilisation possible: you import it, and you use it. The processing function, or different frameworks, are handled internaly.

Let's import an image:

```
import numpy as np
import cv2
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

Now, let's create our models:

```python3
from narya.models.keras_models import DeepHomoModel
from narya.models.keras_models import KeypointDetectorModel
from narya.models.gluon_models import TrackerModel

direct_homography_model = DeepHomoModel()

keypoint_model = KeypointDetectorModel(
    backbone='efficientnetb3', num_classes=29, input_shape=(320, 320),
)

tracking_model = TrackerModel(pretrained=True, backbone='ssd_512_resnet50_v1_coco')
```

We can now directly make predictions: 

```
homography_1 = direct_homography_model(image)
keypoints_masks = keypoint_model(image)
cid, score, bbox = tracking_model(image)
```

In the tracking class, we also process the homography we estimate with interpolation and filters. This ensure smooth estimation during the entire video.

### Processing: 

We can now vizualise or use each of this predictions.
For example, visualize the predicted keypoints: 

```
from narya.utils.vizualization import visualize
visualize(
        image=denormalize(image.squeeze()),
        pr_mask=keypoints_masks[..., -1].squeeze(),
    )
```

## Full Tracker: 

Given a list of images, one can easily apply our tracking algorithm: 

```
from narya.tracker.full_tracker import FootballTracker
```

This tracker contains the 3 models seen above, and the tracking/ReIdentification model. 
You can create it by specifying your frame rate, and the size of the memory frames buffer:

```
tracker = FootballTracker(frame_rate=24.7,track_buffer = 60)
```

Given a list of image, the full tracking is computed using:

```
trajectories = tracker(img_list,split_size = 512, save_tracking_folder = 'test_tracking/',
                        template = template,skip_homo = None)
```

We also built post processing functions to handle the mistakes the tracker can make, and also visualization tools to plot the data.

# EDG:

The best way to use our EDG agent is to first convert your tracking data to a google format, using the utils functions: 

```python3
from narya.utils.google_football_utils import _save_data, _build_obs_stacked

data_google = _save_data(df,'test_temo.dump')
observations = {
    'frame_count':[],
    'obs':[],
    'obs_count':[],
    'value':[]
}
for i in range(len(data_google)):
    obs,obs_count = _build_obs_stacked(data_google,i)
    observations['frame_count'].append(i)
    observations['obs'].append(obs)
    observations['obs_count'].append(obs_count)
```

You can now easily load a pretrained agent, and use it to get the value of any action with:

```python3
from narya.analytics.edg_agent import AgentValue

agent = AgentValue(checkpoints = checkpoints)
value = agent.get_value([obs])
```

## Processing:

You can use these values to plot the value of an action, or plot map of values at a given time.
You can use: 

```python3 
map_value = agent.get_edg_map(observations['obs'][20],observations['obs_count'][20],79,57,entity = 'ball')
```

and

```python3
for indx,obs in enumerate(observations['obs']):
    value = agent.get_value([obs])
    observations['value'].append(value)
df_dict = {
    'frame_count':observations['frame_count'],
    'value':observations['value']
}
df_ = pd.DataFrame(df_dict)
```

to compute an EDG map and the EDG overtime of an action.

# Open Source

Our goal with this project was to both build a powerful tool to analyse soccer plays. This led us to build a soccer player tracking model on top of it. We hope that by releasing our codes, weights, and datasets, more people will be able to perform amazing projects related to soccer/sport analysis.

If you find any bug, please open an issue. If you see any improvements, or trained a model you want to share, please open a pull request!

# Thanks

A special thanks to [Last Row](https://twitter.com/lastrowview), for providing some tracking data at the beginning, to try our agent, and to [Soccermatics](https://twitter.com/Soccermatics) for providing Vizualisation tools (and some motivation to start this project).

# Citation

If you use Narya in your research and would like to cite it, we suggest you use the following citation:
```
@misc{garnier2021evaluating,
      title={Evaluating Soccer Player: from Live Camera to Deep Reinforcement Learning}, 
      author={Paul Garnier and Théophane Gregoir},
      year={2021},
      eprint={2101.05388},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
# Links:

## Links to the models and datasets from the original Paper

| Model                  | Description                               | Link                                                                        |
|------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| 11_vs_11_selfplay_last | EDG agent                                 | https://storage.googleapis.com/narya-bucket-1/models/11_vs_11_selfplay_last |
| deep_homo_model.h5     | Direct Homography estimation Weights      | https://storage.googleapis.com/narya-bucket-1/models/deep_homo_model.h5     |
| deep_homo_model_1.h5   | Direct Homography estimation Architecture | https://storage.googleapis.com/narya-bucket-1/models/deep_homo_model_1.h5   |
| keypoint_detector.h5   | Keypoints detection Weights               | https://storage.googleapis.com/narya-bucket-1/models/keypoint_detector.h5   |
| player_reid.pth        | Player Embedding Weights                  | https://storage.googleapis.com/narya-bucket-1/models/player_reid.pth        |
| player_tracker.params  | Player & Ball detection Weights           | https://storage.googleapis.com/narya-bucket-1/models/player_tracker.params  |

The datasets can be downloaded at: 

| Dataset                | Description                                                             | Link                                                                         |
|------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------|
| homography_dataset.zip | Homography Dataset (image,homography)                                   | https://storage.googleapis.com/narya-bucket-1/dataset/homography_dataset.zip |
| keypoints_dataset.zip  | Keypoint Dataset (image,list of mask)                                   | https://storage.googleapis.com/narya-bucket-1/dataset/keypoints_dataset.zip  |
| tracking_dataset.zip   | Tracking Dataset in VOC format (image, bounding boxes for players/ball) | https://storage.googleapis.com/narya-bucket-1/dataset/tracking_dataset.zip   |

## Links to models trained by the community 

### Experimental data for vertical pitches:

| Model                                    | Description                               | Link                                                                                            |
|------------------------------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------|
| vertical_HomographyModel_0.0001_32.h5    | Direct Homography estimation Weights      | https://storage.googleapis.com/narya-bucket-1/models/vertical_HomographyModel_0.0001_32.h5      |
| vertical_FPN_efficientnetb3_0.0001_32.h5 | Keypoints detection Weights               | https://storage.googleapis.com/narya-bucket-1/models/vertical_FPN_efficientnetb3_0.0001_32.h5   |

| Dataset                                | Description                                                             | Link                                                                                        |
|----------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| vertical_samples_direct_homography.zip | Homography Dataset (image,homography)                                   | https://storage.googleapis.com/narya-bucket-1/dataset/vertical_samples_direct_homography.zip|
| vertical_samples_keypoints.zip         | Keypoint Dataset (image,list of mask)                                   | https://storage.googleapis.com/narya-bucket-1/dataset/vertical_samples_keypoints.zip        |

# Tools

## Tool for efficient creation of training labels:

Tool built by [@larsmaurath](https://github.com/larsmaurath) to label football images: https://github.com/larsmaurath/narya-label-creator

## Tool for creation of keypoints datasets:

Tool built by [@kkoripl](https://github.com/kkoripl) to create keypoints datasets - xml files and images resizing: https://github.com/kkoripl/NaryaKeyPointsDatasetCreator
