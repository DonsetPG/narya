# Narya

The Narya API allows you track soccer player from camera inputs, and evaluate them with an Expected Discounted Goal (EDG) Agent. This repository contains the implementation of the [following paper](https://). We also make available all of our pretrained agents, and the datasets we used as well. 

The goal of this repository is to allow anyone without any access to soccer data to produce its own and to analyse them with powerfull tools. We also hope that by releasing our training procedures and datasets, better models will emerge and make this tool better iteratively. 

We tried to make everything easy to reuse, we hope anyone will be able to:

* Use our datasets to train other models
* Finetune some of our trained models
* Use our trackers
* Evaluate players with our EDG Agent
* and much more

# Installation  

You can either install narya from source:

```git clone https://github.com/DonsetPG/narya.git && cd narya && pip3 install -r requirements.txt```

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

The models will be downloaded automatically with the library. If needed, they can be access at the end of the readme. The datasets are also available below. [Not here]

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
