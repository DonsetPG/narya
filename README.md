# Narya

The Narya API allows you to use a Google Football DRL agent to compute the value of football game. 
At the moment, we provide the following : 

### EDG (Expected Discounted Goal):

Tools: 
  * Functions to compute the value of an action at a certain time, overtime, and EDG maps 
  * Vizualisation tools
  
Data:
  * Checkpoints from pre-trained agents 
  * LastRow tracking data from Liverpool games

### Avallone (Player and Event tracking):

WIP - 

## Setup 

WIP

## How to use it 

We provide two notebooks in the /edg folder, to learn how to use an agent with tracking data. We will extend this base of notebooks overtime. 
The main way to use this is to: 
* Convert your tracking data to a google format, using the utils functions 

```python3
data_google = utils._save_data(df,'test_temo.dump')
observations = {
    'frame_count':[],
    'obs':[],
    'obs_count':[],
    'value':[]
}
for i in range(len(data_google)):
    obs,obs_count = utils._build_obs_stacked(data_google,i)
    observations['frame_count'].append(i)
    observations['obs'].append(obs)
    observations['obs_count'].append(obs_count)
```
* Use an agent 

```python3
agent = AgentValue(checkpoints = checkpoints)
value = agent.get_value([obs])
```

to compute the value of an action. 

### The google format :

The agent takes the tracking data in a very particular format : 

```
The observation is composed of 4 planes of size 'channel_dimensions'.
Its size is then 'channel_dimensions'x4 (or 'channel_dimensions'x16 when
stacked is True).
The first plane P holds the position of players on the left
team, P[y,x] is 255 if there is a player at position (x,y), otherwise,
its value is 0.
The second plane holds in the same way the position of players
on the right team.
The third plane holds the position of the ball.
The last plane holds the active player.
```

To convert them, you need to place your coordinates in a (-1,-1) - (1,1) format, and then place them in the frames mentioned above. At the moment, we provide such transcription only from the LastRow tracking data format.

## In the future 

* We plan to add transcription method from other tracking data format. 
* We will also extend this project with open source models to compute the tracking data directly from video stream.
* While the agents are trained purely against bots or with selfplay, we will release methods to fine tune the agent with real world data (tracking data as inputs, events data as actions). Thanks to this, the agents will be able to model much better real players.
* We will add checkpoints for multi-agent models. 

# Thanks

A special thanks to [Last Row](https://twitter.com/lastrowview), for providing some tracking data, and to [Soccermatics](https://twitter.com/Soccermatics) for providing Vizualisation tools (and some motivation to start this project).


