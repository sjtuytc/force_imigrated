## Sec1. Original json data format.

1. time_to_clip_ind_image_adr.
   Time -> clip index and image address.

```shell
time_to_clip_ind_image_adr['006_mustard_bottle']['2019-09-13 15:05:40.211267'] =
{'image_adr': 'LMJTFY/images/006_mustard_bottle/image_11541_2019_09_13_15_05_40_211267.jpeg', 'clip_ind': 0}
```

2. clip_to_contact_point.

   clip index -> contact points (5*3).

```shell
clip_to_contact_point['035_power_drill']['8']['contact_point']=[[0.14613500237464905, 0.3589400053024292, -0.002804999938234687], [0.14613500237464905, 0.3589400053024292, -0.002804999938234687], ..., ]
```

3. time_to_obj_state_fps.

   time -> position/rotation.

```shell
time_to_obj_state_fps['035_power_drill']['2019-09-13 15:01:37.457229'] = {'position': [-1.1046428680419922, -0.21831931173801422, 3.5171849727630615], 'rotation': [0.16976842284202576, 0.30221450328826904, 0.059603139758110046, -0.9361050128936768]}
```

4. time_to_keypoint_fps.

   time -> key point [x, y]

```shell
time_to_keypoint_fps['035_power_drill']['2019-09-13 15:01:37.457229'] = {'0': {'x': 567.329958417948, 'y': 570.1807852238662, 'label': 0.0}, ..., }
```

## Sec2. Data statistics.

|     object      | TRAIN NUM | reported |
| :-------------: | :-------: | :------: |
|  pitcher base   |    703    |   843    |
| bleach cleanser |    738    |   994    |
|     skillet     |    890    |   904    |
|   power drill   |    908    |   1202   |
|     hammer      |    819    |   970    |
|  toy airplane   |   1722    |   1762   |
| tomato soup can |    503    |   634    |
| mustard bottle  |    736    |   1007   |

