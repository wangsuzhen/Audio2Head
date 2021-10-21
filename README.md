# Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion (IJCAI 2021)

#### [Paper](https://www.ijcai.org/proceedings/2021/0152.pdf) | [Demo](https://www.youtube.com/watch?v=xvcBJ29l8rA)

#### Requirements

- Python 3.6 , Pytorch >= 1.6 and ffmpeg

- Other requirements are listed in the 'requirements.txt'

  

#### Pretrained Checkpoint

Please download the pretrained checkpoint from [google-drive](https://drive.google.com/file/d/1tvI43ZIrnx9Ti2TpFiEO4dK5DOwcECD7/view?usp=sharing) and put it within the folder (`/checkpoints`).



#### Generate Demo Results

```
python inference.py --audio_path xxx.wav --img_path xxx.jpg
```

Note that the input images must keep the same height and width and the face should be appropriately cropped as in `/demo/img`.



#### License and Citation

```
@InProceedings{wang2021audio2head,
author = Suzhen Wang, Lincheng Li, Yu Ding, Changjie Fan, Xin Yu
title = {Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion},
booktitle = {the 30th International Joint Conference on Artificial Intelligence (IJCAI-21)},
year = {2021},
}
```



#### Acknowledgement

This codebase is based on [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model), thanks for their contribution.





