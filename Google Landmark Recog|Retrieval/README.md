## Final rank: 35/736 Private: 0.5140; Public: 0.5414

Competition description: https://www.kaggle.com/c/landmark-recognition-2020

Special thanks to Ragnar's awesome kernel of training Effnet, we find some practical ways to train our own models using this large dataset. 
After reading Keetar's fantastic writeup of his GLD retrieval, we trained our Effnet B6 and B7 using refined ArcFace Loss first with 384 sized images. CV is 0.84 and 0.85 respectively. 
Then we use the increasing 512 sized images to further tuned our B6 and B7. The training environment is Colab Pro. Then we put the two model ensembling predictions to the _global_ feature extraction.

#### Model weight (effb7 trained with 512 resized images): https://www.kaggle.com/tangshuyun/effb7-512-ep12

Reference: https://arxiv.org/pdf/1801.07698.pdf
