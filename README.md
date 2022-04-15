# EDI
Emotion-Drive Interpretable Fake News Detection
## An Overall Framework
An overall framework for interpretable fake news detection. 
![model-4 12](https://user-images.githubusercontent.com/61655426/163506287-916b5fec-5e90-4114-a95a-2f051a1580cc.png)

The model feature representation consists of 5 components: 

**a)**  Emotion selection based on emotional value from news and user comments; 

**b)**  Emotion representation is obtained by CNN, after emotion embedding; 

**c)**  Emotion attention is a measure of the importance of each user comment.

**d)**  Emotion-emotion co-attention Represents the emotion Correlation of News and User comments

## Datasets

The dataset contains Chinese Weibo and English Twitter datasets，The Weibo datasets are available at https://drive.google.com/drive/folders/1pjK0BYiiJt0Ya2nRIrOLCVo-o53sYRBV?usp=sharing. The Twitter datasets are available at here.

### Twitter

> Ma J, Gao W, Wong KF. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. In: Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics;. p. 708-17. Available from: http://aclweb.org/anthology/P17-1066.

### Weibo-16

The original dataset is firstly proposed in:

> Jing Ma, Wei Gao, Prasenjit Mitra, Sejeong Kwon, Bernard J Jansen, Kam-Fai Wong, and Meeyoung Cha. 2016. Detecting rumors from microblogs with recurrent neural networks. In IJCAI 2016. 3818–3824.

The existing dataset was finally proposed  in:
> Zhang X, Cao J, Li X, Sheng Q, Zhong L, Shu K. Mining Dual Emotion for Fake News Detection. In: Proceedings of the Web Conference 2021. WWW ’21. New York, NY, USA: Association for Computing Machinery; 2021. p. 3465–3476. Available from: https://doi.org/10.1145/3442381.3450004.

### Weibo-20

Weibo-20 dataset is newly proposed in:

> Zhang X, Cao J, Li X, Sheng Q, Zhong L, Shu K. Mining Dual Emotion for Fake News Detection. In: Proceedings of the Web Conference 2021. WWW ’21. New York, NY, USA: Association for Computing Machinery; 2021. p. 3465–3476. Available from: https://doi.org/10.1145/3442381.3450004.
## Code

### Requirements

```
Python == 3.7.9
Keras == 2.6.0
Pytorch-GPU == 1.6.0
numpy == 1.19.2
sklearn == 0.24
json == 2.0.9
heapq
```
