conda create --name tweetDisaster

source activate tweetDisaster

!pip install environment-kernels
conda install pandas
conda install tensorflow
conda install scikit-learn
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
conda install nltk