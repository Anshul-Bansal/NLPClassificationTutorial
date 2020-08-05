
# Data cleaning

<p> Following are standard text cleaning step required in NLP projects.</p>

1. Handling apostrophe
2. Manage special/non-ascii characters
3. stemming
4. stop words removal

# Exploratory data Analysis

1. frequency count of the words in text corpus
2. vocabulary of the text corpus


# Feature engineering/Preprocessing

## Pretrained embedding


1. GloVe-300d-840B
2. FastText-Crawl-300d-2M

<p> The embedding are stored in a form of dictionary. Pre-trained embedding can be evaluated based on coverage of vobaulary and text. The coverage of the data can drastically change after data cleaning. </p>

```
glove_embeddings = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)
fasttext_embeddings = np.load('../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl', allow_pickle=True)


def check_embeddings_coverage(X, embeddings):
    
    vocab = build_vocab(X)    
    
    covered = {}
    oov = {}    
    n_covered = 0
    n_oov = 0
    
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
            
    vocab_coverage = len(covered) / len(vocab)
    text_coverage = (n_covered / (n_covered + n_oov))
    
    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_oov, vocab_coverage, text_coverage

#https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#4.-Embeddings-and-Text-Cleaning

```



## Padding of text input

<p>Padding is required to make all the input of same length. Same length input are necessary for many nn models. Following are some question that we need to explore?</p>

1. The padding should be pre or post? 
2. How padding and truncating affects accuracy of the model?
3. How padding affect the speed of the algorithm?
4. Truncating should be pre or post?


# Model training

## batch size
1. How bacth size affects the accuracy and time in model training and inference?




## Regular Expression





## Documentation

1. Find a tool to draw architecture diagrams quickly


# Bert Theory
## Encoder
1. Self attention
2. Feed-Forward networks

## Decoder
1. self attention
2. Encoder-Decoder Attention
3. Feed forward network


## Popular pretrained BERT models
1. BERT base: 12 encoder-decoder layers
2. BERT large: 24 encoder-decoder layers

## 
maxSeqLength = 5
embedding = 3


x1,x2,x3,x4,x5

w1(5x1)


y1,y2,y3,y4,y5




Learn about different types of attention.


## Input to a Bert model
<p> </p>

1. Input Embeddings:
2. Segment Embedding: These embedding respensent sentences. Changes value from once sentence to another. These embedding are relvant in case where we generate next sentences. They are not relevant for classification and clustering tasks.
3. Position Embedding:  BERT learns a unique position embedding for each positions in the input sequence, and this position-specific information can flow through the model to the key and query vectors.

## Tokenization in bert model
<p> BERT model is trained on wikipedia and book corpus. The vocabulary of the BERT for english is around 30k words but because it uses subwords for new words this is sufficient. Bert tokenization uses word piece tokenizer. 
https://github.com/google-research/bert/issues/149

When multiple sentence are present in one input text, we don't know how we should exactly reshape the input
</p>


## loss function for BERT
<p>BERT's weights are learned in advance through two unsupervised tasks: masked language modeling (predicting a missing word given the left and right context) and next sentence prediction (predicting whether one sentence follows another).  </p>


## Question
What is attention head


## Referlinks

### Code links

1. https://www.kaggle.com/c/nlp-getting-started/notebooks
2. https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#4.-Embeddings-and-Text-Cleaning
3. https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert
4. https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
5. https://www.kaggle.com/ratan123/in-depth-guide-to-google-s-bert
6. https://github.com/strongio/keras-elmo
7. https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
8. https://huggingface.co/transformers/pretrained_models.html
9. https://github.com/google-research/bert
10. https://www.tensorflow.org/guide/keras/sequential_model

### Course Links
1. Theorectical: https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
2. Attention model by Andrew NG: https://www.youtube.com/watch?v=SysgYptB198
3. Attention model by Andrew NG: https://www.youtube.com/watch?v=FMXUkEbjf9k
4. https://jalammar.github.io/illustrated-transformer/
5. Bert Paper: https://arxiv.org/pdf/1810.04805.pdf
6. http://jalammar.github.io/illustrated-bert/
7. https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1


## ToDo

1. funational and sequential apis keras
2. Bert architecture

