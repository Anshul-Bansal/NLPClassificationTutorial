
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

<p> The embedding are stored in a form of dictionary. Pre-trained embedding can be evaluated based on coverage of vobaulary and text. The covertage of the data can drastically change after data cleaning. </p>

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