import os
import json
import numpy as np

class TweetUtils:

    def __init__(self, TWITTER16_ROOT='./', TWITTER16_LABEL_RN_ROOT='./'):
        self.TWITTER16_ROOT = TWITTER16_ROOT
        self.TWITTER16_LABEL_RN_ROOT = TWITTER16_LABEL_RN_ROOT
    
    def loadAllPosts(self, labels_file = 'label.txt'):
        """
        Carga todos los posts en formato json desde el directorio ./post.
        
        Retorna:
        
        - all_posts, diccionario indexado por tweet id
        - labeled_posts, diccionaro con tweets etiquetados
        - number_of_tweets
        """
        label_path = os.path.join(self.TWITTER16_LABEL_RN_ROOT, labels_file)
        print(f"Using label file: {label_path}")

        post_path_root = os.path.join(self.TWITTER16_ROOT, 'post')
        
        def parseTwitterTree(tree_file):
            tree_data = list()
            for line in tree_file:
                _, second_part = line.split('->')
                second_part = second_part.rstrip()
                second_part = second_part.replace("'", "\"")
                tree_data.append(json.loads(second_part))     
            return tree_data
        
        ### Obtener diccionario con todos los posts
        all_posts = {}
        for file in os.listdir(post_path_root):
            if file.endswith(".json"):        
                try:
                    with open(os.path.join(post_path_root, file), 'r') as f:
                        tweet_id  = os.path.splitext(file)[0]
                        tweet_dic = json.load(f)
                        all_posts[tweet_id] = tweet_dic
                except:
                    pass

        ### Obtener ids de tweets etiquetados
        labels = {}
        with open(label_path) as label_f:
            for label_line in label_f:
                label, tweet_id = label_line.split(':')
                tweet_id = tweet_id.rstrip()
                labels[tweet_id] = label

        print("Tweets etiquetados      : ", len(labels))        

        seqs_lens = []
        labeled_posts = {}
        number_of_tweets = 0
        number_of_retweets = 0
        number_of_invalid_tweets = 0
        no_in_data = 0
        for tweet_id in labels.keys():
            try:
                if tweet_id in all_posts:
                    tree_path = os.path.join(self.TWITTER16_ROOT, 'tree', tweet_id + '.txt')
                    with open(tree_path) as tree_file:
                        tree_data = parseTwitterTree(tree_file)

                        ### Remover retweets                
                        first = tree_data[0]
                        without_rt = list(filter(lambda t: t[1] != tweet_id, tree_data))
                        number_of_retweets = number_of_retweets + (len(tree_data) - len(without_rt))
                        only_valid = list(filter(lambda t: t[1] in all_posts, without_rt))
                        number_of_invalid_tweets = number_of_invalid_tweets + (len(without_rt) - len(only_valid))
                        seqs_lens.append(len(only_valid))
                        labeled_posts[tweet_id] = (labels[tweet_id], [first] + only_valid)
                        number_of_tweets = number_of_tweets + 1                
                else:
                     no_in_data = no_in_data + 1  

            except Exception as e:
                print(e)

        print("no_in_data              : ", no_in_data) ## están etiquetados, pero no en los post
        print("number_of_tweets        : ", number_of_tweets)        
        print("all_posts               : ", len(all_posts))
        print("number_of_retweets      : ", number_of_retweets) ## En árbol de propagación
        print("number_of_invalid_tweets: ", number_of_invalid_tweets) ## En árbol de propagación
        
        #La red neuronal necesita un tamaño fijo para la secuencia (datos de entrada)
        #¿Que largo de secuencia utilizar?
        counts = np.bincount(seqs_lens) ## seqs_len sólo de los 753
        mode_seq_len = np.argmax(counts)
        mean_seq_len = int(np.mean(seqs_lens))
        min__seq_len = min(seqs_lens)
        max__seq_len = max(seqs_lens) 

        print("len(seqs_lens)   : ", len(seqs_lens))
        print("min__seq_len: ", min__seq_len)
        print("max__seq_len: ", max__seq_len)
        print("mean_seq_len: ", mean_seq_len)
        print("mode_seq_len: ", mode_seq_len)

        tree_max_num_seq = mean_seq_len
        
        return (all_posts, labeled_posts, number_of_tweets, tree_max_num_seq)
    
    
    def generate_XY(
        self,
        _all_posts,
        _model,
        _model_vocab,
        _emb_size,
        _number_of_tweets,
        _labeled_posts,
        _tree_max_num_seq,
        _categories,
        _BERT = False):
        """
        Genera matrices X, Y listos para ser aplicados a un modelo de red neuronal.
        Además retorna la lista de palabras encontradas en los posts pero que no pertenecen al vocabulario del modelo.

        Parametros:
        - _model: modelo indexable que representa los embeddings asociados a cada palabra
        - _model_vocab: diccionario con las palabras que efectivamente están en el modelo _model

        Retorna: (X, Y, words_not_in_model)
        """

        #entrega un vector one-hot de la categoria, de largo 4 (por el número de ategorias)
        def to_category_vector(_category, _categories):
            vector = np.zeros(len(_categories)).astype(np.float32)
            for i in range(len(_categories)):
                if _categories[i] == _category:
                    vector[i] = 1.0
                    break
            return vector

        ## padding al final, con empty
        def padAWE(empty, max_num, seq):
            from itertools import repeat
            seq.extend(repeat(empty, max_num - len(seq)))
            return seq

        def normalizarTexto(docText):
            # En gensim.utils, pasa a minúsculas, descarta palabras muy grandes o muy pequeñas.
            from gensim.utils import simple_preprocess
            return simple_preprocess(docText)

        words_not_in_model = list()    
        def computeDocumentAWE(docText, _model, _model_vocab, _emb_size):
            """
            Only for Gensim 4.x
            Calcula el AWE del texto recibido en el parámetro docText.
            Se considero una palabra para el cálculo sólo si esta pertenece
            al vocabulario del modelo. Si no, no es considerada en la suma 
            ni tampoco en el calculo de n.
            """    
            docSum = np.zeros(_emb_size)
            n = 0

            ####
            ## AWE = 1/n * Sum w_embedding, para cada w en docText
            ####
            normalizedDocText = normalizarTexto(docText)
            for w in normalizedDocText:
                ## Se descartan palabras que no están en el modelo de embeddings (vocabulario)
                if w in _model_vocab:
                    n = n + 1
                    w_embedding = _model.wv[w]
                    docSum = docSum + w_embedding
                else:
                    words_not_in_model.append(w)

            return docSum / n if n > 0 else docSum 
        
        def computeAweBERT(docTexts):
            return None
            # bc = BertClient(ip='reaver')
            # encoded = bc.encode(docTexts, show_tokens=True)
            # return encoded[0].tolist()
            

        ## Cada palabra tiene un embedding que viene del modelo
        ## Se calcula AWE para cada post del árbol de propagación (lista de propagación que el primer elemento es el tweet original)
        def computeTreeAWE(tree, _model, _model_vocab, _emb_size):
            return list(map(lambda t: [t[0], computeDocumentAWE(_all_posts[t[1]]['text'], _model, _model_vocab, _emb_size), t[2]], tree))

        empty_awe = np.zeros(_emb_size)
        _num_categories = len(_categories)

        ## Calcula AWE de cada árbol
        if _BERT is False:
            labeled_posts_awe = { k: (v[0], computeTreeAWE(v[1], _model, _model_vocab, _emb_size)) for k, v in _labeled_posts.items() }
        else:
            labeled_posts_awe = { k: (v[0], computeAweBERT(list(map(lambda x: _all_posts[x[1]]['text'], v[1])))) for k, v in _labeled_posts.items() }

        ## Realiza padding a las secuencias
        padded_labeled_posts_awe = {k: (v[0], padAWE(empty_awe, _tree_max_num_seq, v[1])) for k, v in labeled_posts_awe.items()}

        #Genera los datos X e Y para alimentar el modelo de red neuronal
        #Inicialmente con ceros y con la forma adecuada.
        X = np.zeros(shape=(_number_of_tweets, _tree_max_num_seq, _emb_size)).astype(np.float32)
        Y = np.zeros(shape=(_number_of_tweets, _num_categories)).astype(np.float32)

        # Asigna al vector X los datos correspondientes
        for idx, (tweet_id, tweet_data) in enumerate(list(padded_labeled_posts_awe.items())):
            for jdx, tweet_d in enumerate(tweet_data[1]):
                ### tweet_d = [uid, tweet_awe, time]
                if jdx == _tree_max_num_seq:
                    break            
                else:            
                    X[idx, jdx, :] = tweet_d[1]

        # Asigna al vector Y los datos correspondientes            
        for idx, (tweet_id, tweet_data) in enumerate(list(padded_labeled_posts_awe.items())):
            Y[idx, :] = to_category_vector(tweet_data[0], _categories)

        print("X.shape: ", np.shape(X))
        print("Y.shape: ", np.shape(Y))
        print("#Words not in model: ", len(words_not_in_model))
        return X, Y, words_not_in_model
    
    def bestInHistory(self, _history, field = 'mean_acc'):
        max_idx = max(range(len(_history)),key = lambda index: _history[index][field])
        return _history[max_idx]    