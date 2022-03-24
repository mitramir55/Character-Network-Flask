from graphviz import Graph
import spacy
import json
import pandas as pd
import regex as re
import numpy as np #, torch
# from transformers import pipeline
# import networkx as nx
# import matplotlib.pyplot as plt
from afinn import Afinn
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain 
import string
from collections import Counter, defaultdict
from typing import Tuple

# for scraping
from bs4 import BeautifulSoup
import requests
from googlesearch import search

class Book_content_analyzer:

    def __init__(self, spacy_model='en_core_web_sm') -> None:

        self.nlp = spacy.load(spacy_model, disable=["ner","pos", 'lemmatizer'])


    def clean_content(self, book_content:str, cu_patterns_to_remove:list = []) -> str:

        # remove punctuations
        punctuation = re.sub(r"[\?\!\,\.\'\,\:\;]", '', string.punctuation)
        book_content = book_content.translate(str.maketrans('', '', punctuation))

        # remove patterns
        for patt in cu_patterns_to_remove:
            book_content = re.sub(patt,  '', book_content)
            
        # substitute patterns
        sub_1 = ["\'", "\n", '-']
        sub_2 = ["'", ' ', '']

        for i in range(len(sub_1)):
            book_content = re.sub(sub_1[i], sub_2[i], book_content)

        return book_content

                
    def spacy_detect_sentences(self, corpus:str)->list:

        """
        Detect sentences with spacy outputs the list of sentences
        """
        Doc = self.nlp(corpus)
        sentences = [str(i) for i in list(Doc.sents)]

        return sentences


    def clean_sentences(self, sentences:list, chapter_regex:str = 'no chapter')->list:
        '''
        check for "chapter" and separate it from its span
        flatten this list of sentences
        outputs correct sentences
        '''    

        # without chapters
        if chapter_regex=='no chapter': return sentences

        # with chapter
        elif chapter_regex:
            for i, sent in enumerate(sentences): 
                if re.findall(chapter_regex,  sent):
                    sentences[i] = re.split(f' *({chapter_regex})+ *' ,sent)

            finalized_sents = []
            # flatten the lists (these were created after separating the chapter regex from sents)
            for i, sent in enumerate(sentences):
                if type(sent) == list:
                    for s in sent: finalized_sents.append(s.strip())
                else: finalized_sents.append(sent.strip())

    
            unwanted_sents_idx = []
            for i, sentence in enumerate(finalized_sents): 

                # chapter regex and number after it
                if re.match(f'.*{chapter_regex} *$', sentence.strip()):
                    unwanted_sents_idx.append(i+1)
                    finalized_sents[i] = finalized_sents[i] + finalized_sents[i+1]
                else:
                    unwanted_sents_idx.append(i)

            for i in reversed(unwanted_sents_idx):
                del finalized_sents[i]

            return finalized_sents


    # sentiment analysis --------------------------------------------------------------
    def decide_for_transformer_sentiment_label(self, senti_dicts:dict) ->list:
        """
        takes in dictionaries of sentiment analysis
        in this format: {'label': '...', 'score':...} 
        (label: 'POSITIVE'/'NEGATIVE', 'score': 0<s<1)
        outputs a list representing the emotion of each sentence
        """
        labels = []

        for senti_dict in senti_dicts:

            # senti_dict['score'] = how certain the model is about an emotion
            if float(senti_dict['score'])< 0.7:
                labels.append('NEUTRAL')

            else: labels.append(senti_dict['label'])

        # encode the labels
        emotions_label = {'POSITIVE': 1, 'NEUTRAL':0, 'NEGATIVE':-1}

        # encode the three types
        encoded_lables = [emotions_label[em] for em in labels]
        

        return labels, encoded_lables


    def senti_analysis_transformers(self, sentences:list, plot=False):


        """
        given the fact that we have GPU
        We empty the cache and initialize our sentiment analysis
        """
        torch.cuda.empty_cache()

        sentiments_lists = []

        #classifier = pipeline('sentiment-analysis', device=0)
        # will be added after immigration ==================================================

        # we use small batches to prevent crashing
        for i in range(0, len(sentences), 50):
            # if i%1000==0 : print('sentence', colored(f"{i}", 'blue'))
            sentiments_lists.append(classifier(sentences[i: i+50]))
        
        # list of list -> list
        all_sentences_sentiment = list(chain.from_iterable(sentiments_lists))

        labels, encoded_labels = self.decide_for_transformer_sentiment_label(all_sentences_sentiment)

        
        # count the labels
        emotions_count = dict(Counter(labels))
        #print(colored('Emotions dominance: \n', 'blue'), emotions_count)
        
        #if plot:plot_emotions(emotions_count)
        
        return labels, encoded_labels, emotions_count


    def senti_analysis_Afinn(self, sentence_list:list):
        '''
        Function to calculate the align_rate of the whole novel
        param sentence_list: the list of sentence of the whole novel.
        '''
        afn = Afinn()

        # encoded_labels = sentiments_lists in transformers
        encoded_labels = []

        # the score is divided to be compatible with the transformers approach
        # |max or min Afinn| = 5
        for sent in sentence_list:
            encoded_labels.append(afn.score(sent)/5)
        

        labels = sentiment = [
            'POSITIVE' if score > 0 
            else 'NEGATIVE' if score < 0 
            else 'NEUTRAL' 
            for score in encoded_labels
            ]

        
        # count the labels
        emotions_count = dict(Counter(labels))
        
        return labels, encoded_labels, emotions_count


    # ner ----------------------------------------------------------------------------------
    def find_most_pop_names(self, list_sents:list)->dict:
        '''
        first for loop: takes out the names
        second for loop: removes the honorary and adds it to the dictionary
        output: a sorted dictionary based on values which are the number of occurrences for each name
        {'Mit mirshafiee': 5, ...}
        '''
        self.nlp.enable_pipe("ner")

        titles = ['Great Uncle', 'Uncle', 'Aunt', "'s", "Mister", "Mistress", "professor", r' \w\. *']

        names_dict = defaultdict(int)
        for doc in self.nlp.pipe(list_sents):
            for ent_ in doc.ents:
                if ent_.label_ == 'PERSON':

                    # remove ...'d ...'m from list of names expample: "who'd"
                    initial_name = ent_.text
                    if re.findall(pattern=f" *[{string.punctuation}’”]\w *", string=initial_name):
                        pass
                    else:
                        name_per = ent_.text.strip()

                        for title in titles: 
                            name_per = re.sub(title, '', name_per)
                        names_dict[name_per] += 1
                            
        names_dict = {k:v for k, v in sorted(names_dict.items(), key=lambda i : i[1], reverse=True)}
        return names_dict


    def add_or_remove_names(self, list_sents:list, names_dict:dict, unwanted_names:str, missing_names:str)->dict:
        """
        inputs the previous dictionary and the name that was not recognized
        I can also use tfidf, but I think this is a better approach
        """
        
        names_dict = defaultdict(int, names_dict)

        if unwanted_names:
            unwanted_names_in_dict = []
            dict_names = list(names_dict.keys())
            unwanted_names = [name.strip() for name in list(unwanted_names.split(','))]

            for dict_name in dict_names:
                for name in unwanted_names:

                    if name.lower() == dict_name.lower():
                        unwanted_names_in_dict.append(dict_name)
                        del names_dict[dict_name]


        if missing_names:
            missing_names_list = [name.strip() for name in list(missing_names.split(','))]

            for name in missing_names_list:
                for sent in list_sents:
                    matches = re.findall(pattern=name.strip(), string=sent)
                    if matches:
                        names_dict[name] += len(matches)

        new_names_dict = {k:v for k, v in sorted(names_dict.items(), key=lambda i : i[1], reverse=True)}

        return new_names_dict


    def flatten_names(self, names_dict:dict) -> dict:
        """
        looks for names with two parts in the dict and saves only the firs part
        ('Mit mirshafiee' -> 'Mit')
        {'Mit mirshafiee': 5, 'Mit': 6} -> {'Mit': 11}
        """

        unwanted = []
        items_ = list(names_dict.items())

        for i, (k, v) in enumerate(items_):
            splitted = k.split(' ')
            if len(splitted)>1:
                name = splitted[0].strip()
                unwanted.append(k)
                
                if name in names_dict.keys():
                    names_dict[name] += names_dict[k]
                else: names_dict[name] = 1
            
        for k in unwanted: del names_dict[k]
        #selected = names_dict[:threshold]

        return names_dict


    def _zero_diag(self, mat:np.matrix) -> np.matrix:
        diag_range = mat.shape[0]
        mat[[range(diag_range)], [range(diag_range)]] = 0
        return mat
    def _zero_below_threshold(self, mat:np.matrix, threshold:int) -> np.matrix:
        # zero out when the characters don't have a certain number of sentences
        # in which they both appeared
        for [i,j] in  np.argwhere(mat<=threshold) :
            mat[i,j] = 0
        return mat
    def _divide_by_max(self, mat:np.matrix) -> np.matrix:
        return mat / np.max(np.abs(mat))
    def _reduce_numbers(self, mat)->np.matrix:

        """
        reduces the number of decimals that appear after each number
        input: mat = [[1.222, 5.2222]]
        output: mat = [[1.2, 5.2]]
        """
        s1 = mat.shape[0]
        s2 = mat.shape[1]
        for i in range(s1):
            for j in range(s2):
                mat[i,j] = round(mat[i,j], 3)

        return mat


    def create_cooccurrence_matrices(self, top_n_popular_names:list, book_sents:list, encoded_senti_labels:list,
     normalize_mode=True, threshold = 2)-> Tuple[np.matrix, np.matrix, np.matrix]:

        """
        inputs the popular names and book sents
        creates tfidf matrix, filters names, and creates a n x n matrix of names
        then removes the bottum half of the matrix
        threshold: min number of times two names are seen together
        outputs two matrixes: one is affected by the sentiments another is not
        """
        
        # first create the sentence - count matrix 
        # doc_word = (documents, words)
        count_model = CountVectorizer(ngram_range=(1,1), token_pattern=r"[A-Za-z]+", lowercase=False)
        doc_word = count_model.fit_transform(book_sents)

        # check for number of sentences
        assert doc_word.shape[0] == len(book_sents) # num docs = num sents

        # create a smaller version by filtering the words 

        # name the columns of the array
        count_df = pd.DataFrame(doc_word.toarray(),
                                columns=count_model.get_feature_names())


        # popular words filtered out
        pop_names_df = count_df.loc[:, top_n_popular_names]
        
        # create the co occurrence matrix and remove one half of it
        cooccurrence_matrix = np.dot(pop_names_df.T, pop_names_df) # can i remove this?
        cooccurrence_matrix = self._zero_below_threshold(cooccurrence_matrix, threshold=threshold)
        
        # multiply frequencies of words in sentences with the sentence sentiments
        count_df_with_sentiments = np.multiply(pop_names_df.to_numpy() , np.array(encoded_senti_labels).reshape(-1,1))

        # co occurrence with sentiments
        cooccurrence_matrix_with_senti = np.dot(count_df_with_sentiments.T, count_df_with_sentiments)
            
        # setting the diagonal axis to zero 
        cooccurrence_matrix_with_senti = self._zero_diag(cooccurrence_matrix_with_senti)
        cooccurrence_matrix = self._zero_diag(cooccurrence_matrix)

        if normalize_mode == True:
            
            cooccurrence_matrix_with_senti = self._divide_by_max(cooccurrence_matrix_with_senti)
            cooccurrence_matrix = self._divide_by_max(cooccurrence_matrix)

        cooccurrence_matrix = self._reduce_numbers(cooccurrence_matrix)
        cooccurrence_matrix_with_senti = self._reduce_numbers(cooccurrence_matrix_with_senti)


        return pop_names_df, cooccurrence_matrix, cooccurrence_matrix_with_senti


    def matrix_to_edge(self, cooccurrence_matrix:np.matrix, cooccurrence_matrix_with_senti:np.matrix,
     pop_names_df:pd.DataFrame, top_n_popular_names:list)->Graph:

        graph_ = {'nodes':[], 'links':[]}
        shape = cooccurrence_matrix.shape[0]
        name_freq = pop_names_df.sum()


        # for nodes
        # {'nodes' : [{'id': , 'group':2 , 'size':name_freq[char]}]}
        for char in top_n_popular_names:
            graph_['nodes'].append({'id': char, 'group': 1, "size":np.float(np.log(name_freq)[char])*5 })


        # for edges
        # {'links' : [{'source': , 'target': , 'value': , 'color': }]}
        for i in range(shape):
            for j in range(shape):
                if i>j:
                    graph_['links'].append({'source': top_n_popular_names[i], 
                                            'target': top_n_popular_names[j],
                                        'value': np.float(cooccurrence_matrix[i, j]), 
                                        'color':cooccurrence_matrix_with_senti[i,j]})
                    
        return graph_



    def create_plot_df(self, top_n_popular_names:list, pop_names_df:pd.DataFrame, n_sections:int=5)->pd.DataFrame:
        """
        create a dataset that 
        df.columns = [names of chars + section i]
        """
        df_sectioned = pd.DataFrame(columns=top_n_popular_names)
        df_len = len(pop_names_df)
        n_sections = int(n_sections)
        
        hop = int(np.round(df_len/n_sections)+1)
        j=0

        for i in range(0, df_len, hop):
            df_sectioned.loc['section '+ str(j+1),:] = pop_names_df.loc[j*hop:(j+1)*hop, :].sum()
            j+=1

        df_sectioned = df_sectioned.T.reset_index(drop=False)
        df_sectioned.rename(columns={'index': 'characters'}, inplace=True)

        df_sectioned = df_sectioned.melt(
            id_vars="characters",
            var_name="section", 
            value_name="occurrence")


        # create the graph ---------------------------------------------
        fig = px.bar(df_sectioned, x="section", y='occurrence',
                    color='characters', barmode='group',
                    height=400)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON



class Book_info_scraper:

    def __init__(self) -> None:
        pass
    
    def find_genres(self, soup):

        all_genres = []
        for x in soup.find_all("a", { "class" : "actionLinkLite bookPageGenreLink" }):
            all_genres.append(x.get_text())
        all_genres = [genre for genre in all_genres if genre.lower()!= 'audiobook']

        return all_genres


    def find_reviews_and_ratings(self, soup):

        text = soup.find_all("div", { "class" : "reviewControls--left greyText" })[0].get_text().strip()

        text = re.sub('\n', ' ', text)
        text = re.sub('( \s+)', ' ', text)

        ratings = re.findall('[\d\,]* ratings',text)[0]
        reviews = re.findall('[\d\,]* reviews',text)[0]

        return reviews, ratings

    def find_author(self, soup):
        return soup.find_all("a", { "class" : "authorName" })[0].get_text()

    def find_year_published(self, soup):
    
        text = soup.find_all("div", { "class" : "uitext darkGreyText"})[0].get_text()

        text = re.sub('\n', ' ', text)
        text = re.sub('( \s+)', ' ', text)
        year_published = re.findall('Published \w+ \w+ (\d+)', text)[0]
        
        return year_published

    def get_goodreads_info(self, book_name):

        query = book_name + ' goodreads'

        url = next(search(query, tld="co.in", num=1, stop=1))
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, 'html.parser')

        genres = self.find_genres(soup)
        reviews, ratings = self.find_reviews_and_ratings(soup)
        author = self.find_author(soup)
        year_published = self.find_year_published(soup)

        return genres, reviews, ratings, author, year_published

