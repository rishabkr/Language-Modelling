
import numpy as np
import re
import os
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import sys
from sklearn.model_selection import train_test_split


class Tokenizer:
    def __init__(self,use_regex=False,punctuations=[',','?','!',':','-','\'']):
        self.corpus=None
        self.use_regex = use_regex
        self.punctuations=punctuations
        self.re_pattern=re.compile(r'\b\S+\b')
        
        self.tokenized_corpus=[]
    
    
    def custom_tokenizer(self,sentence):      
        for punct in self.punctuations:
            if punct in sentence:
                sentence=sentence.replace(punct,"")
        
        custom_tokens=sentence.split()
        return custom_tokens
        
    def regex_tokenizer(self,sentence):
        tokens_re=re.findall(r'\b\S+\b',sentence)
        return tokens_re
    
    def tokenize(self,corpus):
        self.corpus=corpus
        #print(len(self.corpus))
        
        if self.use_regex == True:
            print('using regex')
            for sentence in self.corpus:
                re_tokens=self.regex_tokenizer(sentence)
                self.tokenized_corpus.append(re_tokens)
                
        else:
            #print('nt')
            sentences=[]
            for sent in self.corpus:
                if sent.endswith("."):
                    sentences.append(sent[:-1])
                else:
                    sentences.append(sent)
            for sentence in self.corpus:
                tokens=self.custom_tokenizer(sentence)
                self.tokenized_corpus.append(tokens)
            
        return self.tokenized_corpus
    
    def sentence_tokenizer(self,sentence):
        sentence_tokens=[]
        if self.use_regex==True:
            sentence_tokens=self.regex_tokenizer(sentence)
        else:
            sentence_tokens=self.custom_tokenizer(sentence)
        return sentence_tokens
    
    def get_sentence_n_grams(self,tokenized_sentences,n_gram,sos='<s>',eos='</s>'):
        sentence_n_grams=list()

        for tokenized_sentence in tokenized_sentences:
            for i in range(0,n_gram-1):
                tokenized_sentence.insert(0,sos)
                tokenized_sentence.append(eos)

            length=len(tokenized_sentence)
            sentence_n_gram=[]

            for n in range(length - (n_gram-1)):
                sentence_n_gram.append(tokenized_sentence[n : n + n_gram])

            sentence_n_grams.extend(sentence_n_gram)

        return sentence_n_grams
    
    
    def get_n_gram_frequency(self,tokenized_list,n_gram,sos='<s>',eos='</s>'):
        frequency=defaultdict(int)
        reverse_freq=defaultdict(lambda: defaultdict(int))
        if n_gram == 1:
            for tokenized_sentence in tokenized_list:
                for token in tokenized_sentence:
                    frequency[token]+=1
        else:
            temp_sentences=copy.deepcopy(tokenized_list)
            sentence_n_grams =  self.get_sentence_n_grams(temp_sentences,n_gram,sos,eos)
            for n_grams in sentence_n_grams:
                    frequency[tuple(n_grams)]+=1
                    
                    key1=tuple(n_grams[:len(n_grams)-1])
                    key2=n_grams[-1]
                    
                    #print((key1,key2))
                    reverse_freq[key1][key2]+=1
                    

        return frequency,reverse_freq
    
    def n_grams_without_tags(self,tokenized_list=[],n_gram=4):
            num_tokens=len(tokenized_list)
            n_gram_tokens=[]
            
            for n in range(num_tokens-(n_gram-1)):
                n_gram_tokens.append(tokenized_list[n : n + n_gram])
            
            return n_gram_tokens

    
    def tokenize_to_n_grams(self,tokenized_list=[],inference=False,n_gram=4,use_tags=True,sos='<s>',eos='</s>'):
        
        if inference==True:
            if(use_tags == True):
                for i in range(0,n_gram-1):
                    tokenized_list.insert(0,sos)
                    tokenized_list.append(eos)

            num_tokens=len(tokenized_list)
            n_gram_tokens=[]
            
            for n in range(num_tokens-(n_gram-1)):
                n_gram_tokens.append(tokenized_list[n : n + n_gram])
            
            return n_gram_tokens
        
        
        n_gram_frequencies=list()
        n_gram_frequencies.append([])
        
        reverse_n_grams=list()
        reverse_n_grams.append([])

        for i in range(n_gram):
            n_gram_frequencies.append(defaultdict(int))
        
        for i in range(n_gram):
            reverse_n_grams.append(defaultdict(lambda: defaultdict(int)))
        
        
        if use_tags == False:
            for i in range(1,n_gram+1):
                reverse_freq=defaultdict(lambda: defaultdict(int))
                
                n_gram_sentences=defaultdict(int)
                
                for tokenized_sentence in tokenized_list:
                    tokenized_n_grams=self.n_grams_without_tags(tokenized_sentence,i)
                    for ngram in tokenized_n_grams:
                        if i == 1:               
                            n_gram_sentences[ngram[0]]+=1
                        else:
                            n_gram_sentences[tuple(ngram)]+=1
                            
                            key1=tuple(ngram[:len(ngram)-1])
                            key2=ngram[-1]
                    
                            #print((key1,key2))
                            reverse_freq[key1][key2]+=1
                            
                n_gram_frequencies[i] = n_gram_sentences
                reverse_n_grams[i]=reverse_freq
        else:
            for i in range(1,n_gram+1):
                freqs,rev_freqs=self.get_n_gram_frequency(tokenized_list,i,sos,eos)
                n_gram_frequencies[i]=freqs
                reverse_n_grams[i]=rev_freqs

        return n_gram_frequencies,reverse_n_grams


class lm_smoothing:
    def __init__(self,tokenized_list,n_gram_frequencies,reverse_frequencies):
        
        self.number_of_tokens=0
        self.vocab_size=0
        dd=defaultdict(int)
        
        for sentence in tokenized_list:
            for token in sentence:
                dd[token]+=1
                self.number_of_tokens+=1;
                
        self.vocab_size=len(dd)
        self.n_gram_frequencies=n_gram_frequencies
        self.reverse_frequencies = reverse_frequencies

        self.one_word_contexts=[]
        for i in range(1,5):
            self.one_word_contexts.append(defaultdict(int))
    
    
        for i in range(2,4):
            for ngrams in self.n_gram_frequencies[i]:
                rhs=ngrams[1:]
                self.one_word_contexts[i][rhs]+=1
        
    def P_kneser_ney_base_condn(self,W_i,highest_order,discount):        
        #unigram_case
        if highest_order == True:
            word_freq=self.n_gram_frequencies[1][W_i]
            
            if word_freq == 0:
                return discount*self.vocab_size/self.number_of_tokens
            
            return word_freq*self.vocab_size/self.number_of_tokens
        else:
            #bigram_case
                 
            bigram_occurances=sum([1 for bigrams in self.n_gram_frequencies[2]
                                  if bigrams[1] == W_i])
            
            total_number_of_bigrams = len(self.n_gram_frequencies[2])
            
            if bigram_occurances == 0:
                
                return discount*self.vocab_size/self.number_of_tokens

            return bigram_occurances / total_number_of_bigrams 
                    
                    
    
    def P_kneser_ney(self,W_i,W_prev,highest_order,discount=0.65):
        
        def match_ngrams(seq,match,start,end):
            seq_list=list(seq)
            seq_len=len(seq_list)
            to_match = tuple(seq_list[start: seq_len-end])
           
            if to_match == match :
                return True
            else:
                return False
        
        def is_a_single_token(words):
            if len(words) == 1:
                return True
            else:
                return False
                
        
        def get_lambda_new(W_prev,N_gram_total):
            
            count_preceeding=len(self.reverse_frequencies[N_gram_total][tuple(W_prev)])
            
            if count_preceeding == 0:
                count_preceeding = 1
            
            return count_preceeding

            
            
        if len(W_prev) == 0 :
            return self.P_kneser_ney_base_condn(W_i,highest_order,discount)
        
        all_words = [x for x in W_prev]
        all_words.append(W_i)
        
        N_gram_total= len(all_words)
        N_gram_prev= len(W_prev)
        n_gram_words=tuple(all_words)
        
        if highest_order == True:
            C_kn_all=0
            if n_gram_words in self.n_gram_frequencies[N_gram_total]:
                C_kn_all= max(self.n_gram_frequencies[N_gram_total][n_gram_words] - discount ,
                      0)
        else:
            C_kn_all=0
            following_words = tuple(all_words[1:])
            # C_kn_all=sum([1 for current_ngram in self.n_gram_frequencies[N_gram_total] 
            #               if(match_ngrams(current_ngram,following_words,1,0) == True)])

            C_kn_all= self.one_word_contexts[N_gram_total][following_words]

            C_kn_all=max(C_kn_all - discount, 0)

        
        n_gram_prev=tuple(W_prev)
        
        if is_a_single_token(n_gram_prev) == True:
            n_gram_prev=n_gram_prev[0]
            
        if highest_order == True:
            C_kn_prev = 0
            
            if n_gram_prev in self.n_gram_frequencies[N_gram_prev]:
                C_kn_prev = self.n_gram_frequencies[N_gram_prev][n_gram_prev]
            else:
                C_kn_prev= 1 #prevent divide by 0 and allow unseen 
        else:
            C_kn_prev = len(self.n_gram_frequencies[N_gram_total])
        
        lambda_prev2= get_lambda_new(W_prev,N_gram_total)
        
        #print((lambda_prev1,lambda_prev2))
        
        if C_kn_prev == 0 :
            C_kn_prev = 1
            
        #print(f'C_kn_all : {C_kn_all} , C_kn_prev: {C_kn_prev} , lambda :{lambda_prev}')
        
        if C_kn_all > C_kn_prev:
            C_kn_prev = C_kn_all
        
        if lambda_prev2 == 0 :
            if(C_kn_all == 0):
                C_kn_all = 1
            return C_kn_all / C_kn_prev
                
        normalized_lambda = lambda_prev2 / C_kn_prev
        
        P_cont = self.P_kneser_ney(W_i,W_prev[1:],highest_order=False,discount=discount)
        
        res= (C_kn_all / C_kn_prev) + (discount * normalized_lambda * P_cont)
        
        #print(f'non-zero lambda C_kn_all : {C_kn_all} , C_kn_prev: {C_kn_prev} , lambda :{lambda_prev2} P_cont : {P_cont} Res: {res}')
        
        return (C_kn_all / C_kn_prev) + (discount * normalized_lambda * P_cont)
            

    def P_witten_bell(self,W_i,W_prev):
        def c(W): 
            if len(W) == 1:
                return self.n_gram_frequencies[1][W[0]]
            
            return self.n_gram_frequencies[len(W)][tuple(W)]
        
        
        def get_N_1_plus_new(W_prev,N_gram_total):
            
            count_preceeding=len(self.reverse_frequencies[N_gram_total][tuple(W_prev)])
            
            if count_preceeding == 0:
                count_preceeding = 1
            
            return count_preceeding

        if len(W_prev) == 0 :
            c_tot= self.n_gram_frequencies[1][W_i[0]]
            tot_gram =  self.number_of_tokens
    
            if c_tot == 0:
                c_tot = 1
            return c_tot / tot_gram
        
        c_total=c(W_prev + W_i)
        
        N_gram_total = len(W_prev + W_i)
    
        n_prev1 = get_N_1_plus_new(W_prev,N_gram_total)
        
        #n_prev2 = get_N_1_plus(W_prev,N_gram_total)
      
        total_n_gram =  sum([val for key,val in self.reverse_frequencies[N_gram_total][tuple(W_prev)].items()])
        

        P_wb_prev = self.P_witten_bell(W_i,W_prev[1:])

        return (c_total + n_prev1 * P_wb_prev )/(total_n_gram + n_prev1)
        
    def calculate_perplexity_exp(self,P):
        #print(((P,P * np.log(P))/np.log(2)))
        return (P * np.log(P))/np.log(2)
    
    def perplexity_kneser_ney(self,current_sentence,n_gram,discount=0.65):
        
        kn_tokenizer=Tokenizer()
        tokenized_sentence=kn_tokenizer.sentence_tokenizer(current_sentence)
        #print(tokenized_sentence)
        
        n_gram_tokens=kn_tokenizer.tokenize_to_n_grams(tokenized_sentence,inference=True,use_tags=True,n_gram=n_gram)
        
        P_kn=1
        
        perplexity_exp=0
        for n_grams in n_gram_tokens:

            current_word=n_grams[-1]
            previous_words=n_grams[:-1]
            
            #print(P_kn)
            
            P_kn*=self.P_kneser_ney(current_word,previous_words,highest_order=True,discount=discount)
            
            if P_kn == 0:
                continue
            #print(P_kn,perplexity_exp)
                
            perplexity_exp += self.calculate_perplexity_exp(P_kn)
        
        overall_perplexity = 2 ** (-perplexity_exp)
      
        
        return overall_perplexity,P_kn
    
    
    
    def perplexity_witten_bell(self,current_sentence,n_gram):
        
        wb_tokenizer=Tokenizer()
        tokenized_sentence=wb_tokenizer.sentence_tokenizer(current_sentence)
        
        n_gram_tokens=wb_tokenizer.tokenize_to_n_grams(tokenized_sentence,inference=True,use_tags=True,n_gram=n_gram)
        
        P_wb=1
        
        perplexity_exp=0

        for n_grams in n_gram_tokens:
            current_word=[]
            cw=n_grams[-1]
            current_word.append(cw)
            previous_words=n_grams[:-1]
            
            #print(P_wb)
            P_wb *= self.P_witten_bell(current_word,previous_words)
            if P_wb == 0:
                continue
            #print(P_wb,perplexity_exp)
            
            perplexity_exp += self.calculate_perplexity_exp(P_wb)
        
        overall_perplexity = 2 ** (-perplexity_exp)
        
        return overall_perplexity,P_wb


class language_model:
    def __init__(self,n_gram,smoothing='k'):
        self.n_grams=n_gram
        self.n_gram_freq=None
        self.reverse_n_gram_freq=None
        self.smoothing = smoothing
        self.tokens=None
        self.smoother=None

    def train(self,corpus,use_regex=False,use_tags=False):
        tokenizer=Tokenizer(use_regex=False)

        self.tokens=tokenizer.tokenize(corpus)
        self.n_gram_freq,self.reverse_n_gram_freq = tokenizer.tokenize_to_n_grams(self.tokens,n_gram=self.n_grams,use_tags=use_tags)

    def init_smoother(self):
        self.smoother=lm_smoothing(self.tokens,n_gram_frequencies=self.n_gram_freq,
                reverse_frequencies=self.reverse_n_gram_freq)
        return

    def predict_probab(self,sentence):

        if self.smoothing == 'k':
            perplexity,probability=self.smoother.perplexity_kneser_ney(sentence,self.n_grams,discount=0.65)
        else:
            perplexity,probability=self.smoother.perplexity_witten_bell(sentence,self.n_grams)

        return perplexity,probability


def isEnglish(s):
    return s.isascii()


def filter_corpus(corpus_lines):
    lines_in_file=[]
    for line in corpus_lines:
        if isEnglish(line) == True:
            lines_in_file.append(line)

    return lines_in_file

if __name__=='__main__':
    
    smoothing_type=sys.argv[1]
    corpus_dir=sys.argv[2]

    file=open(corpus_dir,'r',encoding='utf-8')
    corpus_lines=file.readlines()

    corpus = filter_corpus(corpus_lines)

    #inp = input()

    lang_model=language_model(n_gram=4,smoothing=smoothing_type)

    lang_model.train(corpus,use_tags=True)

    lang_model.init_smoother()

    input_sentence=input('Input Sentence :')

    perp,prob=lang_model.predict_probab(input_sentence)

    print(prob)
    #print(perp)

    





