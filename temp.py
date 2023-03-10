import string
from flashtext import KeywordProcessor
from sense2vec import Sense2Vec
from collections import OrderedDict
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.corpus import stopwords
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import pke.unsupervised
import torch
import spacy

def MCQs_available(word,s2v):
    '''
    Takes in a word nd tells the sense of the word. Example- boy-NOUN, John-PROPN(proper noun)
    '''
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    return (not (sense==None))


def edits(word):
    '''takes in a string and returns all the possible strings with an edit distance of one'''
    ''' string.punctuation --> is a pre-initialized string  used as string contant and it returns all the sets of punctuation'''
    
    '''alph_punc --> storing all the alphabets and sets of punctuation( combiningly )'''
    alph_punc=string.ascii_lowercase+" "+string.punctuation
    '''word_split --> list of tuple storing the splitting of word into left and right part around a given pivot  
       For Example : "a b" 
       word_split stores : [('','a b'),('a',' b'),('a ','b'),('a b','')]'''
    size=len(word)
    word_split=[]
    for j in range(size+1):
        left=word[:j]
        right=word[j:]
        word_split.append((left,right))    
    ''' del_word --> list of word by deleting the first character of right substring but right substring must be 
    non-empty '''
    del_word=[]
    for left,right in word_split:
        if(len(right)!=0):
            fin_word=left+right[1:]
            del_word.append(fin_word)
    '''adj_word --> list of words by swapping the adjacent character in every non-empty  right substring of a given 
    input word . '''
    adj_word=[]
    for left,right in word_split: 
        if(len(right)>1):
            fin_word=left+right[1]+right[0]+right[2:]
            adj_word.append(fin_word)
    '''rep_word --> list of words by substituting all available letters from a predetermined set of letters for each character
    in each non-empty right-side substring of a given input word'''
    rep_word=[]
    for left,right in word_split:
        if(len(right)!=0):
            for i in alph_punc:
                fin_word=left+i+right[1:]
                rep_word.append(fin_word)
    ''' ins_word --> list of word by substituting all available letters from a ppredetermined '''
    ins_word=[]
    for left,right in word_split:
        for alphabet in alph_punc:
            fin_word=left+alphabet+right
            ins_word.append(fin_word)
    return set(del_word+adj_word+rep_word+ins_word) 


def sense2vec_get_words(word,s2v):
    '''
    Takes a word and returns all the words in the database that are similar to the given word
    with the same sense. In case of multiple words it returns at most 15 words ordered in 
    descending order by their closeness. The best match word comes before.
    '''
    output = []
    word_preprocessed =  word.translate(word.maketrans("","", string.punctuation))
    word_preprocessed = word_preprocessed.lower()
    word_edits = edits(word_preprocessed)
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    if sense==None:
        return None,"None"
    most_similar = s2v.most_similar(sense, n=15)
    compare_list = [word_preprocessed]
    for each_word in most_similar:
        append_word = ((each_word[0].split("|")[0].replace("_", " ")).strip()).lower()
        append_word = append_word.translate(append_word.maketrans("","", string.punctuation))
        if append_word not in compare_list and word_preprocessed not in append_word and append_word not in word_edits:
            output.append(append_word.title())
            compare_list.append(append_word)
    out = list(OrderedDict.fromkeys(output))
    print(out)
    return out,"sense2vec"


def tokenize_sentences(text):
    ''' tokenize_sentences --> function takes text as a input string and performs string tokenization on it i.e
    divide it into individual sentence '''
    #  print(sent_tokenize(text))
    '''sent_tokenize() --> split the text into individual sentences'''
    sent= sent_tokenize(text)
    ''' fin_sen() --> storing those sentence having length greater than 20 /'''
    fin_sent=[]
    for i in sent:
        if(len(i)>20):
            fin_sent.append(i)
    return fin_sent


def is_far(words_list,currentword,threshold):
    '''
    Calculates the normalized edit distance between the currentword and elements of words_list.
    if all the words have a distance greater than a threshhold then returns true else false
    '''
    min_score=1000000000
    for word in words_list:
        min_score=min(min_score,NormalizedLevenshtein().distance(word.lower(),currentword.lower()))
    return min_score>=threshold


def get_sentences_for_keyword(k, s):
    kp = KeywordProcessor()
    ans = {}
    #initializing dictionary
    for i in k:
        i = i.strip()
        kp.add_keyword(i)
        ans[i] = []
    #extracting
    for i in s:
        ext = kp.extract_keywords(i)
        for j in ext:
            ans[j].append(i)
    #sorting and storing
    for i in ans.keys():
        ans[i] = sorted(ans[i], key=len, reverse=True)
    #deleting unnecessary keys
    dele=[k for k in ans.keys() if len(ans[k])==0]
    for i in dele:
      del ans[i]

    return ans


def filter_phrases(ph_keys, max_phrases):
    """
    Filter a list of phrases to a maximum number based on a scoring metric.

    Args:
        ph_keys (list): A list of phrases to filter.
        max_phrases (int): The maximum number of phrases to return.

    Returns:
        list: The filtered list of phrases, containing at most max_phrases phrases.
    """
    ans = [ph_keys[0]] if len(ph_keys) > 0 else []
    for phrase in ph_keys[1:]:
        if is_far(ans, phrase, 0.71) and len(ans) < max_phrases:
            ans.append(phrase)
        if len(ans) >= max_phrases:
            break
    return ans



def get_nouns_multipartite(text):
    """
    Extract the top 10 noun and proper noun phrases from a text using the MultipartiteRank algorithm.

    Args:
        text (str): The input text.

    Returns:
        list: The top 10 noun and proper noun phrases extracted from the text.
    """
    ans = []

    try:
        e = pke.unsupervised.MultipartiteRank()
        e.load_document(input=text, language='en',stoplist=list(string.punctuation) + stopwords.words('english'))
        e.candidate_selection(pos={'PROPN', 'NOUN'})
        e.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyph = e.get_n_best(n=10)
        ans=[key[0] for key in keyph]
    except:
        pass

    return ans


def get_phrases(doc):
    """
    Extract up to 50 longest noun phrases from a spaCy document object.

    Args:
        doc (spacy.tokens.Doc): The spaCy document object.

    Returns:
        list: The up to 50 most frequent noun phrases in the document.
    """
    ph = {}
    for noun in doc.noun_chunks:
        p = noun.text
        if len(p.split()) > 1:
            if p not in ph:
                ph[p] = 1
            else:
                ph[p] += 1

    ph_keys = sorted(list(ph.keys()), key=lambda x: len(x), reverse=True)[:50]
    return ph_keys

def get_keywords(nlp, text, max, s2v, fd, nos):
    """
    Extract up to max_keywords keywords from a given text using a combination of
    approaches, including MultipartiteRank, noun phrases, and filtering.

    Args:
        nlp : The  model to use for text processing.
        text (str): The input text to extract keywords from.
        max (int): The maximum number of keywords to return.
        s2v : The sense2vec model to use for filtering out irrelevant keywords.
        fd : A frequency distribution of words in the text.
        nos (int): The number of sentences in the input text.

    Returns:
        list: The up to max_keywords most relevant keywords extracted from the text.
    """

    tp = filter_phrases(sorted(get_nouns_multipartite(text), key=lambda x: fd[x]), int(max)) + filter_phrases(get_phrases(nlp(text)), int(max))

    tpf = filter_phrases(tp, min(int(max), 2*nos))

    ans = []
    for answer in tpf:
        if answer not in ans:
          if MCQs_available(answer, s2v):
            ans.append(answer)

    ans = ans[:int(max)]
    return ans


def generate_questions_mcq(keyword_sent_mapping,device,tokenizer,model,sense2vec):
    batch_text = []
    answers = keyword_sent_mapping.keys()
    for answer in answers:
        txt = keyword_sent_mapping[answer]
        print(txt)
        context = "context: " + txt
        text = context + " " + "answer: " + answer + " </s>"
        batch_text.append(text)
    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")
    print ("Running model for generation")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    with torch.no_grad():
        outs = model.generate(input_ids=input_ids,attention_mask=attention_masks,max_length=150)
    output_array ={}
    output_array["questions"] =[]
#     print(outs)
    for index, val in enumerate(answers):
        individual_question ={}
        out = outs[index, :]
        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        Question = dec.replace("question:", "")
        Question = Question.strip()
        individual_question["question_statement"] = Question
        individual_question["question_type"] = "MCQ"
        individual_question["answer"] = val
        individual_question["id"] = index+1
        individual_question["options"], individual_question["options_algorithm"] = sense2vec_get_words(val, sense2vec)

        individual_question["options"] =  filter_phrases(individual_question["options"], 10)
        index = 3
        individual_question["extra_options"]= individual_question["options"][index:]
        individual_question["options"] = individual_question["options"][:index]
        individual_question["context"] = keyword_sent_mapping[val]
     
        if len(individual_question["options"])>0:
            output_array["questions"].append(individual_question)

    return output_array


s2v=Sense2Vec().from_disk("s2v_old")
text="Diophantus, the “father of algebra,” is best known for his book Arithmetica, a work on the solution of algebraic equations and the theory"
text+=" of numbers. However, essentially nothing is known of his life, and"
text+=" there has been much debate regarding precisely the years in which"
text+=" he lived. Diophantus did his work in the great city of Alexandria. At"
text+=" this time, Alexandria was the center of mathematical learning. The period "
text+="from 250 bce to 350 ce in Alexandria is known as the Silver Age, also the Later "
text+="Alexandrian Age. This was a time when mathematicians were discovering many ideas "
text+="that led to our current conception of mathematics. The era is considered silver "
text+="because it came after the Golden Age, a time of great development in the field "
text+="of mathematics. This Golden Age encompasses the lifetime of Euclid."
nlp = spacy.load('en_core_web_md')
doc = nlp(text)
f=FreqDist(brown.words())
k=get_keywords(nlp,text,10,s2v,f,10)
sentences=tokenize_sentences(text)
ksm=get_sentences_for_keyword(k,sentences)
for lund in ksm.keys():
    text_snippet = " ".join(ksm[lund][:3])
    ksm[lund] = text_snippet
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('Parth/result')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
final=generate_questions_mcq(ksm,device,tokenizer,model,s2v)
print(final)
