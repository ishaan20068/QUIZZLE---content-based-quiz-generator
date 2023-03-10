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


def edits(token):
    '''
    takes in a string and returns all the possible strings with an edit distance of one 
    '''
    alphabets_and_punctuations = "abcdefghijklmnopqrstuvwxyz " + string.punctuation
    token_split = []

    for i in range(len(token)+1):    # splits token into all possible pairs of left and right substrings
        left = token[:i]
        right = token[i:]
        token_split.append((left, right))

    final_tokens = []
    for left, right in token_split:
        if len(right) != 0:     # list of words by deleting each non-empty right substring of a given input word
            fin_token = left + right[1:]
            final_tokens.append(fin_token)

        if len(right) > 1:      # list of words by swapping the adjacent character in every non-empty right substring of a given word
            fin_token = left + right[1] + right[0] + right[2:]
            final_tokens.append(fin_token)
        
        if len(right) != 0:     # list of words by replacing each character in every non-empty right substring of a given word
            for alpha_punct in alphabets_and_punctuations:
                fin_token = left + alpha_punct + right[1:]
                final_tokens.append(fin_token)
        
        for alphabet in alphabets_and_punctuations:     # list of words by inserting each character in every possible position of every non-empty right substring of a given word
            fin_token = left + alphabet + right
            final_tokens.append(fin_token)
    
    return set(final_tokens)


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


def get_sentences_for_keyword(keywords, sentences):
    '''
    Given some keywords and a set of sentences, it returns a dictionary where the keys are
    the keywords and for each key, the elements are the sentences in which that key is present
    '''
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        word = word.strip()
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    delete_keys = []
    for k in keyword_sentences.keys():
        if len(keyword_sentences[k]) == 0:
            delete_keys.append(k)
    for del_key in delete_keys:
        del keyword_sentences[del_key]

    return keyword_sentences


def filter_phrases(phrase_keys,max):
    filtered_phrases =[]
    if len(phrase_keys)>0:
        filtered_phrases.append(phrase_keys[0])
        for ph in phrase_keys[1:]:
            if is_far(filtered_phrases,ph,0.7):
                filtered_phrases.append(ph)
            if len(filtered_phrases)>=max:
                break
    return filtered_phrases


def get_nouns_multipartite(text):
    '''
    gives best 10 nouns from the text
    '''
    out = []
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='en')
    pos = {'PROPN', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos)
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    try:
        extractor.candidate_weighting(alpha=1.1,threshold=0.75,method='average')
    except:
        return out
    keyphrases = extractor.get_n_best(n=10)
    for key in keyphrases:
        out.append(key[0])
    return out


def get_phrases(doc):
    '''
    Gives the longest phrasal nouns of a text stored in a dictionary format with the 
    phrases being the keys and their count being the value
    '''
    phrases={}
    for np in doc.noun_chunks:
        phrase =np.text
        len_phrase = len(phrase.split())
        if len_phrase > 1:
            if phrase not in phrases:
                phrases[phrase]=1
            else:
                phrases[phrase]=phrases[phrase]+1
    phrase_keys=list(phrases.keys())
    phrase_keys = sorted(phrase_keys, key= lambda x: len(x),reverse=True)
    phrase_keys=phrase_keys[:50]
    return phrase_keys


def get_keywords(nlp,text,max_keywords,s2v,fdist,no_of_sentences):
    doc = nlp(text)
    max_keywords = int(max_keywords)
    keywords = get_nouns_multipartite(text)
    keywords = sorted(keywords, key=lambda x: fdist[x])
    keywords = filter_phrases(keywords, max_keywords)
    phrase_keys = get_phrases(doc)
    filtered_phrases = filter_phrases(phrase_keys, max_keywords)
    total_phrases = keywords + filtered_phrases
    total_phrases_filtered = filter_phrases(total_phrases, min(max_keywords, 2*no_of_sentences))
    answers = []
    for answer in total_phrases_filtered:
        if answer not in answers and MCQs_available(answer,s2v):
            answers.append(answer)
    answers = answers[:max_keywords]
    return answers


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
