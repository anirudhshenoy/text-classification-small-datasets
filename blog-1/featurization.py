from nltk import word_tokenize
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
import numpy as np
import multiprocessing
import pandas as pd
import os
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from numpy import hstack
from scipy import sparse
import string
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from pymagnitude import *
from nltk.corpus import stopwords 




def starts_with_number(df):
    is_number = []
    for title in df.title.values:
        if re.findall('[\d*]',' '.join(title.split()[0:2])):
            is_number.append(1)
        else:
            is_number.append(0)
    return np.array(is_number).reshape(-1,1)

def click_bait_phrases(df):
    clickbait_phrases = {
		"A Single" : "A",
		"Absolutely" : "Moderately",
		"Amazing" : "Barely Noticeable",
		"Awesome" : "Probably Slightly Less Boring Than Working",
		"Best" : "Most Unexceptional",
		"Breathtaking" : "Fleetingly Inspirational",
		"But what happened next" : "And As You Expect It",
		"Can change your life" : "Will Not Change Your Life in ANY Meaningful Way",
		"Can't Even Handle" : "Can Totally Handle Without Any Significant Issue",
		"Can't Handle" : "Can Totally Handle Without Any Significant Issue",
		"Cannot Even Handle" : "Can Probably Totally Handle",
		"Doesn't want you to see" : "Doesn't Really Care If You See",
		"Epic" : "Mundane",
        "Everything You Need To Know" : "Something You Don't Need To Know",
        "Gasp-Worthy" : "Yawn-Worthy",
		"Go Viral" : "Be Overused So Much That You'll Silently Pray for the Sweet Release of Death to Make it Stop",
		"Greatest" : "Average",
		"Incredible" : "Painfully Ordinary",
		"Infuriate" : "Mildly Annoy",
		"Literally" : "Figuratively",
		"Mind Blowing" : "Mind-Numbingly Ordinary",
		"Mind-Blowing" : "Painfully Ordinary",
		"Mind BLOWN" : "Meh",
		"Mind Blown" : "Meh",
        "Need To Visit Before You Die" : "May Enjoy If You Get Around To It",
		"Nothing Could Prepare Me For" : "Does ANYONE Fucking Care About",
		"Of All Time" : "For Now",
		"Of All Time" : "Of The Last 30 Seconds",
		"Of All-Time" : "For Now",
		"OMG" : "*yawn*",
		"OMG" : "No One Cares. At All",
		"One Weird Trick" : "One Piece of Completely Anecdotal Horseshit",
		"Perfection" : "Mediocrity",
		"Priceless" : "Painfully Ordinary",
        "Prove" : "Suggest",
		"Right Now" : "Eventually",
		"Scientific Reasons" : "Vaguely Science-y Reasons",
		"Shocked" : "Vaguely Surprised",
		"Shocking" : "Barely Noticeable",
		"Simple Lessons" : "Inane Pieces of Bullshit Advice",
		"Stop What You're Doing" : "Bookmark Now and Later Completely Forget About",
		"Stop What You’re Doing" : "Bookmark Now and Later Completely Forget About",
		"Stop What You&#8217;re Doing" : "Bookmark Now and Later Completely Forget About",
		"TERRIFYING" : "MODERATELY UNCOMFORTABLE",
		"Terrifying" : "Thoroughly Banal",
		"That Will Make You Rethink" : "That You May Find Vaguely Interesting But Won't Change Your Life in Any Way",
        "The World's Best" : "An Adequate",
		"This Is What Happens" : "This Is Our Bullshit Clickbait Version Of What Happens",
		"Totally blew my mind" : "Bored Me To Tears",
		"Unbelievable" : "Painfully Ordinary",
		"Unimaginable" : "Actually Kind of Droll",
		"WHAT?" : "Some Other Crap",
		"Whoa" : "*yawn*",
		"WHOA" : "Zzzzzzzzzzz",
		"Whoah" : "*yawn*",
		"Will Blow Your Mind" : "Might Perhaps Mildly Entertain You For a Moment",
		"Will Change Your Life Forever" : "Will Not Change Your Life in ANY Meaningful or Lasting Way",
		"Won the Internet" : "Seemed Pretty Cool",
		"Wonderful" : "Mildly Decent",
		"Worst" : "Vaguely Unpleasant",
		"Wow" : "Oh GOD This is SO Boring. Please Kill Me",
		"WOW" : "Zzzzzzzzzzz",
		"You Didn't Know Exist" : "No One Gives a Shit About",
		"You Didn't Know Existed" : "No One Gives a Shit About",
		"You Didn’t Know Exist" : "No One Gives a Shit About",
		"You Didn’t Know Existed" : "No One Gives a Shit About",
		"You Didn&#8217;t Know Exist" : "No One Gives a Shit About",
		"You Didn&#8217;t Know Existed" : "No One Gives a Shit About",
		"You Won't Believe" : "In All Likelihood, You'll Believe",
		"You Won’t Believe" : "In All Likelihood, You'll Believe",	
		"You Won&#8217;t Believe" : "In All Likelihood, You'll Believe",
		"You Wont Believe" : "In All Likelihood, You'll Believe",
		"Have To See To Believe": "Might Have Trouble Picturing"
	}
    clickbait_phrases = [phrase.lower() for phrase in list(clickbait_phrases.keys())]

    with open('../corpus/common_phrases.txt') as f:
        common_phrases = [line.rstrip(' \n') for line in f]
    

    clickbait_phrases += common_phrases

    is_click = []
    for title in df.title.values:
        no_of_phrases = 0
        for phrase in clickbait_phrases:
            if title.find(phrase) != -1:
                no_of_phrases = 1
                break
        is_click.append(no_of_phrases)
    return np.array(is_click).reshape(-1,1)


def click_bait_re(df):
    clickbait_re = ['\b^(Is|Can|Do|Will)(.*)\?\B',
    '\b[Rr]estored [Mm]y [Ff]aith [Ii]n [Hh]umanity\b',
    '\b[Rr]estored [Oo]ur [Ff]aith [Ii]n [Hh]umanity\b',
    'The Best(\s\w+)+\s',
    '\b([Rr]easons\s|[Ww]hy\s|[Hh]ow\s|[Ww]hat\s[Yy]ou\s[Ss]hould\s[Kk]now\s[Aa]bout\s)(.*)\b']
    is_click = []
    for title in df.title.values:
        no_of_phrases = 0
        for re_patten in clickbait_re:
            if re.findall(re_patten, title):
                no_of_phrases = 1
                break
        is_click.append(no_of_phrases)
    return np.array(is_click).reshape(-1,1)

def at_mentions(df):
    is_click = []
    for title in df.title.values:
        is_click.append(title.count('@'))
    return np.array(is_click).reshape(-1,1)

def num_dots(df):
    num_dots = []
    for title in df.title.values:
        num_dots.append(title.count('.'))
    return np.array(num_dots).reshape(-1,1)

def readability_scores_mp(data):
    result_dict, idx, text = data

  #  flesch_reading_ease =  textstat.flesch_reading_ease(text)
    flesch_kincaid_grade =  textstat.flesch_kincaid_grade(text)
    dale_chall_readability_score =  textstat.dale_chall_readability_score(text) 

    result_dict[idx] = [flesch_kincaid_grade, dale_chall_readability_score]

def calc_readability_scores(df):
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    mp_list = [(result_dict, idx, title) for idx, title in enumerate(df.title.values)]

    with multiprocessing.Pool(os.cpu_count()) as p:
        r = list(tqdm_notebook(p.imap(readability_scores_mp, mp_list), total=len(mp_list)))
    rows = [result_dict[idx] for idx in range(df.title.values.shape[0])]
    return pd.DataFrame(rows).values

def text_features(df):
    longest_word_length = []
    mean_word_length = []
    length_in_chars = []

    for title in df.title.values:
        length_in_chars.append(len(title))
        longest_word_length.append(len(max(title.split(), key = len)))
        mean_word_length.append(np.mean([len(word) for word in title.split()]))

    longest_word_length = np.array(longest_word_length).reshape(-1,1)
    mean_word_length = np.array(mean_word_length).reshape(-1,1)
    length_in_chars =  np.array(length_in_chars).reshape(-1,1)
    
    return np.concatenate([longest_word_length, mean_word_length, length_in_chars], axis = 1)

def count_punctuations(df):
    puncts = []
    punctuations = set(string.punctuation)
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    for title in df.title.values:
        puncts.append(count(title,punctuations))
    return np.array(puncts).reshape(-1,1)



def word_ratio(df):
    with open('../corpus/DaleChallEasyWordList.txt') as f:
        easy_words_list = [line.rstrip(' \n') for line in f]
    
    with open('../corpus/terrier-stopword.txt') as f:
        terrier_stopword_list = [line.rstrip(' \n') for line in f]

    terrier_stopword_list += stopwords.words('english')

    with open('../corpus/common.txt') as f:
        common = [line.rstrip(' \n') for line in f]

    terrier_stopword_list += common

    with open('../corpus/contractions.txt') as f:
        contractions_list = [line.rstrip(' \n') for line in f]

    with open('../corpus/hyperbolic.txt') as f:
        hyperbolic_list = [line.rstrip(' \n') for line in f]

    clickbait_subjects = ['dog', 'everyone', 'girl', 'girls', 'guy', 'guys', 'he', 'here', 'i', 'it', 'kid', 'kids', 'man', 'men', 'mom', 'one', 'parent', 'people', 'photos', 'reasons', 'she', 'signs', 'something', 'that', 'they', 'thing', 'things', 'this', 'thoughts', 'times', 'video', 'ways', 'we', 'what', 'who', 'woman', 'women', 'you']

    non_clickbait_subjects = ['bomb', 'court', 'crash', 'earthquake', 'explosion', 'fire', 'government', 'group', 'house', 'u.s.', 'china', 'india', 'iran', 'israel', 'korea', 'leader', 'obama', 'police', 'president', 'senate']

    easy_words_ratio = []
    stop_words_ratio = []
    contractions_ratio = []
    hyperbolic_ratio = []
    clickbait_subs_ratio = []
    non_clickbait_subs_ratio = []
    for title in df.title.values:
        easy_words = 0 
        stop_words = 0
        total_words = 0
        contracted_words = 0
        hyperbolic_words = 0
        clickbait_subs = 0
        nonclickbait_subs = 0

        for word in title.split():
            if word.lower() in easy_words_list:
                easy_words += 1
            if word.lower() in terrier_stopword_list:
                stop_words += 1
            if word.lower() in contractions_list:
                contracted_words += 1
            if word.lower() in hyperbolic_list:
                hyperbolic_words += 1
            if word.lower() in clickbait_subjects:
                clickbait_subs += 1
            if word.lower() in non_clickbait_subjects:
                nonclickbait_subs += 1
            total_words += 1
            
        easy_words_ratio.append(easy_words/total_words)
        stop_words_ratio.append(stop_words/total_words)
        contractions_ratio.append(contracted_words/total_words)
        hyperbolic_ratio.append(hyperbolic_words/total_words)
        clickbait_subs_ratio.append(clickbait_subs/total_words)
        non_clickbait_subs_ratio.append(nonclickbait_subs/total_words)

    easy_words_ratio = np.array(easy_words_ratio).reshape(-1,1)
    stop_words_ratio = np.array(stop_words_ratio).reshape(-1,1)
    contractions_ratio = np.array(contractions_ratio).reshape(-1,1)
    hyperbolic_ratio = np.array(hyperbolic_ratio).reshape(-1,1)
    clickbait_subs_ratio = np.array(clickbait_subs_ratio).reshape(-1,1)
    non_clickbait_subs_ratio = np.array(non_clickbait_subs_ratio).reshape(-1,1)

    return np.concatenate([easy_words_ratio, stop_words_ratio, contractions_ratio, hyperbolic_ratio, clickbait_subs_ratio, non_clickbait_subs_ratio], axis = 1)

def num_hashtags(df):
    return np.array([title.count('#') for title in df.title.values]).reshape(-1,1)


def calc_sentiment_scores(df):
    sid = SentimentIntensityAnalyzer()
    neg = []
    neu = []
    pos = []
    compound = []

    for title in df.title.values:
        sentiments = sid.polarity_scores(title)
        neg.append(sentiments['neg'])
        neu.append(sentiments['neu'])
        pos.append(sentiments['pos'])
        compound.append(sentiments['compound'])

    neg = np.array(neg).reshape(-1,1)
    neu = np.array(neu).reshape(-1,1)
    pos = np.array(pos).reshape(-1,1)
    compound = np.array(compound).reshape(-1,1)
    return np.concatenate([neg,  pos, compound], axis = 1)

def get_glove_vectors(df, glove):
    vectors = []
    for title in tqdm_notebook(df.title.values):
        vectors.append(np.average(glove.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)


def tfidf_w2v(df, idf_dict, glove):
    vectors = []
    for title in tqdm_notebook(df.title.values):
        w2v_vectors = glove.query(word_tokenize(title))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(title)]
        vectors.append(np.average(w2v_vectors, axis = 0, weights = weights))
    return np.array(vectors)


def featurize(train_df, test_df, embedding_type):

    print('Starts with number....')
    
    
    train_starts_with_number = starts_with_number(train_df)
    test_starts_with_number = starts_with_number(test_df)
    
    print('Clickbait Phrases....')
    train_cb_phrases = click_bait_phrases(train_df)
    test_cb_phrases = click_bait_phrases(test_df)


    print('Clickbait re....')
    train_cb_re = click_bait_re(train_df)
    test_cb_re = click_bait_re(test_df)


    print('Num dots....')
    train_num_dots = num_dots(train_df)
    test_num_dots = num_dots(test_df)


    print('Text Features....')
    train_text_features = text_features(train_df)
    test_text_features = text_features(test_df)


    print('Punctuation....')
    train_num_punctuations = count_punctuations(train_df)
    test_num_punctuations = count_punctuations(test_df)


    print('Word ratios....')
    train_word_ratio = word_ratio(train_df)
    test_word_ratio = word_ratio(test_df)


    print('Sentiment Scores....')
    train_sentiment = calc_sentiment_scores(train_df)
    test_sentiment = calc_sentiment_scores(test_df)

    print('Readability Scores....')

    train_readability_scores = calc_readability_scores(train_df)
    test_readability_scores = calc_readability_scores(test_df)

    if embedding_type == 'tfidf':
        print('TFIDF Title....')

        tfidf_word = TfidfVectorizer()

        print('TFIDF Word....')
        train_word_features = tfidf_word.fit_transform(train_df.title.values)
        test_word_features = tfidf_word.transform(test_df.title.values)


        normalizer_tfidf = MinMaxScaler()
        train_embedding_features = sparse.csr_matrix(normalizer_tfidf.fit_transform(train_word_features.todense()))


        test_embedding_features = sparse.csr_matrix(normalizer_tfidf.fit_transform(test_word_features.todense()))
    
    elif embedding_type == 'glove':
        print('Glove.....')
        glove = Magnitude("../vectors/glove.6B.100d.magnitude")
        train_glove = get_glove_vectors(train_df, glove)
        test_glove = get_glove_vectors(test_df, glove)

        normalizer_glove = MinMaxScaler()
        train_glove = normalizer_glove.fit_transform(train_glove)
        test_glove = normalizer_glove.transform(test_glove)


        train_embedding_features = sparse.csr_matrix(train_glove)
        test_embedding_features = sparse.csr_matrix(test_glove)
        
    elif embedding_type == 'tfidf_glove':
        print('Glove.....')
        
        glove = Magnitude("../vectors/glove.6B.100d.magnitude")
        
        tfidf = TfidfVectorizer()
        tfidf.fit(train_df.title.values)
        idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

        train_glove = tfidf_w2v(train_df, idf_dict, glove)
        test_glove = tfidf_w2v(test_df, idf_dict, glove)

        normalizer_glove = MinMaxScaler()
        train_glove = normalizer_glove.fit_transform(train_glove)
        test_glove = normalizer_glove.transform(test_glove)


        train_embedding_features = sparse.csr_matrix(train_glove)
        test_embedding_features = sparse.csr_matrix(test_glove)
   

    train_features = hstack((train_starts_with_number,
                             train_cb_phrases,
                             train_cb_re,
                             train_num_dots,
                             train_text_features,
                             train_word_ratio,
                             train_sentiment,
                             train_readability_scores,
                             train_num_punctuations))

    normalizer = MinMaxScaler()
    train_features = normalizer.fit_transform(train_features)

    train_features = sparse.csr_matrix(train_features)

    train_features = sparse.hstack((
                            train_features,
                            train_embedding_features
                            ))




    test_features = hstack((test_starts_with_number,
                             test_cb_phrases,
                             test_cb_re,
                             test_num_dots,
                             test_text_features,
                             test_word_ratio,
                             test_sentiment,
                             test_readability_scores,
                             test_num_punctuations))
    test_features = normalizer.transform(test_features)

    test_features = sparse.csr_matrix(test_features)
    test_features = sparse.hstack((
                             test_features,
                             test_embedding_features
                            ))



    



    feature_names = ['starts_with_number',
                     'clickbait_phrases',
                     'clickbait_re',
                     'num_dots',
                     'longest_word_length',
                     'mean_word_length',
                     'length_in_chars',
                     'easy_words_ratio',
                     'stop_words_ratio',
                     'contractions_ratio',
                     'hyperbolic_ratio',
                     'clickbait_subs_ratio', 
                     'nonclickbait_subs_ratio', 
                     'sentiment_neg',
                     'senitment_pos',
                     'sentiment_compound',
                     'flesch_kincaid_grade',
                     'dale_chall_readability_score',
                     'num_punctuations'
                     ]

    if embedding_type == 'tfidf':
        feature_names = feature_names + ['tfidf_word_' + col for col in tfidf_word.get_feature_names()] 
    else:

        feature_names = feature_names + ['glove_' + str(col) for col in range(100)] 
    print('DONE!')
    
    return train_features, test_features, feature_names
