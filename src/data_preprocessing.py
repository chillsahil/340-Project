import re
import string
import emoji
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

slang_dict = {
    "u": "you", "ur": "your", "r": "are", "lol": "laughing out loud", "lmao": "laughing my ass off",
    "idk": "i do not know", "imo": "in my opinion", "tbh": "to be honest", "omg": "oh my god",
    "smh": "shaking my head", "fml": "fuck my life", "ily": "i love you", "nvm": "never mind",
    "ty": "thank you", "np": "no problem", "pls": "please", "plz": "please", "bc": "because",
    "w/": "with", "w/o": "without", "gr8": "great", "ya": "you", "tho": "though", "cuz": "because",
    "wat": "what", "bruh": "bro", "sis": "sister", "fam": "family", "hbu": "how about you",
    "wyd": "what are you doing", "rn": "right now", "irl": "in real life", "asap": "as soon as possible"
}

def normalize_slang(text, mapping):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in mapping) + r')\b')
    return pattern.sub(lambda m: mapping[m.group()], text)

def aggressive_clean_text(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = normalize_slang(text, slang_dict)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_dataframe(df):
    df['clean_text'] = df['text'].apply(aggressive_clean_text)
    df['labels'] = df['subreddit'].apply(lambda x: [lab.strip() for lab in x.split(',')])
    return df

def get_mlb_labels(df, disorder_list):
    mlb = MultiLabelBinarizer(classes=disorder_list)
    y = mlb.fit_transform(df['labels'])
    return mlb, y
