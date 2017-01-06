'''This bot searches through the descriptions from the tech meetup data
for gender-specific pronouns and honorifics. It then codes the speaker as
being female or not female, male or not male and looks for conflicts.
'''

import pandas as pd
import numpy as np
import re
from nltk.tag import StanfordNERTagger
import gender_guesser.detector as gender

os.environ["STANFORD_MODELS"] = "/Users/-----------/Documents/JavaJuno/stanford-ner-2015-04-20"

# st = StanfordNERTagger('stanford_ner\english.all.3class.distsim.crf.ser.gz')


meetup = pd.read_csv('techMeetups.csv')

def find_female(x):
    if re.search(r'(\bshe\b | \bher\b | \bMs.\b | \bMrs.\b | \bMiss\b)', x,
                 flags=re.I) is not None:
        return 1
    else:
        return 0

def find_male(x):
    if re.search(r'\bhe\b | \bhim\b | \bhis\b | \bMr.\b', x,
                 flags=re.I) is not None:
        return 1
    else:
        return 0

meetup['female'] = meetup['desc'].apply(lambda x: find_female(x))
meetup['male'] = meetup['desc'].apply(lambda x: find_male(x))
meetup['applicable'] = np.where(meetup['male'] + meetup['female'] > 0, 1, 0)
meetup['multi'] = np.where(meetup['male'] + meetup[
    'female'] > 1, 1, 0)

meetup.to_csv('techMeetupsNotated.csv', encoding='utf-8', index=False)

print('Done!')
print('Number of rows is:', meetup.shape[0])
print('Number that have gender words:', meetup['applicable'].sum())
print('Number that have female words:', meetup['female'].sum())
