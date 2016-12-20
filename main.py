import commentCollectorPandas
from tabulate import tabulate
import pandas as pd

allPostsSequences, allPostsAnnotated = commentCollectorPandas.getPostSeqeuences()
ratingColumnsdf = pd.DataFrame((allPostsAnnotated[:,5:8]).tolist(), columns=['neg', 'ntl', 'pos'])


# Drop the PostNo and SentenceNo columns
allPostSequencesStripped = allPostsSequences.ix[:,2:]
allPostsSequences = pd.concat([allPostSequencesStripped, ratingColumnsdf], axis=1, join='inner')

# With the ratings appended to the end
print(tabulate(allPostsSequences, headers='keys', tablefmt='psql'))

# This comes with a frequency count column called 'size' for each rule. The rules are aggregated
# somePosts = allPostSequencesStripped.reset_index().groupby(allPostSequencesStripped.columns.difference(['index']).tolist())['index'].agg(['first', 'size']).reset_index().set_index(['first']).sort_index().rename_axis(None)

# Here we aggregate by row, and then do a sum for the polarity columns (neg, ntl, pos)

polarityAggPostsDF = allPostsSequences.reset_index().groupby(allPostSequencesStripped.columns.difference(['index', 0, 1, 'neg','ntl','pos']).tolist())['neg','ntl','pos'].agg(lambda x: x.sum()).reset_index()
print(tabulate(polarityAggPostsDF, headers='keys', tablefmt='psql'))

print('Finished')
