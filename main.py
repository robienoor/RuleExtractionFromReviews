import commentCollectorPandas
from tabulate import tabulate

allPostSequences = commentCollectorPandas.getPostSeqeuences()

allPostSequencesStripped = allPostSequences.ix[:,2:]
print(tabulate(allPostSequencesStripped, headers='keys', tablefmt='psql'))

allPostSequencesStripped = allPostSequencesStripped.drop_duplicates()
print(tabulate(allPostSequencesStripped, headers='keys', tablefmt='psql'))

print('here')
