__author__ = 'kaiolae'
__author__ = 'kaiolae'
import NeuralNetwork as nn 

#Class for holding your data - one object for each line in the dataset
class dataInstance:

    def __init__(self,qid,rating,features):
        self.qid = qid #ID of the query
        self.rating = rating #Rating of this site for this query
        self.features = features #The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: "+ str(self.qid)+ ". rating: "+ str(self.rating)+ ". features: "+ str(self.features)


#A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:

    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self,file):
        #Input: A file with the data.
        #Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open(file)
        dataset = {}
        for line in data:
            #Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem: #We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            #Creating a new data instance, inserting in the dict.
            di = dataInstance(qid,rating,features)
            if qid in dataset.keys():
                dataset[qid].append(di)
            else:
                dataset[qid]=[di]
        return dataset

def createPairs(data):
    pairs = []
    for qid in data.dataset:
        q = data.dataset[qid] 
        q = sorted(q, key=lambda x: x.rating, reverse=True)
	for i in xrange(len(q)-1):
            for j in xrange(i+1,len(q)):
                if q[i].rating == q[j].rating: continue
                pairs.append((q[i], q[j]))
    return pairs
          
def runRanker(trainingset, testset):
    #TODO: Insert the code for training and testing your ranker here.
    #Dataholders for training and testset
    dhTraining = dataHolder(trainingset)
    dhTesting = dataHolder(testset)

    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    network = nn.NeuralNetwork([46,10,1],0.001)

    trainingPatterns = createPairs(dhTraining)
    testPatterns = createPairs(dhTesting)
#    for tup in testPatterns:
#        print(tup[0].features)
#        print(tup[1])
#        print('this was one tuple')


#    #Check ANN performance before training
    network.countMisorderedPairs(testPatterns)
    network.train(trainingPatterns)
#    for i in range(25):
#        #Running 25 iterations, measuring testing performance after each round of training.
#        #Training
#        nn.train(trainingPatterns,iterations=1)
#        #Check ANN performance after training.
#        nn.countMisorderedPairs(testPatterns)
#
#    #TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.
#


runRanker("train.txt","test.txt")
