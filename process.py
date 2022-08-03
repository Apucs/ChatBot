import random
import trax
from config import *
from utils import Util
from tqdm import tqdm

class Process:
    def __init__(self) -> None:
        pass


    def stream(self, data):
        # loop over the entire data
        while True:
            # get a random element
            d = random.choice(data)
            
            # yield a tuple pair of identical values 
            # (i.e. our inputs to the model will also be our targets during training)
            yield (d, d)


    def train_test(self):

        #db = Util()
        DIALOGUE_DB = Util.load_json(DATA_DIR, DATA_FILE)
        print(f'The number of dialogues is: {len(DIALOGUE_DB)}')

        all_files = DIALOGUE_DB.keys()

        # initialize empty list
        untokenized_data = []

        # loop over all files
        for file in tqdm(all_files):
            # this is the graded function you coded
            # returns a string delimited by Person 1 and Person 2
            result = Util.get_conversation(file, DIALOGUE_DB)
            
            # append to the list
            untokenized_data.append(result)

        # print the first element to check if it's the same as the one we got before
        # print("\n", untokenized_data[0])
        # print(len(untokenized_data))

        # shuffle the list we generated above
        random.shuffle(untokenized_data)

        # a cutoff (5% of the total length for this assignment)
        # convert to int because we will use it as a list index
        cut_off = int(len(untokenized_data) * .05)

        # slice the list. the last elements after the cut_off value will be the eval set. the rest is for training. 
        train_data, eval_data = untokenized_data[:-cut_off], untokenized_data[-cut_off:]

        # print(f'number of conversations in the data set: {len(untokenized_data)}')
        # print(f'number of conversations in train set: {len(train_data)}')
        # print(f'number of conversations in eval set: {len(eval_data)}')

        

        # trax allows us to use combinators to generate our data pipeline
        data_pipeline = trax.data.Serial(
            # randomize the stream
            trax.data.Shuffle(),
            
            # tokenize the data
            trax.data.Tokenize(vocab_dir=VOCAB_DIR,
                            vocab_file=VOCAB_FILE),
            
            # filter too long sequences
            trax.data.FilterByLength(2048),
            
            # bucket by length
            trax.data.BucketByLength(boundaries=[128, 256,  512, 1024],
                                    batch_sizes=[16,    8,    4,   2, 1]),
            
            # add loss weights but do not add it to the padding tokens (i.e. 0)
            trax.data.AddLossWeights(id_to_mask=0)
        )

        # apply the data pipeline to our train and eval sets
        train_stream = data_pipeline(self.stream(data = train_data))
        eval_stream = data_pipeline(self.stream(data = eval_data))

        return train_data, eval_data, train_stream, eval_stream


# p = Process()
#
# train_data, eval_data, train_stream, eval_stream = p.train_test()
#
# # the stream generators will yield (input, target, weights). let's just grab the input for inspection
# inp, _, _ = next(train_stream)
#
# # print the shape. format is (batch size, token length)
# print("input shape: ", inp.shape)
#
# # detokenize the first element
# print(trax.data.detokenize(inp[0], vocab_dir=VOCAB_DIR, vocab_file=VOCAB_FILE))