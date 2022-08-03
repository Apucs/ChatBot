import json
from termcolor import colored
from config import *


class Util:
    def __init__(self):
        pass

    # load a JSON file

    def load_json(directory, file):

        # self.directory = directory
        # self.file = file

        with open(f'{directory}/{file}') as file:
            db = json.load(file)
        return db

    def get_conversation(file, data_db):
        '''
        Args:
            file (string): filename of the dialogue file saved as json
            data_db (dict): dialogue database
        
        Returns:
            string: A string containing the 'text' fields of  data[file]['log'][x]
        '''

        # initialize empty string
        result = ''

        # get length of file's log list
        len_msg_log = len(data_db[file]['log'])

        # set the delimiter strings
        delimiter_1 = ' Person 1: '
        delimiter_2 = ' Person 2: '

        # loop over the file's log list
        for i in range(len_msg_log):

            # get i'th element of file log list
            cur_log = i

            # check if i is even
            if i % 2 == 0:
                # append the 1st delimiter string
                result += delimiter_1
            else:
                # append the 2nd delimiter string
                result += delimiter_2

                # append the message text from the log
            result += data_db[file]['log'][i]['text']

        return result

    def print_conversation(conversation):

        delimiter_1 = 'Person 1: '
        delimiter_2 = 'Person 2: '

        split_list_d1 = conversation.split(delimiter_1)
        #print(split_list_d1)

        for sublist in split_list_d1[1:]:
            split_list_d2 = sublist.split(delimiter_2)
            print(colored(f'Person 1: {split_list_d2[0]}', 'red'))

            if len(split_list_d2) > 1:
                print(colored(f'Person 2: {split_list_d2[1]}', 'green'))

        #print("Done........")

# load the dialogue data set into our dictionary
# print(DATA_DIR, DATA_FILE)
# p = Util()
# DIALOGUE_DB = p.load_json(DATA_DIR, DATA_FILE)
# print(f'The number of dialogues is: {len(DIALOGUE_DB)}')
# # print 7 keys from the dataset to see the filenames
# print(list(DIALOGUE_DB.keys())[0:7]) 

# file = 'SNG01856.json'
# conversation = p.get_conversation(file, DIALOGUE_DB)
# p.print_conversation(conversation)
