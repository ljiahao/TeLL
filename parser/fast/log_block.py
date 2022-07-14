import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from util.setting import logger

def camel_case_split(identifier: str) -> list:
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

class LoggedBlock():
    """Structure of Logged Block.
    """
    def __init__(self, block_id: str, level: str, content: str, start_idx: int, end_idx: int, dataset_name: str) -> None:
        self.block_id = block_id
        self.level = level
        self.content = content
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.dataset_name = dataset_name
        self.log_message = ''
        self.extract_message()

    def print_logged_block(self) -> None:
        logger.info(f'Block ID: {self.block_id} \nLog Level: {self.level} \nContent: {self.content}\n')

    def extract_message(self) -> None:
        """Extract log message from content.
        """
        start_idx = self.content.find(self.level) + len(self.level)
        str_candidate = self.content[start_idx:]
        is_message = False

        left_bracket = 0
        right_bracket = 0
        quat_num = 0
        for idx in range(len(str_candidate)):
            if str_candidate[idx] == '(':
                left_bracket += 1
            elif str_candidate[idx] == ')':
                right_bracket += 1
            elif str_candidate[idx] == '"':
                quat_num += 1
                if quat_num % 2 != 0:
                    is_message = True
                else:
                    is_message = False
            if is_message and str_candidate[idx] != '"':
                self.log_message += str_candidate[idx]
            if left_bracket == right_bracket and left_bracket != 0:
                break

    def extract_message_tokens(self) -> list:
        """NLP preprocessing for log message (lower, removing punctuation, tokenization, stopword filtering, stemming)
        """
        log_message_p = ''.join([char for char in self.log_message if char not in string.punctuation])
        words = list()
        tmp = word_tokenize(log_message_p)
        for word in tmp:
            words += camel_case_split(word)
        words = [word.lower() for word in words]
        stop_words = stopwords.words('english')
        filtered_words = [word for word in words if word not in stop_words]
        filtered_words = [word.replace('0x', '') for word in filtered_words]
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in filtered_words]

        return stemmed