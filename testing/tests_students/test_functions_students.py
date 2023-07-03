# 3, 3x3 = 
import unittest
import numpy as np
from collections import Counter
from classifier import feature_extractor, classifier_agent
from classifier import custom_feature_extractor, classifier_agent, tokenize, load_data, data_processor
import scipy
from scipy import sparse
import json 

def load_from_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


class TestFunctions(unittest.TestCase):
    def setUp(self):
        ## For evaluating each function module
        ## please specify correct path to files being read.
        with open('./vocab.txt') as file:
            reading = file.readlines()
            vocab_list = [item.strip() for item in reading]
            vocab_dict = {item: i for i, item in enumerate(vocab_list)}
        fe = feature_extractor(vocab_list, tokenize)
        self.fe = fe
        
        self.vocab_list = vocab_list
        self.ca = classifier_agent(fe, np.array([0.53, -0.89, 1.663])) # params shape is (3,) for format of (d,)

        self.error_margin = 1e-3
        self.err = 0.999
        
        self.testcases = load_from_json("./tests_students/tests_data/dummy_data.json")


    def bow_features_unittest(self,student_features,true_features):
        template = "An exception of type {0} occurred.:\n{1!r}"
        message = "No error"

        try:
            self.assertTrue(isinstance(student_features,scipy.sparse.csc_array),'Return type incorrect!')
        except Exception as error:
            message = template.format(type(error).__name__,error.args)
            return message 
        try:
            self.assertTrue(student_features.shape == true_features.shape,'Output dimension incorrect!')
        except Exception as error:
            message = template.format(type(error).__name__,error.args)
            return message 
        
        temp_check = (student_features.data == true_features.data)
        if type(temp_check) == bool:
            try:
                self.assertTrue(temp_check == True ,'Feature Values do not match!')
            except Exception as error:
                message = template.format(type(error).__name__,error.args)
                return message 
        else:
            try:
                self.assertTrue((temp_check).all() == True ,'Feature Values do not match!')
            except Exception as error:
                message = template.format(type(error).__name__,error.args)
                return message 
        
        return message
     
    def loss_function_unittest(self, x, y, out):
        template = "An exception of type {0} occurred.:\n{1!r}"
        out_student = self.ca.loss_function(x, y)
        message = "No error"
        try:
            distance = abs(out_student - out)
            self.assertTrue(distance<=self.error_margin, 'Output value incorrect! Test input: {}, expected output: {}'.format(np.array2string(x, precision=3, separator=','), out))
        except Exception as error:
            message = template.format(type(error).__name__, error.args)
        return message
    
    def score_function_unittest(self, x1, y1, FLAG=False):
        template = "An exception of type {0} occurred.:\n{1!r}"
        out_student = self.ca.score_function(x1)
        message = "No error"

        if FLAG == True:
            if np.array_equal(out_student, np.array([0, 0, 0, 0])):
                self.ca.params = np.array([0.53, -0.89, 1.663])
                return message
            else:
                return "There is an error, input passed: {}, expected output: {}".format(x1, np.array([0, 0, 0, 0]))

        try:
            self.assertTrue(out_student.shape==y1.shape, 'Output dimension incorrect!')
        except Exception as error:
            message = template.format(type(error).__name__, error.args)
            return message 
        try:
            self.assertTrue(isinstance(out_student, np.ndarray), 'Return type incorrect!')
        except Exception as error:
            message = template.format(type(error).__name__, error.args)
            return message 
            
        try:
            distance = np.linalg.norm(out_student - y1)
            self.assertTrue(distance<=1, 'Output value incorrect! Test input: {}'.format(np.array2string(x1, precision=3, separator=',')))
        except Exception as error: 
            message = template.format(type(error).__name__, error.args)
            return message 
    
        return message
    
    def predict_function_unittest(self, x, out, FLAG=False):
        template = "An exception of type {0} occurred.:\n{1!r}"
        out_student = self.ca.predict(x, RAW_TEXT=FLAG)
        message = "No error"

        if FLAG == False:
            try:
                self.assertTrue(out_student.shape == out.shape,'Output dimension incorrect! Test input: {}, expected output: {}'.format(np.array2string(x, precision=3,separator=','), out))
            except Exception as error:
                message = template.format(type(error).__name__, error.args)
                return message
            try:
                self.assertTrue((out_student==out).all(), 'Output value incorrect! Test input: {}, expected output: {}'.format(np.array2string(x, precision=3,separator=','), out))
            except Exception as error:
                message = template.format(type(error).__name__, error.args)
                return message
            
        if FLAG == True:
            try:
                self.assertTrue(out_student.shape == out.shape,'Output dimension incorrect! Test input: {}, expected output: {}'.format(x, out))
            except Exception as error:
                message = template.format(type(error).__name__, error.args)
                return message
            try:
                self.assertTrue((out_student==out).all(), 'Output value incorrect! Test input: {}, expected output: {}'.format(x, out))
            except Exception as error:
                message = template.format(type(error).__name__, error.args)
                return message

        return message
    
    def error_function_unittest(self, x, y, out, FLAG=False):
        template = "An exception of type {0} occurred.:\n{1!r}"
        out_student = self.ca.error(x, y, RAW_TEXT=FLAG)
        message = "No error"
        try:
            distance = abs(out_student - out)
            if type(x) != list:
                self.assertTrue(distance<=self.error_margin, 'Output value incorrect! Test input: {}, expected output: {}'.format(np.array2string(x, precision=3), out))
            else:
                self.assertTrue(distance<=self.error_margin, 'Output value incorrect! Test input: {}, expected output: {}'.format(x, out))

        except Exception as error:
            message = template.format(type(error).__name__, error.args)
            return message
        
        return message
    
    def gradient_function_unittest(self, x, y, out):
        template = "An exception of type {0} occurred.:\n{1!r}"
        out_student = self.ca.gradient(x, y)
        message = "No error"
        try:
            self.assertTrue(out_student.shape==out.shape, 'Output dimension incorrect!')
        except Exception as error:
            message = template.format(type(error).__name__, error.args)
            return message
        
        try:
            self.assertTrue(isinstance(out_student, np.ndarray), 'Return type incorrect!')
        except Exception as error:
            message = template.format(type(error).__name__, error.args)
            return message
        try: # new 
            self.assertTrue(out_student.shape==self.ca.params.shape, 'Shape of output not same as shape of params!')
        except Exception as error:
            message = template.format(type(error).__name__, error.args)
            return message
        try:
            distance = np.linalg.norm(out_student - out)
            self.assertTrue(distance<=self.error_margin, 'Output value incorrect! Test input: {}, expected output: {}'.format(np.array2string(x, precision=3), out))
        except Exception as error:
            message = template.format(type(error).__name__, error.args)
            return message
        return message
    

    def train_gd_function_unittest(self, x, y, out):
        template = "An exception of type {0} occurred.:\n{1!r}"
        message = "No error"
        if self.ca.params.sum() != 0:
            self.ca.params = np.load("./tests_students/tests_data/gd_sgd_params_initialization.npy") # need to re-initialize with vocab size. Cannot change the size.
        
        self.ca.train_gd(x, y, 1)
        try:
            distance = np.linalg.norm(self.ca.params - out)
            self.assertTrue(distance<=self.error_margin, 'Output value incorrect! Test input: {}, expected output: {}'.format(x, out))
        except Exception as error: 
            message = template.format(type(error).__name__, error.args)
            return message 
        self.ca.params = np.load("./tests_students/tests_data/gd_sgd_params_initialization.npy")
        return message 
    

    def train_sgd_function_unittest(self, x, y, out):
        template = "An exception of type {0} occurred.:\n{1!r}"
        message = "No error"
        np.random.seed(5201314)
        if self.ca.params.sum() != 0:
            self.ca.params = np.load("./tests_students/tests_data/gd_sgd_params_initialization.npy") # need to initialize with vocab size. Cannot change the size.
        
        self.ca.train_sgd(x, y, 1)
        try:
            distance = np.linalg.norm(self.ca.params - out)
            self.assertTrue(distance<=self.error_margin, 'Output value incorrect! Test input: {}, expected output: {}'.format(x, out))
        except Exception as error: 
            message = template.format(type(error).__name__, error.args)
            return message 
        self.ca.params = np.load("./tests_students/tests_data/gd_sgd_params_initialization.npy")
        return message 
        
    ############ main test functions ############### 
    def test_bow_feature(self):
        ''' Creating sentences for test cases'''
        message = "No error"
        sent1 = 'The sun set behind the mountains, casting a golder glow on the horizon.'
        sent2 = 'She opened the book and began reading the first chapter'
        sent3 = 'She began reading the book and opened the first chapter'
        sent4 = 'I really like kiwano fruit'

        true1 = sparse.load_npz('./tests_students/tests_data/sent1_features.npz')
        true2 = sparse.load_npz('./tests_students/tests_data/sent2_features.npz')
        true3 = sparse.load_npz('./tests_students/tests_data/sent3_features.npz')
        true4 = sparse.load_npz('./tests_students/tests_data/sent4_features.npz')
        # test 1
        output1 = self.fe.bag_of_word_feature(sent1)
        message = self.bow_features_unittest(output1,true1)
        print("Running test 1 for bag_of_word_feature check {}".format(message))
        # test 2
        output2 = self.fe.bag_of_word_feature(sent2)
        message = self.bow_features_unittest(output2,true2)
        print("Running test 2 for bag_of_word_feature check {}".format(message))
        # test 3
        output3 = self.fe.bag_of_word_feature(sent3)
        message = self.bow_features_unittest(output3,true3)
        print("Running test 3 for bag_of_word_feature check: {}".format(message))
        # test 4
        output4 = self.fe.bag_of_word_feature(sent4)
        message = self.bow_features_unittest(output4,true4)
        print("Running test 4 for bag_of_word_feature check, result: {}".format(message), "\n")
    
    def test_score_function(self):
        """ tests score function in classifier_agent against dummy data """
        # tests mode 1
        x1 = np.array(self.testcases['score']['x1']) # input shape now is (3x4) => d=3, m=4
        out1 = np.array(self.testcases['score']['out1']) # expected output

        message = self.score_function_unittest(x1, out1, FLAG=False) 
        print("Running test 1 for score function, result: {}".format(message))
        
        # tests mode 2 (only to check dimension assigment in score function)
        x2 = np.array(self.testcases['score']['x2']) # shape (2,4): d=2, m=4
        message = self.score_function_unittest(x2, None, FLAG=True)
        print("Running test 2 for score function, result: {}".format(message), "\n")
        
    def test_error(self):
        """ tests error function in classifier_agent against dummy data. 
        Input shape: 
                    X: d by m sparse (csc_array) matrix
                    param y: m dimensional vector (numpy.array) of true labels
                    param RAW_TEXT: if True, then X is a list of text string,
                                    and y is a list of true labels
                    return: The average error rate
        """
        # mode 1 check
        x1 = scipy.sparse.csr_matrix(np.array(self.testcases['error']['x1']))
        y1 = np.array(self.testcases['error']['y1'])
        out1 = self.testcases['error']['out1']
        
        message = self.error_function_unittest(x1, y1, out1, FLAG=False)
        print("Running test 1 for error function: {}".format(message))

        # mode 2 check
        x2 = self.testcases['error']['x2']
        y2 = np.array(self.testcases['error']['y2']) # predicted labels: [False, False]
        out2 = self.testcases['error']['out2']

        message = self.error_function_unittest(x2, y2, out2, FLAG=True)
        print("Running test 2 for error function, result: {}".format(message), "\n")
        
    def test_predict(self):
        """ X: dxm
            returns a score
        """
        # Case 1
        x1 = np.array(self.testcases['predict']['x1']) #3x4
        out1 = np.array(self.testcases['predict']['out1'])

        message = self.predict_function_unittest(x1, out1, FLAG=False)
        print("Running test 1 for predict function check, result: {}".format(message))

        # Case 2
        # x2 = 'The remainder of the documentation explores the full feature set from first principles.'
        # out2 = np.array([False], dtype = bool)
        x2 = self.testcases['predict']['x2'][0]
        out2 = np.array(self.testcases['predict']['out2'])

        message = self.predict_function_unittest(x2, out2, FLAG=True)
        print("Running test 2 for predict function check, result: {}".format(message), "\n")

    def test_loss_function(self):
        # test 1
        x1 = scipy.sparse.csr_matrix(np.array(self.testcases['loss_function']['x1'])) # shape (3,4)
        y1 = np.array(self.testcases['loss_function']['y1'])
        out1 = self.testcases['loss_function']['out1']

        message = self.loss_function_unittest(x1, y1, out1)
        print("Running test 1 for loss_function , result: {}".format(message))
        
        # test 2: behaviour with large values
        x2 = scipy.sparse.csr_matrix(np.array(self.testcases['loss_function']['x2'])) # shape (3,4)
        y2 = np.array(self.testcases['loss_function']['y2']) # shape (4,)
        out2 = self.testcases['loss_function']['out2']

        message = self.loss_function_unittest(x2, y2, out2)
        print("Running test 2 for loss_function , result: {}".format(message), "\n")

    def test_gradient(self):
        """Return an nd.array of size the same as self.params"""
        # test 1: normal value check
        x1 = scipy.sparse.csr_matrix(np.array(self.testcases['gradient']['x1'])) # shape: d=3, m=4
        y1 = np.array(self.testcases['gradient']['y1']) # shape: (m,)
        out1 = np.array(self.testcases['gradient']['out1'])
        
        message = self.gradient_function_unittest(x1, y1, out1)
        print("Running test 1 for gradient function , result: {}".format(message))

        # test 2: checking large values
        x2 = scipy.sparse.csr_matrix(np.array(self.testcases['gradient']['x2'])) # shape: d=3, m=4
        y2 = np.array(self.testcases['gradient']['y2'])  # shape: (m,): m=4
        out2 = np.array(self.testcases['gradient']['out2'])

        message = self.gradient_function_unittest(x2, y2, out2)
        print("Running test 2 for gradient function , result: {}".format(message), "\n")

    def test_train_gd(self):
        # test 1
        x1 = ["I don't like peope", "The remainder of the documentation explores the full feature set from first principles."]
        y1 = [0, 1]
        out1 = np.load("./tests_students/tests_data/gd_test1_wts.npy")
        
        message = self.train_gd_function_unittest(x1, y1, out1)
        print("Running test 1 for train_gd function , result: {}".format(message))

        # test 2
        x2 = ["", "The remainder of the documentation explores the full feature set from first principles.", "What are you doing so late at night?"]
        y2 = [0, 1, 1]
        out2 = np.load("./tests_students/tests_data/gd_test2_wts.npy")

        message = self.train_gd_function_unittest(x2, y2, out2)
        print("Running test 2 for train_gd function , result: {}".format(message), "\n")

    def test_train_sgd(self):
        # test 1
        x1 = ["I don't like peope", "The remainder of the documentation explores the full feature set from first principles.", "What are you doing so late at night?"]
        y1 = [0, 1, 1]
        out1 = np.load("./tests_students/tests_data/sgd_test1_wts.npy")

        message = self.train_sgd_function_unittest(x1, y1, out1)
        print("Running test 1 for train_sgd function , result: {}".format(message), "\n")

        # test 2
        x2 = ["I don't like avacado"]
        y2 = [0]
        out2 = np.load("./tests_students/tests_data/sgd_test2_wts.npy")

        message = self.train_sgd_function_unittest(x2, y2, out2)
        print("Running test 2 for train_sgd function , result: {}".format(message), "\n")


if __name__ == '__main__':
    a = TestFunctions()
    a.setUp()