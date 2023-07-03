This README provides instructions on how to test your classifier.py implementation. There are multiple unittests per function. You should pass all the tests for full grade. 

# Instructions to use:
Add your ‘classifier.py’ function to the folder that was shared with you.
Execute the program ‘run_tests_students.py’ to evaluate your code.
You can use ‘python run_tests_students.py’ in the terminal from inside the folder.
Please DO NOT modify any unittest functions for example: score_function_unittest

# Requirements:
Python, unittest, scipy, numpy.

# Updated Directory Structure
The folder that was shared with you has the following file structure.

├── README.md
├── classifier.py
├── run_tests_students.py
├── tests_students
│   ├── test_functions_students.py
│   └── tests_data
│       ├── dummy_data.json
│       ├── gd_sgd_params_initialization.npy
│       ├── gd_test1_wts.npy
│       ├── gd_test2_wts.npy
│       ├── sent1_features.npz
│       ├── sent2_features.npz
│       ├── sent3_features.npz
│       ├── sent4_features.npz
│       ├── sgd_test1_wts.npy
│       └── sgd_test2_wts.npy
└── vocab.txt

# How to read the output
Once you run the tests, if any of the test cases fail, on terminal you can see what error type was thrown, what was the input test to your function and what is expected outcome of that function (that is being tested). If all test cases pass then the returned message prints “no error”. 

# New changes log
We have now moved most of the dummy data into a separate file (previously it was inside test functions). The dimensions of dummy data should now match with Project1 instructions. Few minor bug fixed but there is no major change in unitests structure.
