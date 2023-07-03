import json
import unittest
import numpy as np

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests_students')
    unittest.TextTestRunner(verbosity=1).run(suite) # change verbosity to your use-case. Max = 2

        
