import unittest
import spacy
from src.main.model_utilities import *

class UnitTests(unittest.TestCase):

    def test_get_model_evaluation_metrics(self):
        '''
        Unit test for function computes manually different metrics
        '''
        confusion_matrix = np.array([[10, 5], [2, 20]])
        expected_metrics = {'accuracy': 0.8108108108108109,
        'f1_score': 0.7959,
        'precision': 0.81667,
        'recall': 0.78788,
        'specificity': 0.78788}
        actual_metrics = get_model_evaluation_metrics(confusion_matrix)
        self.assertEqual(actual_metrics, expected_metrics)

        confusion_matrix = np.zeros((2, 2))
        expected_metrics = {}
        actual_metrics = get_model_evaluation_metrics(confusion_matrix)
        self.assertEqual(actual_metrics, expected_metrics)

    def test_split_model_data(self):
        '''
        Unit test for function that splits the data into training and testing sets
        '''
        X_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_data = np.array([0, 1, 0, 1])
        X_train, X_test, y_train, y_test = split_model_data(X_data, y_data,  test_size_value = 2)

        self.assertEqual(len(X_train) + len(X_test), len(X_data))
        self.assertEqual(len(y_train) + len(y_test), len(y_data))
        self.assertEqual(len(np.unique(y_train)), len(np.unique(y_data)))


        X_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_data = np.array([0, 1, 0, 1])
        test_size_value = 0.5
        random_state_val = 42

        X_train, X_test, y_train, y_test = split_model_data(X_data, y_data, test_size_value=test_size_value,
                                                            random_state_val=random_state_val)

        self.assertEqual(len(X_train) + len(X_test), len(X_data))
        self.assertEqual(len(y_train) + len(y_test), len(y_data))
        self.assertEqual(len(np.unique(y_train)), len(np.unique(y_data)))


    def test_build_data_dictionary(self):
        '''
        Unit test for function that  Construct a dictionary with the training and testing data
        '''
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[7, 8], [9, 10]])
        y_train = np.array([0, 1, 0])
        y_test = np.array([1, 0])

        data_dict = build_data_dictionary(X_train, X_test, y_train, y_test)

        self.assertIn(X_TRAIN, data_dict)
        self.assertIn(X_TEST, data_dict)
        self.assertIn(Y_TRAIN, data_dict)
        self.assertIn(Y_TEST, data_dict)

        self.assertTrue(np.array_equal(data_dict[X_TRAIN], X_train))
        self.assertTrue(np.array_equal(data_dict[X_TEST], X_test))
        self.assertTrue(np.array_equal(data_dict[Y_TRAIN], y_train))
        self.assertTrue(np.array_equal(data_dict[Y_TEST], y_test))

        X_train = np.array([])
        X_test = np.array([])
        y_train = np.array([])
        y_test = np.array([])

        data_dict = build_data_dictionary(X_train, X_test, y_train, y_test)

        self.assertIn(X_TRAIN, data_dict)
        self.assertIn(X_TEST, data_dict)
        self.assertIn(Y_TRAIN, data_dict)
        self.assertIn(Y_TEST, data_dict)

        self.assertTrue(np.array_equal(data_dict[X_TRAIN], X_train))
        self.assertTrue(np.array_equal(data_dict[X_TEST], X_test))
        self.assertTrue(np.array_equal(data_dict[Y_TRAIN], y_train))
        self.assertTrue(np.array_equal(data_dict[Y_TEST], y_test))

    def test_shuffle_dataframe(self):
        '''
        Unit test for function  that shuffles x times a dataframe
        :return:
        '''
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        shuffled_df = shuffle_dataframe(df, no_of_times=1)

        self.assertEqual(len(shuffled_df), len(df))
        self.assertTrue(all(x in df.values for x in shuffled_df.values))
        self.assertTrue(all(x in shuffled_df.values for x in df.values))


        num_shuffles = 3
        shuffled_df = shuffle_dataframe(df, no_of_times=num_shuffles)

        self.assertEqual(len(shuffled_df), len(df))
        self.assertTrue(all(x in df.values for x in shuffled_df.values))
        self.assertTrue(all(x in shuffled_df.values for x in df.values))
        self.assertFalse(np.array_equal(shuffled_df.values, df.values))

    def test_vocabulary_dict_to_json(self):
        dictionary = {'key1': 'value1', 'key2': 123, 'key3': [1, 2, 3]}
        output_file_path = 'test_output.json'

        save_dict_to_json_file(dictionary, output_file_path)

        self.assertTrue(os.path.exists(output_file_path))

        with open(output_file_path, 'r') as json_file:
            saved_dictionary = json.load(json_file)

        self.assertEqual(saved_dictionary, dictionary)

        if os.path.exists(output_file_path):
            os.remove(output_file_path)

if __name__ == '__main__':
    unittest.main()