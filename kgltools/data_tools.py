import os
from datetime import datetime

import numpy as np
import pandas as pd

__all__ = ['KglToolsContext']


class DataTools():

	def __init__(settings: dict) -> None:
		self.settings = settings

	def write_submission(predictions: np.ndarray) -> bool:
		submission_settings = self.settings['submission_params']

		sample_submission_path = os.path.join(self.settings['path'], submission_settings['sample_file'])

		if not os.path.exists(sample_submission_path):
            print('DataTools::write_submission(): sample submission file is not exist!')
            return False

        sample_sbm = pd.read_csv(sample_submission_path, **submission_settings['pd_read_csv_params'])
        sample_sbm[submission_settings['target_fields']] = predictions

        sbm_filename = datetime.now().strftime("%Y-%m-%d_%H-%M")
	    smb_filepath = os.path.join(submission_settings['submissions_dir'], '{}_sbm.csv'.format())

	    sample_sbm.to_csv(filename, **submissions_dir['pd_write_csv_params'])

	    print('save submission:\n{}'.format(sbm_filename))
