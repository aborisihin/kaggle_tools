import os
from datetime import datetime

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

__all__ = ['DataTools']


class DataTools():

    def __init__(self, settings: dict) -> None:
        self.settings = settings

    def write_submission(self, predictions: np.ndarray) -> bool:
        submission_settings = self.settings['submission_params']

        sample_submission_path = os.path.join(self.settings['path'], submission_settings['sample_file'])

        if not os.path.exists(sample_submission_path):
            print('DataTools::write_submission(): sample submission file is not exist!')
            return False

        sample_sbm = pd.read_csv(sample_submission_path, **submission_settings['pd_read_csv_params'])
        sample_sbm[submission_settings['target_fields']] = predictions

        sbm_filename = '{}_sbm.csv'.format(datetime.now().strftime("%Y-%m-%d_%H-%M"))
        smb_filepath = os.path.join(submission_settings['submissions_dir'], sbm_filename)

        sample_sbm.to_csv(smb_filepath, **submission_settings['pd_write_csv_params'])

        print('save submission:\n{}'.format(sbm_filename))

    def get_validate_split(self,
                           X: pd.DataFrame,
                           y: Optional[pd.DataFrame] = None,
                           validation_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_train, X_val = train_test_split(X, test_size=validation_size, shuffle=True, stratify=y)
        return (X_train, X_val)
