import os
import json

from kgltools.data_io.submission_writer import SubmissionWriter

__all__ = ['KglToolsContext']


class KglToolsContext():

	def __init__(self, settings_path: str) -> None:
        if not os.path.exists(settings_path):
            print('Settings file {} is not exist!'.format(settings_path))
            return

        with open(settings_path, 'r') as settings_file:
        	self.settings = json.load(param_file)

    def getDataTools():
    	pass
