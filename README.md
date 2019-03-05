# kgltools
Useful tools for Kaggle competitions

* core

	- [KglToolsContext](./kgltools/context.py)<br>
  	Класс контекста задачи. Используется для поддержки общей структуры задач через трансляцию пользовательских настроек.<br>

	- [Logger](./kgltools/logger.py)<br>
  	Класс общего логгера. Используется для однотипного вывода сообщений в лог с поддержкой подсчета используемого времени и отступами.<br>

* data_tools module

	- [DataTools](./kgltools/data_tools/_data_tools.py)<br>
	Класс для работы с данными (загрузка, сохранение, разбивка и т.д.)<br>

* iterative_param_search module

	- [IPSPipeline](./kgltools/iterative_param_search/_pipeline.py)<br>
	Класс пайплайна итерационного подбора параметров. Помогает автоматизировать этапы настройки гиперпараметров модели.<br>

* stacking module

	- [Stacker](./kgltools/stacking/_stacker.py)<br>
	Класс, реализующий стекинг моделей.<br>
