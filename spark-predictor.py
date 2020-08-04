from datetime import datetime
import numpy as np
import requests
from sklearn.neural_network import MLPRegressor
import pickle
import pandas as pd

GB = 1024*1024*1024
features = 6
threshold=0.03

class Executor:
    def __init__(self, exe_id, cores, memory, input_size):
        self._exe_id = exe_id
        self._cores = cores
        self._memory = memory
        self._input_size = input_size

    @property
    def cores(self):
        return self._cores

    @property
    def memory(self):
        return self._memory

    def __str__(self):
        return "ID:" + str(self._exe_id) + \
               " cores:" + str(self._cores) + \
               " memory:" + str(self._memory) + \
               " inputFileSize:" + str(self._input_size)

class Predictor:
    def __init__(self, host, base_app, first_stage):
        self.__host = host
        self.__stage_thresh = threshold
        self.__models = {}
        self.__base_app = base_app
        self.__first_stage = first_stage

    @staticmethod
    def get_duration(start, end):
        format_start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f%Z")
        format_end = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%f%Z")
        return format_end.timestamp() - format_start.timestamp()

    def get_executors(self, app_id):
        url = "http://" + self.__host + ":18080/api/v1/applications/" + app_id + "/executors"
        response = requests.get(url).json()
        executors = []
        for executor in response:
            exe = Executor(
                executor['id'],
                executor['totalCores'],
                executor['maxMemory'] / GB,
                executor['totalInputBytes'] / GB)
            executors.append(exe)
        return executors

    def training_stage(self, apps, stages, loop_time):
        dataset, time = self.__get_data(apps, stages, loop_time)
        clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(21,), random_state=1,max_iter=50)
        clf.fit(dataset, time)
        self.__models[loop_time] = clf

    @staticmethod
    def __predict_stage(dataset, index):
        file_obj = open("./model/model-ann", 'rb')
        nn_model = pickle.load(file_obj)[index]
        return nn_model.predict(dataset)

    def __save_models(self):
        modelobj = open("./model/model-ann", 'wb')
        pickle.dump(self.__models, modelobj)
        modelobj.close()

    def get_apps(self, start, end):
        url = "http://{host}:18080/api/v1/applications"
        response = requests.get(url.format(host=self.__host)).json()
        i = 0
        index_end = 1000
        apps = []
        for app in response:
            current = app['id']
            if current == end:
                index_end = i
            if current == start:
                apps.append(current)
                break
            if i >= index_end:
                apps.append(current)
            i = i + 1
        apps.reverse()
        return apps

    def get_ratio(self, app_id):
        valid = self.valid_stages_app(app_id)
        url = "http://" + self.__host + ":18080/api/v1/applications/" + app_id + "/stages"
        stages = requests.get(url).json()

        file_ratio = np.zeros((len(valid), 3))
        initial_size = 0

        i = 0
        for stage in stages:
            if stage['stageId'] in valid:
                input_size = stage['inputBytes'] / GB
                shuffle_read = stage['shuffleReadBytes'] / GB
                shuffle_write = stage['shuffleWriteBytes'] / GB
                file_ratio[i] = [input_size, shuffle_read, shuffle_write]
                i = i + 1
                if stage['stageId'] == self.__first_stage:
                    initial_size = input_size
        result = file_ratio / initial_size
        return np.flipud(result)

    def training_all(self, apps):
        valid_stages,valid_apps = self.get_valid_stages(apps)
        first_key = next(iter(valid_stages))
        for i in range(len(valid_stages[first_key])):
            self.training_stage(valid_apps, valid_stages, i)

        self.__save_models()

    def __get_data(self, apps, target_stages, loop_time):
        dataset = np.zeros((len(target_stages), features))
        time = np.zeros((len(target_stages), ))
        url = "http://" + self.__host + ":18080/api/v1/applications/{appsId}/stages"

        for i,app in zip(range(len(apps)),apps):
                valid_stage = target_stages[app]
                stages = requests.get(url.format(appsId=app)).json()
                executors = self.get_executors(app)
                total_mem = 0
                total_cores = 0
                for executor in executors:
                    total_cores = total_cores + executor.cores
                    total_mem = total_mem + executor.memory
                for stage in stages:
                    if stage['stageId'] == valid_stage[loop_time]:
                        inputSize = stage['inputBytes'] / GB
                        shuffle_read = stage['shuffleReadBytes'] / GB
                        shuffle_write = stage['shuffleWriteBytes'] / GB
                        dataset[i] = [len(executors), total_cores, inputSize, shuffle_read, shuffle_write, total_mem]
                        executor_time = self.get_duration(stage['firstTaskLaunchedTime'], stage['completionTime'])
                        time[i] = executor_time
        return dataset, time

    def get_valid_stages(self, apps:list) -> dict and list:
        valid_stages = {}
        valid_apps = []
        previous = 0
        current = 0
        for app in apps:
            stages = self.valid_stages_app(app)
            current = len(stages)
            if current == previous or previous == 0:
                valid_stages[app] = stages
                valid_apps.append(app)
                previous = current
        return valid_stages,valid_apps

    def valid_stages_app(self, app:str) -> list:
        url = "http://" + self.__host + ":18080/api/v1/applications/" + app + "/stages"
        stages = requests.get(url).json()
        total_time = 0
        stages_id = np.zeros(len(stages), )
        stages_ratio = np.zeros(len(stages), )
        for stage, i in zip(stages, range(len(stages))):
            if stage['status'] == "COMPLETE":
                duration = self.get_duration(stage['firstTaskLaunchedTime'], stage['completionTime'])
                stages_ratio[i] = duration
                stages_id[i] = stage['stageId']
                total_time = total_time + duration
        df = pd.DataFrame({
            'ratio': stages_ratio,
            'stageId': stages_id
        }).sort_values(by="stageId")
        df['ratio'] = df['ratio'] / total_time
        result = df.loc[df['ratio'] > self.__stage_thresh]
        return result['stageId'].to_list()

    def predict_all(self, conf:dict):
        mem_map = {
            1: 366.3,
            2: 912.3,
            3: 1433.6,
            4: 2048,
            5: 2764
        }
        required_mem = mem_map[conf['mem']] * (conf['exes'] - 1) + 384.1
        file_ratio = self.get_ratio(self.__base_app)
        num_stages = len(self.valid_stages_app(self.__base_app))
        total_time = 0
        for i, ratio in zip(range(num_stages), file_ratio):
            dataset = np.array([
                [conf['exes'],
                 conf['cores'],
                 conf['input'] * ratio[0],
                 conf['input'] * ratio[1],
                 conf['input'] * ratio[2],
                 required_mem / 1024]
            ])
            stage_predict = self.__predict_stage(dataset, i)
            total_time = total_time+ stage_predict
        return total_time

    def compute_accuracy(self,app:str, predict:int):
        duration = self.actual_time(app)
        accuracy = abs(1 - abs(predict - duration) / duration)
        error = abs(predict-duration)
        print("actual:{var1} predicted:{var2}".format(var1=duration,var2=predict))
        print("accuracy:{var1}%".format(var1=accuracy*100))
        print("--------------------------------------")
        return accuracy , error

    def actual_time(self,app):
        url = "http://"+self.__host+":18080/api/v1/applications/"+app+"/stages"
        stages = requests.get(url).json()
        total_time = 0
        for stage in stages:
            if stage['status'] == "COMPLETE":
                start = stage['submissionTime']
                end = stage['completionTime']
                duration = self.get_duration(start,end)
                total_time = total_time+duration
        return total_time
    def get_configuration(self,app,stageId):
        executors = self.get_executors(app)
        input_size = total_cores = total_mem = 0
        for executor in executors:
            total_cores = total_cores + executor.cores
            total_mem = total_mem + executor.memory

        url = "http://" + self.__host + ":18080/api/v1/applications/{appsId}/stages"
        stages = requests.get(url.format(appsId=app)).json()
        for stage in stages:
            if stage['stageId'] == stageId:
                input_size = stage['inputBytes'] / GB
                break
        return {
            "input":input_size,
            "cores":total_cores,
            "exes":len(executors),
            "mem":total_mem
        }
if __name__ == '__main__':
    """
    Predictor needs three parameters:
    first:  your history server host
    second: base application, used to calculate the ratio between input file size 
            and shuffle data
    third:  which stage reads data from file system
    """
    predictor = Predictor("ygjmaster","app-20200707034055-0020",0)

    """
    get historical execution data from first param(app-20200714123425-0025) to second param (app-20200714124512-0036)
    """
    apps = predictor.get_apps("app-20200714123425-0025","app-20200714124512-0036")
    predictor.training_all(apps)
    conf = {
        "exes": 4, # number of executors
        "cores": 6, # number of cores
        "input": 1.89, # input data size
        "mem": 5 #
    }
    # predict execution time according to given configuration above
    result = predictor.predict_all(conf)
    print(result)
