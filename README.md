# spark-predictor

This is the machine learning based- Spark performance prediction model. Unlike some existing prediction,  this prediction model predict workload execution time against stages in the Spark workload.

### How to use ?

```python
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

```

