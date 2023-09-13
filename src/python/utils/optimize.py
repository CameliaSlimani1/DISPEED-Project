import numpy


def findParetoFront (exploration_results):
    count = 0
    #pareto = {"n_estimators":[], "max_depth":[],  "accuracy":[], "size":[]}
    pareto = {"n_estimators": ["model_summary"],  "accuracy": [], "size": []}

    accuracies = numpy.array(exploration_results["accuracy"])
    sorted_indexes = ((-accuracies).argsort())
    print(sorted_indexes)
    list_indexes = list(sorted_indexes)
    #pareto["n_estimators"].append(exploration_results["n_estimators"][list_indexes[0]])
    pareto["model_summary"].append(exploration_results["model_summary"][list_indexes[0]])
    pareto["accuracy"].append(exploration_results["accuracy"][list_indexes[0]])
    pareto["size"].append(exploration_results["size"][list_indexes[0]])
    for i in range(1,len(list_indexes)) :
        j = i
        add = True
        while j>=0 and add :
            if  exploration_results["size"][list_indexes[i]] > exploration_results["size"][list_indexes[j]] :
                add = False
            j -= 1
        if add :
            #pareto["n_estimators"].append(exploration_results["n_estimators"][list_indexes[i]])
            pareto["model_summary"].append(exploration_results["model_summary"][list_indexes[i]])
            pareto["accuracy"].append(exploration_results["accuracy"][list_indexes[i]])
            pareto["size"].append(exploration_results["size"][list_indexes[i]])
            count += 1
    return pareto, count