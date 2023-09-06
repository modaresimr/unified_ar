from general.utils import MyTask


class AbstractClassifierMetric(MyTask):
    def evaluate_dataset(self, dataset, plabel, pprob):
        pass

    def confusion_matrix_dataset(self, dataset, plabel, pprob):
        pass

    def evaluate(self, rset, rlabel, plabel, pprob):
        pass

    def confusion_matrix(self, rset, rlabel, plabel, pprob):
        pass

    def eval_cm(self, cm, average=None):
        pass


class AbstractEventMetric(MyTask):
    def evaluate_dataset(self, dataset, plabel, pprob):
        pa_events = convertAndMergeToEvent(dataset.set_window, plabel)
        a_events = convertAndMergeToEvent(dataset.set_window, dataset.label)
        return self.qualitymeasurement(pa_events, a_events)

    def confusion_matrix_dataset(self, dataset, plabel, pprob):
        pa_events = convertAndMergeToEvent(dataset.set_window, plabel)
        # convertAndMergeToEvent(dataset.set_window,dataset.label)
        a_events = dataset.a_events
        print(a_events)
        cm = myaccuracy(pa_events, a_events)
        return cm

    def evaluate(self, time_window, rlabel, plabel, pprob):
        pa_events = convertAndMergeToEvent(time_window, plabel)
        a_events = convertAndMergeToEvent(time_window, rlabel)
        return self.qualitymeasurement(pa_events, dataset.a_events)

    def confusion_matrix(self, time_window, rlabel, plabel, pprob):
        pa_events = convertAndMergeToEvent(dataset, plabel)
        a_events = convertAndMergeToEvent(time_window, rlabel)
        cm = myaccuracy(pa_events, a_events)
        return cm

    def qualitymeasurement(self, p_events, r_events):
        cm = myaccuracy(p_events, r_events)
        s = np.array(cm).sum().sum()
        correct = 0
        for i in range(0, len(cm)):
            correct += cm[i, i]
        return correct/s
