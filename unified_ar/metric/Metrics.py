from IPython.display import display


class GEM:
    classical = False

    def eval(self, real_a_event, pred_a_event, acts, debug=0):
        import metric.MyMetric as mymetric
        return mymetric.eval(real_a_event, pred_a_event, acts, debug)

    def __str__(self):
        return 'GEM OLD'


class GEM_NEW:
    classical = False

    def eval(self, real_a_event, pred_a_event, acts, debug=0):
        import metric.GEM_NEW as mymetric

        if 'Activity' in real_a_event.columns:
            real_a_event = real_a_event.rename({'StartTime': 'onset', 'EndTime': 'offset', 'Activity': 'event_label'}, axis=1)
            pred_a_event = pred_a_event.rename({'StartTime': 'onset', 'EndTime': 'offset', 'Activity': 'event_label'}, axis=1)
            import numpy as np
            real_a_event['onset'] = real_a_event['onset'].astype(np.int64)
            real_a_event['offset'] = real_a_event['offset'].astype(np.int64)
            real_a_event = real_a_event.loc[real_a_event['onset'] < real_a_event['offset']]
            pred_a_event['onset'] = pred_a_event['onset'].astype(np.int64)
            pred_a_event['offset'] = pred_a_event['offset'].astype(np.int64)
            pred_a_event = pred_a_event.loc[pred_a_event['onset'] < pred_a_event['offset']]
            real_a_event['filename'] = 'a'
            pred_a_event['filename'] = 'a'

        s = min(real_a_event['onset'].min(), pred_a_event['onset'].min())-1
        e = max(real_a_event['offset'].min(), pred_a_event['offset'].min())+1
        return mymetric.eval(real_a_event, pred_a_event, meta=[s, e], clas=acts, debug=debug)

    def __str__():
        return 'GEM'


class Classical:
    classical = True

    def eval(self, rlabel, plabel, acts):
        import metric.MyClassical as myclassical
        return myclassical.eval(rlabel, plabel, acts)

    def __str__(self):
        return 'Classical'


class EventCM:
    classical = False

    def eval(self, real_a_event, pred_a_event, acts, debug=0):
        from metric.CMbasedMetric import CMbasedMetric
        from metric.event_confusion_matrix import event_confusion_matrix
        quality = {}
        for act in acts:
            real = real_a_event.loc[real_a_event.Activity == act].copy()
            pred = pred_a_event.loc[pred_a_event.Activity == act].copy()
            real.Activity = 1
            pred.Activity = 1
            # display(pred_a_event.Activity)
            event_cm = event_confusion_matrix(real, pred, [0, 1])

            metr = CMbasedMetric(event_cm, None, None)
            quality[act] = {p: metr[p][1] for p in metr if p != 'accuracy'}
            quality[act]['tp'] = event_cm[1, 1]
            quality[act]['fn'] = event_cm[1, 0]
            quality[act]['fp'] = event_cm[0, 1]
            quality[act]['tn'] = event_cm[0, 0]
        if len(acts) == 1:
            return {'EventCM1': quality[act]}
        return {'EventCM3': quality}

    def __str__(self):
        return 'EventCM'


class Tatbul:
    classical = False

    def eval(self, real_a_event, pred_a_event, acts, debug=0):
        import metric.TatbulMetric as mytatbul
        return mytatbul.eval(real_a_event, pred_a_event, acts, debug)

    def __str__(self):
        return 'Tatbul'
