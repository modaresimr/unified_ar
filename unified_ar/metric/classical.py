from metric.metric_abstract import AbstractClassifierMetric
import numpy as np
# class MyEvaluator(AbstractEvaluation):
#     def evaluate_dataset(self,dataset,plabel,pprob):
#         pa_events = convertAndMergeToEvent(dataset.set_window,plabel)
#         a_events = convertAndMergeToEvent(dataset.set_window,dataset.label)
#         return self.qualitymeasurement(pa_events,a_events)
#     def confusion_matrix_dataset(self, dataset,plabel,pprob):
#         pa_events = convertAndMergeToEvent(dataset.set_window,plabel)
#         a_events =dataset.a_events # convertAndMergeToEvent(dataset.set_window,dataset.label)
#         print(a_events)
#         cm=myaccuracy(pa_events,a_events)
#         return cm
#     def evaluate(self,time_window, rlabel,plabel,pprob):
#         pa_events = convertAndMergeToEvent(time_window,plabel)
#         a_events = convertAndMergeToEvent(time_window,rlabel)
#         return self.qualitymeasurement(pa_events,dataset.a_events)


#     def confusion_matrix(self, time_window, rlabel,plabel,pprob):
#          pa_events = convertAndMergeToEvent(dataset,plabel)
#          a_events = convertAndMergeToEvent(time_window,rlabel)
#          cm=myaccuracy(pa_events,a_events)
#          return cm


#     def qualitymeasurement(self,p_events,r_events):
#         cm=myaccuracy(p_events,r_events)
#         s=np.array(cm).sum().sum()
#         correct=0
#         for i in range(0,len(cm)):
#             correct+=cm[i,i]
#         return correct/s


class ClassicalMetric(AbstractClassifierMetric):
    # def evaluate_dataset(self,dataset,plabel,pprob):
    #     return self.evaluate(dataset.set,dataset.label,plabel,pprob,dataset.acts)
    # def confusion_matrix_dataset(self, dataset,plabel,pprob):
    #     return self.confusion_matrix(dataset.set,dataset.label,plabel,pprob,dataset.acts)

    # def evaluate(self, rset, rlabel,plabel,pprob,labels):
    #     ;

    # def confusion_matrix(self, rset, rlabel,plabel,pprob,labels):
    #     cm=confusion_matrix(rlabel,plabel,labels)
    #     return cm

    def get_tp_fp_fn_tn(self, cm):
        cm = np.array(cm)
        np.seterr(divide='ignore', invalid='ignore')
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1).T - TP
        num_classes = len(cm)
        TN = []
        for i in range(num_classes):
            temp = np.delete(cm, i, 0)    # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TN.append(temp.sum())
        return TP, FP, FN, TN

    def eval_cm(self, cm, average=None):
        pass

    # def event_cofusion_matrix(self, p_activity,r_activity):
    #     ;


class Accuracy(ClassicalMetric):
    def evaluate(self, rset, rlabel, plabel, pprob, labels):
        return sklearn.metrics.accuracy_score(rlabel, plabel)

    def eval_cm(self, cm, average=None):
        TP, FP, FN, TN = self.get_tp_fp_fn_tn(cm)
        a = TP.sum()/cm.sum()
        if (average is None):
            return None
        return a


class Precision(ClassicalMetric):

    def evaluate(self, rset, rlabel, plabel, pprob, labels):
        p, r, f, s = sklearn.metrics.precision_recall_fscore_support(rlabel, plabel, 1, labels, average='macro')
        return p

    def eval_cm(self, cm, average=None):
        TP, FP, FN, TN = self.get_tp_fp_fn_tn(cm)
        p = TP/(TP+FP)
        if (average is None):
            return p
        return np.average(p[~np.isnan(p)])


class Recall(ClassicalMetric):
    def evaluate(self, rset, rlabel, plabel, pprob, labels):
        p, r, f, s = sklearn.metrics.precision_recall_fscore_support(rlabel, plabel, 1, labels, average='macro')
        return r

    def eval_cm(self, cm, average=None):
        TP, FP, FN, TN = self.get_tp_fp_fn_tn(cm)
        r = TP/(TP+FN)
        if (average is None):
            return r
        return np.average(r[~np.isnan(r)])


class F1Evaluator(ClassicalMetric):
    def evaluate(self, rset, rlabel, plabel, pprob, labels):
        p, r, f, s = sklearn.metrics.precision_recall_fscore_support(rlabel, plabel, 1, labels, average='macro')
        return f

    def eval_cm(self, cm, average=None):
        TP, FP, FN, TN = self.get_tp_fp_fn_tn(np.array(cm))
        # print(np.array(TP).shape,np.array(FP).shape,np.array(FN).shape,np.array(TN).shape)
        # print(TP,FP,FN,TN)
        p = TP/(TP+FP)
        r = TP/(TP+FN)
        f = 2*r*p/(r+p)
        if (average is None):
            return f
        return np.average(f[~np.isnan(f)])


class All(ClassicalMetric):
    def evaluate(self, rset, rlabel, plabel, pprob, labels, average='macro'):
        p, r, f, s = sklearn.metrics.precision_recall_fscore_support(rlabel, plabel, 1, labels, average=average)
        return {'precision': p, 'recall': r, 'f1': f}

    def eval_cm(self, cm, average=None):
        TP, FP, FN, TN = self.get_tp_fp_fn_tn(np.array(cm))
        # print(np.array(TP).shape,np.array(FP).shape,np.array(FN).shape,np.array(TN).shape)
        # print(TP,FP,FN,TN)
        p = TP/(TP+FP)
        r = TP/(TP+FN)
        f = 2*r*p/(r+p)
        if (average is None):
            return f
        return np.average(f[~np.isnan(f)])


# def myaccuracy(predicted0,real0):
#   begin=real0.StartTime.min()
#   end=real0.EndTime.max()
#   predicted=[p for i,p in predicted0.iterrows()]
#   real=[p for i,p in real0.iterrows()]

# #   predicted.append({'StartTime':begin,'EndTime':end,'Activity':0})
# #   real.append({'StartTime':begin,'EndTime':end,'Activity':0})
#   events=merge_split_overlap_IntervalTree(real,predicted)
#   #predictedtree=makeIntervalTree(labels)

#   cm=np.zeros((len(activities),len(activities)))

#   for eobj in events:
#       e=eobj.data
#       pact=e.P.Activity if not(e.P is None) else 0
#       ract=e.R.Activity if not(e.R is None) else 0
#       cm[pact][ract]+=max((eobj.end-eobj.begin)/pd.to_timedelta('60s').value,0.01);

#   #for p in predicted:
#   #  for q in realtree[p['StartTime'].value:p['EndTime'].value]:
#   #      timeconfusion_matrix[p['Activity']][q.data['Activity']]+=findOverlap(p,q.data);

#   return cm

# #test_a_events_arr=[x for i,x in test_a_events.iterrows() ]
# #matrix=myaccuracy(test_a_events_arr,testactp)
# #pd.DataFrame(matrix)
# #plot_confusion_matrix(matrix,activities)
# #ev=prepareEval(runs)
# # print(ev[0]['cm'])
# # np.set_printoptions(precision=0)
# # np.set_printoptions(suppress=True)

# # print(np.array(ev[0]['mycm']))


# def myaccuracy2(predicted0,real0):
#   predicted=[p for i,p in predicted0.iterrows()]
#   real=[p for i,p in real0.iterrows()]
#   realtree=makeIntervalTree(real)
#   #predictedtree=makeIntervalTree(predicted)
#   predictedtree=makeNonOverlapIntervalTree(predicted)
#   #predictedtree=makeIntervalTree(labels)

#   timeconfusion_matrix=np.zeros((len(activities),len(activities)))

#   for p in real:
#     for q in predictedtree[p['StartTime'].value:p['EndTime'].value]:
#       timeconfusion_matrix[p['Activity']][q.data['Activity']]+=findOverlap(p,q.data);

#   #for p in predicted:
#   #  for q in realtree[p['StartTime'].value:p['EndTime'].value]:
#   #      timeconfusion_matrix[p['Activity']][q.data['Activity']]+=findOverlap(p,q.data);

#   return timeconfusion_matrix

# #test_a_events_arr=[x for i,x in test_a_events.iterrows() ]
# #matrix=myaccuracy(test_a_events_arr,testactp)
# #pd.DataFrame(matrix)
# #plot_confusion_matrix(matrix,activities)


# def event_confusion_matrix(r_activities,p_activities,labels):
#     cm=np.zeros((len(labels),len(labels)))
#     # begin=real0.StartTime.min()
#     # end=real0.EndTime.max()
#     predicted=[p for i,p in p_activities.iterrows()]
#     real=[p for i,p in r_activities.iterrows()]

#     #   predicted.append({'StartTime':begin,'EndTime':end,'Activity':0})
#     #   real.append({'StartTime':begin,'EndTime':end,'Activity':0})
#     events=merge_split_overlap_IntervalTree(real,predicted)
#     #predictedtree=makeIntervalTree(labels)

#     cm=np.zeros((len(activities),len(activities)))

#     for eobj in events:
#         e=eobj.data
#         pact=e.P.Activity if not(e.P is None) else 0
#         ract=e.R.Activity if not(e.R is None) else 0
#         cm[ract][pact]+=max((eobj.end-eobj.begin)/pd.to_timedelta('60s').value,0.01);

#     #for p in predicted:
#     #  for q in realtree[p['StartTime'].value:p['EndTime'].value]:
#     #      timeconfusion_matrix[p['Activity']][q.data['Activity']]+=findOverlap(p,q.data);

#     return cm


# def merge_split_overlap_IntervalTree(r_acts,p_acts):
#     tree=IntervalTree()

#     for act in p_acts:
#         start=act['StartTime'].value;
#         end=act['EndTime'].value;
#         if(start==end):
#             start=start-1
#         #tree[start:end]={'P':{'Activitiy':act.Activity,'Type':'P','Data':act}]
#         d=Data('P-act')
#         d.P=act
#         d.R=None
#         tree[start:end]=d #{'P':act,'PActivitiy':act.Activity}

#     for act in r_acts:
#         start=act['StartTime'].value;
#         end=act['EndTime'].value;
#         if(start==end):
#             start=start-1
#         #tree[start:end]=[{'Activitiy':act.Activity,'Type':'R','Data':act}]
#         d=Data('P-act')
#         d.P=None
#         d.R=act
#         tree[start:end]=d #{'R':act,'RActivitiy':act.Activity}

#     tree.split_overlaps()
#     def data_reducer(x,y):
#         res=x
#         if not(y.P is None):
#             if (res.P is None) or y.P['EndTime']<res.P['EndTime']:
#                 res.P=y.P
#         if not(y.R is None):
#             if (res.R is None) or y.R['EndTime']<res.R['EndTime']:
#                 res.R=y.R
#         return res

#     tree.merge_equals(data_reducer=data_reducer)

#     return tree
