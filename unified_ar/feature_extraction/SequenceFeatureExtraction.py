from feature_extraction.feature_abstract import FeatureExtraction
class SequenceFeatureExtraction(FeatureExtraction):
    
    def featureExtract(self,win):
        window=win['window']
        f=[]
        for i in range(0,window.shape[0]):
            f.append(self.datasetdscr.sensor_id_map_inverse[window.iat[i,0]])
        return f