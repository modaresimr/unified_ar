![MetaDecomposition_page-0001](https://user-images.githubusercontent.com/9498182/226872997-4399d3c5-5a29-4c7f-9d13-48455d76f45c.jpg)
![MetaDecomposition_page-0002](https://user-images.githubusercontent.com/9498182/226873001-1dbc85bd-c5dd-435b-8760-1e18da12ae17.jpg)
![MetaDecomposition_page-0003](https://user-images.githubusercontent.com/9498182/226873004-fce458e6-c1c0-4abe-ac62-455760c6efad.jpg)
![MetaDecomposition_page-0004](https://user-images.githubusercontent.com/9498182/226873005-b54dc1a9-206c-451c-a57a-cb4d08ac918c.jpg)
![MetaDecomposition_page-0005](https://user-images.githubusercontent.com/9498182/226873008-a55e4f13-bf48-4245-beda-43e0edd3de00.jpg)
![MetaDecomposition_page-0006](https://user-images.githubusercontent.com/9498182/226873011-8684a1ae-f6fc-4d2a-88be-42c056512497.jpg)
![MetaDecomposition_page-0007](https://user-images.githubusercontent.com/9498182/226873013-2d92cbdd-47aa-4256-94d6-f37ef0a68531.jpg)
![MetaDecomposition_page-0008](https://user-images.githubusercontent.com/9498182/226873014-aa42b83b-9788-40ea-b537-07d8c5f4ccba.jpg)
![MetaDecomposition_page-0009](https://user-images.githubusercontent.com/9498182/226873016-a60e05de-887a-4c28-ae0c-b97974fc24e9.jpg)
![MetaDecomposition_page-0010](https://user-images.githubusercontent.com/9498182/226873020-a3f78f2e-1f5a-4049-8975-2e8bd69432c1.jpg)
![MetaDecomposition_page-0011](https://user-images.githubusercontent.com/9498182/226873021-dd39fbfd-aad9-49e3-8f9e-cb9a57784cc7.jpg)
![MetaDecomposition_page-0012](https://user-images.githubusercontent.com/9498182/226873023-415a011f-882a-4bd0-a263-f069db43fe1c.jpg)
![MetaDecomposition_page-0013](https://user-images.githubusercontent.com/9498182/226873024-68c2239d-d641-48f8-b5e4-6dff625d668a.jpg)
![MetaDecomposition_page-0014](https://user-images.githubusercontent.com/9498182/226873025-6f3c5930-e05f-422c-81f3-c35e29788b11.jpg)
![MetaDecomposition_page-0015](https://user-images.githubusercontent.com/9498182/226873027-f847c619-dc97-43d6-9684-07b6ab399293.jpg)
![MetaDecomposition_page-0016](https://user-images.githubusercontent.com/9498182/226873028-5de79a7c-914b-4adb-b0ad-42909b9ad7bf.jpg)
![MetaDecomposition_page-0017](https://user-images.githubusercontent.com/9498182/226873032-25df9a71-df4b-4269-83dd-3aab91789b08.jpg)
![MetaDecomposition_page-0018](https://user-images.githubusercontent.com/9498182/226873036-1a47b6fa-b79a-4a21-98e9-1d655d18cab7.jpg)
![MetaDecomposition_page-0019](https://user-images.githubusercontent.com/9498182/226873038-3c59e0ce-16e9-45ee-8136-c60ddad338e1.jpg)
![MetaDecomposition_page-0020](https://user-images.githubusercontent.com/9498182/226873040-cee3fec3-db0d-41ee-b0f0-aa097e5dbead.jpg)
![MetaDecomposition_page-0021](https://user-images.githubusercontent.com/9498182/226873042-12efc30c-e41b-46e7-87b8-6b67c0066d4e.jpg)
![MetaDecomposition_page-0022](https://user-images.githubusercontent.com/9498182/226873044-2fed7db8-fe40-4f00-95b1-5b3f74da39e1.jpg)
![MetaDecomposition_page-0023](https://user-images.githubusercontent.com/9498182/226873049-9e2ba2b8-f983-4097-b3d8-ffe367292901.jpg)
![MetaDecomposition_page-0024](https://user-images.githubusercontent.com/9498182/226873051-60da0e93-2a28-4854-ac62-11f45f44a8fe.jpg)
![MetaDecomposition_page-0025](https://user-images.githubusercontent.com/9498182/226873053-00571444-4883-4296-81ef-df5afb07b3fb.jpg)
![MetaDecomposition_page-0026](https://user-images.githubusercontent.com/9498182/226873056-c2b60e9a-f774-4245-8127-575d922d328c.jpg)
![MetaDecomposition_page-0027](https://user-images.githubusercontent.com/9498182/226873058-8f78d402-e951-4316-b203-582dfe83b367.jpg)
![MetaDecomposition_page-0028](https://user-images.githubusercontent.com/9498182/226873060-f5d8df33-09c0-44e8-b777-cb4c1798acbb.jpg)
![MetaDecomposition_page-0029](https://user-images.githubusercontent.com/9498182/226873062-9187806b-2525-477b-a760-dfee0209d2d9.jpg)



# DataSet
## Dataset Format:
### Object:
    An object is a primitive object, a vector or in the form of a tuple of data components:
    Object ={o|     o is Primitive or
	                o=[o_1, ... , o_n] such that o_i is Object(Vector of object) or
	                o=(Prop_1, ... , Prop_n) forall i in {1...n}, Prop_i(o) is Object}
### Time Object:
Time might be a point, in case of an instantaneous event, or an interval during if it is durative. Supported durative time is range.

    time | [start_time:end_time]


### Event:
|Type|Actor| Time |
|-|-|-|
### Sensor Events:
|(Type, Value)|SensorId| Time |
|-|-|-|

### Activity Events:
|ActivityId|ActorId| Time |
|-|-|-|

### DataInformation:
#### Sensor Info
| Id | Name | Cumulative | OnChange | Nominal | Range | Location | Object | Sensor |
|-|-|-|-|-|-|-|-|-|



#### Activity Info
|Id|Name|
|-|-|




### File format: CSV
#### Sensor Info:
| Id | Name | Cumulative | OnChange | Nominal | Range | Location | Object | Sensor |
|-|-|-|-|-|-|-|-|-|
| int | string | bool | bool | bool | json {min,max}/{items} | string | string | string |
in case of nominal sensors, the range contain items and for numeric sensors, the range contain min and max

#### Sensor events:
|Type | Value | SensorId | Time |
|-|-|-|-|

#### Activity events:
|ActivityId|ActorId| StartTime | EndTime|
|-|-|-|-|

![](http://yuml.me/diagram/scruffy/class/[Preprocessing]->[Dispacher],[Dispacher]->[Segmentation],[Segmentation]->[FeatureExtraction],[FeatureExtraction]->[Classifier],[Classifier]->[Combiner],[Combiner]->[Evaluation])

#### Approaches
\begin{Example}[Different Segmentation approaches]
\end{Example}
    \begin{lstlisting}[mathescape=true]
function Fixed time window(S,X,r,l) {//S=SegmentHistory, X=Events, 
         //r=Shift, l=windowLength
    p=begin(S[last])
    return X.eventsIn([p + r : p + r + l]); 
}
function Fixed siding window(S,X,r,l) {
    prev_w=S[last]; p=begin(S[last])
    be=first({e \in X| p + r $\leq$ time(e)}
    return X.eventsIn([be : be + l]); 
}
function Significant events(S,X,m) {//m=significant events per segments
    se=significantEvents(X) $\subseteq$ X
    begin=time(se[1]);//next significant event 
    end=time(se[1 + m]);
    return X.eventsIn([begin:end]); 
}
//Probabilistic Approach
given:(By analyzing training set) 
    $ws(A_m)$ is average window size of activity $A_m$
    $w_1 = min \{ws(A_1), ws(A_2), ..., ws(A_M)\}$
    $w_L = median\{ws(A_1), ws(A_2), ..., ws(A_M)\}$
    $w_l=(w_L-w_1)\times l/L+w_1$
    $window\_sizes= \{w_1, w_2, . . . , w_L\}$
    $P(w_l /A_m)$//probability of windows length $w_l$ for an activity Am
    $P(A_m /s_i)$//probability of Activity $A_i$ associated with the sensor $s_i$.
function Probabilistic Approach(S,X) {
    x=nextEvent(X)
    $w^{\star} =\underset{w_l}{max}  \{P(w_l /x)\}=\underset{w_l}{max}[P(w_l /A_m)\times P(A_m /x)] $
    end=time(x);//Next event
    return X.eventsIn(end-$w^\star$,end]); 
}
function Metric base Approach(S,X) {//S=SegmentHistory, X=Events    
    indx=len(S[last])+1 //first event not in old segment
    $m_i=metric(\{X[indx],...,X[i]\})$
    find first i which $H(\{m_{0}....m_i\})$ is true// 
    return X.eventsIn([time(X[indx]):time(X[i])]); 
}
function SWAB Approach(S,X,bs) {//bs=Buffer size    
    indx=len(S[last])+1 //first event not in old segment
    $m=BottomUp(\{X[indx],...,X[indx+bs]\})$
    return m[0]; 
}
\end{lstlisting}


# Similar Works
[pyActLearn](https://github.com/TinghuiWang/pyActLearn/) -> [documents](https://pyactlearn.readthedocs.io/en/latest/)
