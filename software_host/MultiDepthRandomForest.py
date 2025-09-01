import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.keras.config.disable_interactive_logging()
import tensorflow as tf
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
np.random.seed(42)

class MultiDepthRandomForestClassifier:
    def __init__(self, min_depth, max_depth, total_estimators = 100, random_state = 3):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.estimator_per_depth = int(total_estimators/(max_depth-min_depth+1))
        assert self.estimator_per_depth>0, "too few estimators for the requested depths"
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        self.n_classes = int(np.max(y_train)+1)
        self.classifiers = {}
        for depth in range(self.min_depth,self.max_depth+1):
            classifier = RandomForestClassifier(n_estimators=self.estimator_per_depth,max_depth=depth,random_state=self.random_state)
            classifier.fit(X_train,y_train)
            self.classifiers[depth] = classifier
        
    def predict_proba(self,X_test,weights='uniform'):
        predictions = np.zeros((X_test.shape[0],self.n_classes))
        if weights == 'uniform':
            for depth in self.classifiers:
                predictions += self.classifiers[depth].predict_proba(X_test)
        if weights == 'linear':
            for depth in self.classifiers:
                predictions += depth*self.classifiers[depth].predict_proba(X_test)
        
        if weights == 'quadratic':
            for depth in self.classifiers:
                predictions += depth**2*self.classifiers[depth].predict_proba(X_test)

        return predictions
    
    def predict(self,X_test,weights='uniform'):
        predictions = self.predict_proba(X_test,weights)
        return np.argmax(predictions,axis=1)
    
    def score(self,X_test,y_test,weights='uniform'):
        assert weights == 'uniform' or weights == 'linear' or weights == 'quadratic', 'Weights should be uniform, linear or quadratic'
        predictions = self.predict(X_test, weights)
        return np.sum(predictions==y_test)/len(predictions)

class MultiDepthNeuralRandomForestClassifier:
    def __init__(self, min_depth, max_depth, depth_distribution = None, total_estimators = 100, random_state = 3):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.estimator_per_depth = np.ones(max_depth-min_depth+1,dtype='int')*int(total_estimators/(max_depth-min_depth+1))
        if depth_distribution is not None:
            assert np.sum(depth_distribution) == total_estimators, "The distribution over the depths must sum the overall number of DTs"
            self.estimator_per_depth = depth_distribution
        self.random_state = random_state
        self.classifiers = {}
    
    def build_model(self, input_shape, lambda_reg):
        seed = 1234
        tf.random.set_seed(seed)
        input = tfkl.Input(shape=input_shape)
        output = tfkl.Conv1D(1,len(self.classifiers.keys()),padding='valid',kernel_initializer = tfk.initializers.Ones(),kernel_regularizer=tfk.regularizers.L1L2(lambda_reg,lambda_reg),activation='relu',strides=len(self.classifiers.keys()), kernel_constraint = tfk.constraints.NonNeg(), name='conv1D')(input)
        output = output[:,:,0]
        model = tfk.Model(inputs=input, outputs=output, name='model')
        model.compile(loss=tfk.losses.CategoricalCrossentropy(from_logits=True), optimizer=tfk.optimizers.Adam(1e-3), metrics=['accuracy'])
        return model

    def to_ohe(self,y,n_classes):
        y_ohe = np.zeros((y.shape[0],n_classes))
        for i in range(y.shape[0]):
            y_ohe[i,int(y[i])] = 1
        return y_ohe

    def fit(self, X_train, y_train, X_val, y_val, patience):
        
        self.n_classes = int(np.max(y_train)+1)
        assert (self.max_depth+1-self.min_depth)%2 == 1, 'The difference between max and min depth must be even'
        
        for i,depth in enumerate(range(self.min_depth,self.max_depth+1)):
            #print(f"Training Random Forest {i+1}")
            classifier = RandomForestClassifier(n_estimators=self.estimator_per_depth[i],max_depth=depth,random_state=self.random_state)
            classifier.fit(X_train,y_train)
            self.classifiers[depth] = classifier
        
        samples = np.zeros((X_train.shape[0],self.n_classes*(self.max_depth-self.min_depth+1)))
        for i,classifier in enumerate(self.classifiers.values()):
            prediction = classifier.predict_proba(X_train)
            for j in range(prediction.shape[1]):
                samples[:,j*len(self.classifiers.values())+i] = prediction[:,j]

        samples_val = np.zeros((X_val.shape[0],self.n_classes*(self.max_depth-self.min_depth+1)))
        for i,classifier in enumerate(self.classifiers.values()):
            prediction = classifier.predict_proba(X_val)
            for j in range(prediction.shape[1]):
                samples_val[:,j*len(self.classifiers.values())+i] = prediction[:,j]

        y_train_ohe = self.to_ohe(y_train,self.n_classes)
        y_val_ohe = self.to_ohe(y_val,self.n_classes)
        
        model = self.build_model((self.n_classes*len(self.classifiers.keys()),1),0)
        #print("training NN")
        history = model.fit(
            x = samples,
            y = y_train_ohe,
            batch_size = 256,
            epochs = 5000,
            validation_data = (samples_val, y_val_ohe),
            callbacks = [
                tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True),
            ]
        ).history

        #xpredictions = model.predict(samples_val)

        self.model = model
    
    def predict_proba(self,X_test):
   
        samples = np.zeros((X_test.shape[0],self.n_classes*(self.max_depth-self.min_depth+1)))
        for i,classifier in enumerate(self.classifiers.values()):
            prediction = classifier.predict_proba(X_test)
            for j in range(prediction.shape[1]):
                samples[:,j*len(self.classifiers.values())+i] = prediction[:,j]     
        
        samples = np.array(samples)
        predictions_nn = self.model.predict(samples)
        return predictions_nn
    
    def predict(self,X_test):
        predictions = self.predict_proba(X_test)
        return np.argmax(predictions,axis=1)
    
    def score(self,X_test,y_test):
        predictions = self.predict(X_test)
        return np.sum(predictions==y_test)/len(predictions)
    
    def all_metrics(self,X_test,y_test):
        predictions = self.predict(X_test)
        probs = self.predict_proba(X_test)
        accuracy = np.sum(predictions==y_test)/len(predictions)
        cm = confusion_matrix(y_test, predictions)
        return accuracy, cm, probs