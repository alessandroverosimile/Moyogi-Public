import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()
np.random.seed(42)

#Jacopo - importo pandas e standardscaler per normalizzare i df di attributi
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from scipy.stats import entropy
#import gower
#from scipy.spatial.distance import squareform
#from sklearn.metrics import silhouette_score
#from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
#import json
from sklearn.tree import DecisionTreeClassifier
import random
from tensorflow.keras import layers, models

#librerie per siamese
from tensorflow.keras import layers, Model
from karateclub import Graph2Vec
import networkx as nx
import itertools

#librerie per autoencoder pytorch
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, DataLoader
import torch.optim as optim

import pickle

class EarlyStoppingByBatch(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, verbose=0):
        super(EarlyStoppingByBatch, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait = 0

    def on_batch_end(self, batch, logs=None):
        current_loss = logs.get('loss')

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                #if self.verbose > 0:
                    #print(f"Early stopping triggered after {self.wait} batches with no improvement.")
                self.model.stop_training = True

class ProgressBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_batches):
        super(ProgressBarCallback, self).__init__()
        self.total_batches = total_batches

    def on_predict_batch_end(self, batch, logs=None):
        # Calcola la percentuale completata
        progress = (batch + 1) / self.total_batches * 100
        #print(f"Inferenza completata: {progress:.2f}%")

class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphAutoencoder, self).__init__()
        
        # Encoder: un GCN (Graph Convolutional Network) per ridurre la dimensione del grafo
        self.encoder = pyg_nn.GCNConv(in_channels, hidden_channels)
        
        # Decoder: una rete per ricostruire il grafo dagli embedding latenti
        self.decoder = pyg_nn.GCNConv(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Encoder
        x = self.encoder(x, edge_index)
        x = torch.relu(x)
        
        # Decoder
        x = self.decoder(x, edge_index)
        
        return x

    def get_embeddings(self, data):
        # Ottieni gli embeddings latenti senza decodificare
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        x = torch.relu(x)
        x = torch.mean(x, dim=0)
        return x

class EarlyStopping:
    def __init__(self, patience=20, delta=0, save_best_model=True, model_path="best_model.pth"):
        self.patience = patience  # Numero di epoche senza miglioramenti prima di fermarsi
        self.delta = delta  # Cambiamento minimo nella perdita per essere considerato un miglioramento
        self.best_loss = np.inf  # Inizializza la migliore perdita come infinita
        self.counter = 0  # Conta le epoche senza miglioramenti
        self.early_stop = False  # Flag per determinare quando fermare l'allenamento
        self.save_best_model = save_best_model
        self.model_path = model_path  # Dove salvare il miglior modello

    def __call__(self, val_loss, model):
        # Se la perdita valida è migliorata, aggiorna e resetta il contatore
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # Salva il modello con la perdita migliore
            if self.save_best_model:
                torch.save(model.state_dict(), self.model_path)
        else:
            self.counter += 1
        
        # Se il contatore supera la pazienza, fermati
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

def train_autoencoder(model, data_loader, optimizer, criterion, epochs=10000, early_stopping=None):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            
            # Esegui il forward pass
            reconstructed_data = model(data)
            
            # Calcola la loss (ad esempio, MSE per la ricostruzione)
            loss = criterion(reconstructed_data, data.x)
            
            # Backpropagation e ottimizzazione
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)

        #if epoch % 10 == 0:
            #print(f'Epoch {epoch}, Loss: {avg_loss}')

        if early_stopping:
            if early_stopping(avg_loss, model):
                #print(f'Early stopping triggered at epoch: {epoch}')
                break

def get_graph_embeddings_autoencoder(model, graphs):
    model.eval()
    
    embeddings = []
    for graph in graphs:
        embedding = model.get_embeddings(graph)
        embeddings.append(embedding.detach().cpu().numpy().tolist())
    
    return embeddings

def extract_tree_as_graph(tree):
    num_nodes = tree.tree_.node_count
    edges = []
    node_features = []

    for i in range(num_nodes):
        # Caratteristiche per ogni nodo
        feature_index = tree.tree_.feature[i]  # Indice della feature di split
        split_value = tree.tree_.threshold[i]  # Valore di split
        impurity = tree.tree_.impurity[i]  # Impurità (Gini o Entropia)
        num_samples = tree.tree_.n_node_samples[i]  # Numero di campioni nel nodo
        
        # Costruisci il vettore delle caratteristiche del nodo
        node_features.append([feature_index, split_value, impurity, num_samples])
        
        # Aggiungi gli archi tra il nodo padre e i figli
        if tree.tree_.children_left[i] != tree.tree_.children_right[i]:  # Se il nodo non è una foglia
            left = tree.tree_.children_left[i]
            right = tree.tree_.children_right[i]
            edges.append([i, left])  # Nodo padre -> figlio sinistro
            edges.append([i, right])  # Nodo padre -> figlio destro
    
    edges = np.array(edges).T  # Ogni colonna è un arco tra due nodi
    edge_index = torch.tensor(edges, dtype=torch.long)
    node_features = np.array(node_features)
    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


class MultiDepthNeuralRandomForestClassifier:
    def __init__(self, min_depth, max_depth, depth_distribution = None, total_estimators = 100, random_state = 3, kernel_constraint = True, weighted=False, weighted_percentage = None, rescheduling = False, exclude_columns=None, method='Hungarian', autoencoder = True, 
            batch_size_mul=None,
            learning_rate=None,
            embedding_dim=None,
            #walk_length=None,
            optimizer=None,
            mul_neurons=None,
            noise_std=None,
            margin=None,
            dropout_rate=None,
            lr_autoencoder=None,
            batch_size_autoencoder=None,
            optimizer_autoencoder=None,
            save_trees=False,
            folder=None):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.estimator_for_depth = np.ones(max_depth-min_depth+1,dtype='int')*int(total_estimators/(max_depth-min_depth+1))
        if depth_distribution is not None:
            assert np.sum(depth_distribution) == total_estimators, "The distribution over the depths must sum the overall number of DTs"
            self.estimator_for_depth = depth_distribution
        self.random_state = random_state
        self.classifiers = {}
        self.kernel_constraint = kernel_constraint
        self.rescheduling = rescheduling
        self.min_data = []
        self.max_data = []

        #Jacopo - variabile per il numero degli alberi
        self.total_estimators = total_estimators
        self.weighted = weighted
        self.weighted_percentage = weighted_percentage

        #Jacopo - variabile per mantenere il df delle predizioni degli alberi sul training set
        self.df_predictions_train = pd.DataFrame()
        self.df_predictions_val = pd.DataFrame()
        self.df_predictions = pd.DataFrame()
        self.reordered_clusters = {}
        self.exclude_columns = exclude_columns
        self.method = method

        self.avg_estimator_per_depth = int(total_estimators/(max_depth-min_depth+1))

        self.autoencoder = autoencoder

        self.batch_size_mul = batch_size_mul
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        #self.walk_length = walk_length
        self.optimizer = optimizer
        self.mul_neurons = mul_neurons
        self.noise_std = noise_std
        self.margin = margin
        self.dropout_rate = dropout_rate

        self.lr_autoencoder = lr_autoencoder
        self.batch_size_autoencoder = batch_size_autoencoder
        self.optimizer_autoencoder = optimizer_autoencoder

        self.save_trees = save_trees
        self.folder = folder

    def build_model(self, input_shape, lambda_reg):
        seed = 1234
        tf.random.set_seed(seed)
        input = tfkl.Input(shape=input_shape)
        if not self.kernel_constraint:
            output = tfkl.Conv1D(1,len(self.classifiers.keys()),padding='valid',kernel_initializer = tfk.initializers.Ones(),kernel_regularizer=tfk.regularizers.L1L2(lambda_reg,lambda_reg),activation='relu',strides=len(self.classifiers.keys()), name='conv1D')(input)
        else:
            output = tfkl.Conv1D(1,len(self.classifiers.keys()),padding='valid',kernel_initializer = tfk.initializers.Ones(),kernel_regularizer=tfk.regularizers.L1L2(lambda_reg,lambda_reg),activation='relu',strides=len(self.classifiers.keys()), kernel_constraint = tfk.constraints.NonNeg(), name='conv1D')(input)
        output = output[:,:,0]
        model = tfk.Model(inputs=input, outputs=output, name='model')
        model.compile(loss=tfk.losses.CategoricalCrossentropy(from_logits=True), optimizer=tfk.optimizers.Adam(1e-3), metrics=['accuracy'])
        return model

    def to_ohe(self,y):
        y_ohe = np.zeros((y.shape[0],int(np.max(y)+1)))
        for i in range(y.shape[0]):
            y_ohe[i,int(y[i])] = 1
        return y_ohe

    def fit(self, X_train, y_train, X_val, y_val):
        
        #calcolo il valore minimo e massimo per l'hungarian rescheduling
        for i in range(X_train.shape[1]):
            min_val = X_train[:, i].min() if isinstance(X_train, np.ndarray) else X_train.iloc[:, i].min()
            max_val = X_train[:, i].max() if isinstance(X_train, np.ndarray) else X_train.iloc[:, i].max()

            self.min_data.append(min_val)
            self.max_data.append(max_val)

        self.n_classes = int(np.max(y_train)+1)
        assert (self.max_depth+1-self.min_depth)%2 == 1, 'The difference between max and min depth must be even'
        for i,depth in enumerate(range(self.min_depth,self.max_depth+1)):
            classifier = RandomForestClassifier(n_estimators=self.estimator_for_depth[i],max_depth=depth,random_state=self.random_state)
            classifier.fit(X_train,y_train)
            self.classifiers[depth] = classifier
        
        # Estrai tutti gli alberi dopo l'addestramento delle RF
        trees = self.extract_trees()
    
        samples = []
        for i in range(len(X_train)):
            sample = []
            predictions = []
            for classifier in self.classifiers.values():
                prediction = classifier.predict_proba(X_train[i:i+1])[0]
                predictions.append(prediction)

            predictions = np.array(predictions)
            
            for i in range(len(predictions[0])):
                for j in range(len(predictions)):
                    sample.append(predictions[j,i])

            samples.append(sample)
        samples = np.array(samples)

        #Jacopo - mi salvo le predictions del train set per albero
        predictions_train = []
        for idx in range(len(X_train)):
            predictions = []
            for single_tree in trees:
                prediction = single_tree.predict_proba(X_train[idx:idx+1])
                predictions.append(prediction)

            predictions = np.array(predictions)

            predictions_train.append(predictions)
        predictions_train = np.array(predictions_train)
        self.predictions_train = predictions_train

        #Jacopo - converto predizioni in un df per essere più comodo nel raggruppare gli alberi
        n_samples = len(X_train)
        n_trees = self.total_estimators
        columns = []
        for tree_idx in range(n_trees):
            tree_predictions = [self.predictions_train[i, tree_idx, :] for i in range(n_samples)]
            max_prob_indices = [np.argmax(prediction) for prediction in tree_predictions]
            columns.append(pd.Series(max_prob_indices, name=f'tree_{tree_idx}'))
        self.df_predictions_train = pd.concat(columns, axis=1)

        samples_val = []
        for i in range(len(X_val)):
            sample = []
            predictions = []
            for classifier in self.classifiers.values():
                prediction = classifier.predict_proba(X_val[i:i+1])[0]
                predictions.append(prediction)

            predictions = np.array(predictions)
            
            for i in range(len(predictions[0])):
                for j in range(len(predictions)):
                    sample.append(predictions[j,i])

            samples_val.append(sample)
        samples_val = np.array(samples_val)

        #Jacopo - mi salvo le predictions del validation set per albero
        predictions_val = []
        for i in range(len(X_val)):
            predictions = []
            for single_tree in trees:
                prediction = single_tree.predict_proba(X_val[i:i+1])[0]
                predictions.append(prediction)

            predictions = np.array(predictions)

            predictions_val.append(predictions)
        predictions_val = np.array(predictions_val)
        self.predictions_val = predictions_val
        
        #Jacopo - converto predizioni in un df per essere più comodo nel raggruppare gli alberi
        n_samples = len(X_val)
        n_trees = self.total_estimators
        columns = []
        for tree_idx in range(n_trees):
            tree_predictions = [self.predictions_val[i, tree_idx, :] for i in range(n_samples)]
            # Trova l'indice della predizione con probabilità massima per ogni campione
            max_prob_indices = [np.argmax(prediction) for prediction in tree_predictions]
            columns.append(pd.Series(max_prob_indices, name=f'tree_{tree_idx}'))
        self.df_predictions_val = pd.concat(columns, axis=1)

        y_train_ohe = self.to_ohe(y_train)
        y_val_ohe = self.to_ohe(y_val)

        model = self.build_model((self.n_classes*len(self.classifiers.keys()),1),0)
        
        history = model.fit(
            x = samples,
            y = y_train_ohe,
            batch_size = 256,
            epochs = 5000,
            validation_data = (samples_val, y_val_ohe),
            callbacks = [
                tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True),
            ]
        ).history

        predictions = model.predict(samples_val)

        self.model = model

        trees, self.weights = self.extract_trees_and_weights()

        if not self.weighted:
            self.weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        #self.df_predictions = self.create_attribute_matrices(trees, X_train, X_val)

        #Jacopo - Creo i pesi per ogni albero
        weights_for_trees = np.repeat(self.weights, self.estimator_for_depth)
        
        self.trees_and_weights = list(zip(trees, weights_for_trees))

        trees_and_weights_with_index = [(idx, tree_weight) for idx, tree_weight in enumerate(self.trees_and_weights)]

        trees_and_weights_with_index_weighted = sorted(trees_and_weights_with_index, key=lambda x: abs(x[1][1]), reverse=True)

        idx_only_weight = [(i + 1, x[0]) for i, x in enumerate(trees_and_weights_with_index_weighted)]

        if not self.rescheduling:
            trees_and_weights_with_index = [(idx, tree_weight) for idx, tree_weight in enumerate(self.trees_and_weights)]
            trees_and_weights_with_index.sort(key=lambda x: abs(x[1][1]), reverse=True)
            #new_indices = [item[0] for item in trees_and_weights_with_index]
            #print(new_indices)
            
            self.trees_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            #print('Nessun Rescheduling')

        else:
            if self.method == 'Hierarchical':
                self.trees_and_weights = self.hierarchical_rescheduling(self.trees_and_weights, self.df_predictions)
            elif self.method == 'ChainCorrelation':
                self.trees_and_weights = self.chaincorrelation_rescheduling(self.trees_and_weights, self.df_predictions_val)
            elif self.method == 'NeuralNetwork':
                self.trees_and_weights = self.neuralnetwork_rescheduling(self.trees_and_weights, self.df_predictions_val)
            elif self.method == 'Hungarian':
                self.trees_and_weights, idx_reordering = self.hungarian_rescheduling(self.trees_and_weights)
            elif self.method == 'Siamese':
                self.trees_and_weights, idx_reordering = self.siamese_rescheduling(self.trees_and_weights)
            else:
                self.trees_and_weights = self.chaincorrelation_rescheduling(self.trees_and_weights, self.df_predictions_val)
        
        print(idx_reordering)
        
        if self.weighted:
            # Combinazione lineare degli ordini
            alpha=self.weighted_percentage

            # Creiamo un dizionario con i nuovi ordinamenti per confrontare e combinare
            weighted_order_dict = {item[1]: i for i, item in enumerate(idx_only_weight)}  # Ordine basato sui pesi
            rescheduling_dict = {item[1]: i for i, item in enumerate(idx_reordering)}  # Ordine basato su order

            # Calcoliamo un nuovo ordinamento combinato
            combined_order = []

            for idx in range(len(self.trees_and_weights)):
                # Per ogni 'idx', prendi i due ordini e fai una combinazione lineare
                weighted_position = weighted_order_dict[idx]
                order_position = rescheduling_dict[idx]
                
                # Combinazione lineare degli indici
                combined_position = alpha * weighted_position + (1 - alpha) * order_position
                combined_order.append((combined_position, idx))

            # Ordinamento finale in base agli indici combinati
            combined_order.sort(key=lambda x: x[0], reverse=True)

            # Ricostruisci l'ordinamento finale mantenendo solo (tree, weight)
            reordered_trees_and_weights = [(self.trees_and_weights[i[1]][0], self.trees_and_weights[i[1]][1]) for i in combined_order]

            # Risultato finale
            self.trees_and_weights = reordered_trees_and_weights
        
        if self.save_trees:
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

            for index, (tree, _) in enumerate(self.trees_and_weights):
                filename = os.path.join(self.folder, f"DT_{index}.pkl") 
                with open(filename, 'wb') as f:
                    pickle.dump(tree, f)
    
    def siamese_rescheduling(self, trees_and_weights):

        all_embeddings = self.get_graph_embeddings(trees_and_weights)
        similarity_matrix = self.compute_similarity_matrix(self.model, all_embeddings)
        
        trees = [tree for tree, _ in trees_and_weights]

        reordered_trees, order = self.reorder_trees_by_similarity_hungarian(trees, similarity_matrix)

        reordered_trees_and_weights = [(trees_and_weights[i][0], trees_and_weights[i][1]) for i in order]

        reordered_trees_and_weights_with_index = [(trees_and_weights[i][0], trees_and_weights[i][1], i) for i in order]

        idx_reordering = [(i + 1, x[2]) for i, x in enumerate(reordered_trees_and_weights_with_index)]

        return reordered_trees_and_weights, idx_reordering

    def get_graph_embeddings(self, trees_and_weights):
        graphs = self.trees_to_graphs(trees_and_weights)
        #print("--------------------------------------------------------------------")
        #print(len(graphs))
        #print("--------------------------------------------------------------------")
        if self.autoencoder:
            print("AUTOENCODER")
            in_channels = 4
            hidden_channels = self.embedding_dim
            out_channels = 4
            embeddings = self.get_graph_embeddings_from_autoencoder(graphs, in_channels, hidden_channels, out_channels)
            #print(len(embeddings))
            #print("--------------------------------------------------------------------")
            #print(len(embeddings[0]))
            #print("--------------------------------------------------------------------")
        else:
            print("GRAPH2VEC")
            embeddings = self.get_graph_embeddings_from_graphs(graphs)
            #print(len(embeddings[0]))
            #print("--------------------------------------------------------------------")
        return embeddings

    def trees_to_graphs(self, trees_and_weights):
        graphs = []
        for tree_id, (tree, _) in enumerate(trees_and_weights):
            graph = self.decision_tree_to_graph(tree, tree_id)
            if not self.autoencoder:
                graph = nx.relabel_nodes(graph, {node: i for i, node in enumerate(graph.nodes())})
            graphs.append(graph)
        return graphs

    def decision_tree_to_graph(self, tree, tree_id):
        tree_ = tree.tree_
        G = nx.DiGraph()

        def add_nodes_edges(tree_, parent_name, node_id):
            if tree_.feature[node_id] != -2:
                threshold = tree_.threshold[node_id]
                G.add_node(f"{tree_id}_{node_id}", Threshold=f'Threshold <= {threshold:.2f}')
                G.add_edge(parent_name, f"{tree_id}_{node_id}")
                add_nodes_edges(tree_, f"{tree_id}_{node_id}", tree_.children_left[node_id])
                add_nodes_edges(tree_, f"{tree_id}_{node_id}", tree_.children_right[node_id])
            else:
                class_name = np.argmax(tree_.value[node_id])
                G.add_node(f"{tree_id}_{node_id}", Class=f'Class: {class_name}')
                G.add_edge(parent_name, f"{tree_id}_{node_id}")

        add_nodes_edges(tree_, 'root', 0)
        
        if not self.autoencoder:
            return G

        else:
            #print("Tieni il fiato sospeso...")
            graph_data = extract_tree_as_graph(tree)
            #print("Forse ci siamo...")
            return graph_data

    def get_graph_embeddings_from_graphs(self, graphs):
        model = Graph2Vec(dimensions = self.embedding_dim)
        model.fit(graphs)
        embeddings = model.get_embedding()
        #embeddings = []
        #for graph in graphs:
            #graph2vec.fit([graph])
            #embedding = graph2vec.get_embedding()
            #embeddings.append(embedding)
        return embeddings

    def get_graph_embeddings_from_autoencoder(self, graphs, in_channels, hidden_channels, out_channels, patience=20):
        # Crea il modello
        #print("FIN QUI TUTTO OK")
        model = GraphAutoencoder(in_channels, hidden_channels, out_channels)

        if self.optimizer_autoencoder == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr_autoencoder)
        elif self.optimizer_autoencoder == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.lr_autoencoder, momentum=0.9)
        elif self.optimizer_autoencoder == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.lr_autoencoder)
        elif self.optimizer_autoencoder == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=self.lr_autoencoder)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr_autoencoder)
        criterion = nn.MSELoss()

        # Usa il DataLoader di PyTorch Geometric per gestire grafi di dimensioni diverse
        data_loader = DataLoader(graphs, batch_size=self.batch_size_autoencoder, shuffle=True)

        early_stopping = EarlyStopping(patience=patience, delta=0.001)

        # Allenamento del modello
        train_autoencoder(model, data_loader, optimizer, criterion, epochs=10000, early_stopping=early_stopping)

        model.load_state_dict(torch.load(early_stopping.model_path))
        
        # Ottenere gli embeddings dai grafi
        embeddings = get_graph_embeddings_autoencoder(model, graphs)

        return embeddings

    def contrastive_loss(self, y_true, y_pred, margin=1.0):

        y_true = tf.cast(y_true, tf.float32)
        loss = 0.5 * (y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(self.margin - y_pred, 0)))
        return tf.reduce_mean(loss)

    def train_siamese_model(self, input_shape=(128,)):

        def create_siamese_model(input_shape, mul_neurons, dropout_rate):
            # Definire un layer condiviso
            input = layers.Input(input_shape)
            x = layers.Dense(32*mul_neurons, activation="relu")(input)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(16*mul_neurons, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(8*mul_neurons, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x)
            model = Model(inputs=input, outputs=x)
            return model

        # Costruire la rete Siamese
        base_model = create_siamese_model(input_shape, mul_neurons = self.mul_neurons, dropout_rate=self.dropout_rate)

        # Due input identici per la rete Siamese
        input_a = layers.Input(input_shape)
        input_b = layers.Input(input_shape)

        # Passare ogni input attraverso la rete condivisa
        output_a = base_model(input_a)
        output_b = base_model(input_b)

        # Calcolare la distanza tra le due uscite (distances)
        distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([output_a, output_b])

        # Aggiungere un layer finale per predire la probabilità di similarità
        output = layers.Dense(1, activation="sigmoid")(distance)

        # Definire il modello
        siamese_model = Model(inputs=[input_a, input_b], outputs=output)

        if self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        # Compilare il modello
        siamese_model.compile(optimizer=optimizer, loss=self.contrastive_loss, metrics=['accuracy'])

        return siamese_model

    def generate_pairs_with_noise(self, graph_embeddings, noise_std=0.01):
        pairs = []
        labels_out = []
        
        # Genera una coppia simile e una coppia diversa per ogni embedding
        for i in range(len(graph_embeddings)):
            # Coppia simile: ogni embedding viene accoppiato con se stesso con rumore gaussiano
            noisy_embedding = graph_embeddings[i] + np.random.normal(0, noise_std, np.array(graph_embeddings[i]).shape)
            pairs.append([graph_embeddings[i], noisy_embedding])
            labels_out.append(1)  # Etichetta 1 per coppie simili con rumore
        
            # Coppia diversa: seleziona un embedding casuale
            j = np.random.randint(0, len(graph_embeddings))  # Scegli un indice casuale
            while i == j:  # Assicurati che non venga scelto lo stesso grafo
                j = np.random.randint(0, len(graph_embeddings))
            pairs.append([graph_embeddings[i], graph_embeddings[j]])
            labels_out.append(0)  # Etichetta 0 per coppie diverse
        
        return np.array(pairs), np.array(labels_out)

    def compute_similarity_matrix(self, siamese_model, graph_embeddings):
        pairs_train, labels_out = self.generate_pairs_with_noise(graph_embeddings, noise_std=self.noise_std)

        siamese_model = self.train_siamese_model(input_shape = (self.embedding_dim,))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',    # Monitoriamo la 'val_loss'
            patience=20,           # Stop dopo 10 epoche senza miglioramenti
            restore_best_weights=True  # Ripristina i pesi migliori (dove val_loss è minimo)
        )

        early_stopping_batch = EarlyStoppingByBatch(patience=200, verbose=1)
        #print("start of the Siamese model's train")
        # Allenare il modello
        siamese_model.fit(
            #[np.squeeze(pairs_train[:,0], axis=1), np.squeeze(pairs_train[:,1], axis=1)],
            [pairs_train[:,0], pairs_train[:,1]], 
            labels_out, 
            epochs=1000,              # Puoi aumentare il numero di epoche
            batch_size=32,           # Impostazione della dimensione del batch
            validation_split=0.2,    # Usa il 20% dei dati per la validazione
            callbacks=[early_stopping_batch]  # Passa il callback per EarlyStopping
        ).history

        #similarities = []
        #print("end of the Siamese model's train")
        pairs = self.generate_all_pairs(graph_embeddings)
        
        #for pair in pairs:
        #    similarity = siamese_model.predict([pair[0].reshape(1, -1), pair[1].reshape(1, -1)])
        #    similarities.append(similarity[0][0])

        embeddings_1 = np.array([pair[0] for pair in pairs])  # Primo elemento di ogni coppia
        embeddings_2 = np.array([pair[1] for pair in pairs])

        #progress_callback = ProgressBarCallback(total_batches=len(pairs) // 128)  # Assumiamo batch_size=32
        #print("start of the Siamese model's inference")
        # Esegui la previsione con il progresso
        similarities = siamese_model.predict(
            [embeddings_1, embeddings_2],
            batch_size=128
            #callbacks=[progress_callback]
        )
        #print("end of the Siamese model's inference")

        n_samples = len(graph_embeddings)
        similarity_matrix = np.zeros((n_samples, n_samples))

        idx = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                similarity_matrix[i, j] = similarities[idx]
                similarity_matrix[j, i] = similarities[idx]
                idx += 1

        return similarity_matrix

    def generate_all_pairs(self, graph_embeddings):
        pairs = []
        for (i, j) in itertools.combinations(range(len(graph_embeddings)), 2):
            pairs.append([graph_embeddings[i], graph_embeddings[i]])
            pairs.append([graph_embeddings[i], graph_embeddings[j]])
        return np.array(pairs)

    def hungarian_rescheduling(self, trees_and_weights):

        # Estrai alberi e pesi
        trees = [tree for tree, _ in trees_and_weights]

        # Aggrega le regioni per label per ciascun albero
        aggregated_regions = [self.aggregate_regions_by_label(tree) for tree in trees]

        # Calcola la matrice di similarità tra alberi
        similarity_matrix = self.calculate_similarity_matrix_hungarian_aggregated(aggregated_regions)

        # Applica l'algoritmo ungherese per riordinare gli alberi
        reordered_trees, order = self.reorder_trees_by_similarity_hungarian(trees, similarity_matrix)

        # Ritorna gli alberi riordinati con i loro pesi originali
        reordered_trees_and_weights = [(trees_and_weights[i][0], trees_and_weights[i][1]) for i in order]

        reordered_trees_and_weights_with_index = [(trees_and_weights[i][0], trees_and_weights[i][1], i) for i in order]

        idx_reordering = [(i + 1, x[2]) for i, x in enumerate(reordered_trees_and_weights_with_index)]

        return reordered_trees_and_weights, idx_reordering

    def aggregate_regions_by_label(self, tree):

        regions, labels = self.extract_regions_with_labels(tree, self.min_data, self.max_data)
        aggregated = {}
        for label in set(labels):
            # Filtra le regioni per questa etichetta
            label_regions = [region for region, l in zip(regions, labels) if l == label]

            # Unisci tutte le regioni
            aggregated_region = {}
            for region in label_regions:
                for feature, low, high in region:
                    if feature not in aggregated_region:
                        aggregated_region[feature] = [low, high]
                    else:
                        aggregated_region[feature][0] = min(aggregated_region[feature][0], low)
                        aggregated_region[feature][1] = max(aggregated_region[feature][1], high)

            # Converti il dizionario in lista
            aggregated[label] = [(feature, bounds[0], bounds[1]) for feature, bounds in aggregated_region.items()]
        return aggregated

    def calculate_similarity_matrix_hungarian_aggregated(self, aggregated_regions):

        n = len(aggregated_regions)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Similarità perfetta con sé stessi
                else:
                    sim = self.aggregated_structural_similarity(aggregated_regions[i], aggregated_regions[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Matrice simmetrica
        return similarity_matrix

    def aggregated_structural_similarity(self, agg1, agg2):

        labels1 = set(agg1.keys())
        labels2 = set(agg2.keys())
        all_labels = labels1.union(labels2)

        # Costruisci una matrice di similarità tra le label
        similarity_matrix = np.zeros((len(all_labels), len(all_labels)))
        label_list = list(all_labels)
        for i, label1 in enumerate(label_list):
            for j, label2 in enumerate(label_list):
                region1 = agg1.get(label1, [])
                #print(region1)
                region2 = agg2.get(label2, [])
                #print(region2)
                similarity_matrix[i, j] = self.region_similarity(region1, region2)
                #print(similarity_matrix[i,j])

        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        total_similarity = similarity_matrix[row_ind, col_ind].sum()
        return total_similarity / len(all_labels)  

    def reorder_trees_by_similarity_hungarian(self, trees, similarity_matrix):

        n = len(trees)
        visited = [False] * n
        order = []

        current = np.argmax(similarity_matrix.sum(axis=1))
        order.append(current)
        visited[current] = True

        while len(order) < n:
            next_tree = np.argmax([similarity_matrix[current, j] if not visited[j] else -np.inf for j in range(n)])
            order.append(next_tree)
            visited[next_tree] = True
            current = next_tree

        reordered_trees = [trees[i] for i in order]
        return reordered_trees, order

    def region_similarity(self, region1, region2):

        overlap_sum = 0
        range_sum_1 = 0
        range_sum_2 = 0

        for feature1, low1, high1 in region1:
            range_sum_1 += max(0, high1 - low1)
            for feature2, low2, high2 in region2:
                if feature1 == feature2:
                    overlap_sum += max(0, min(high1, high2) - max(low1, low2))

        range_sum_2 += sum(max(0, high2 - low2) for _, low2, high2 in region2)
        total_range = range_sum_1 + range_sum_2
        return (2 * overlap_sum) / total_range if total_range > 0 else 0
    
    def extract_regions_with_labels(self, tree, data_min, data_max):
        # Estrarre le informazioni strutturali dell'albero
        tree_ = tree.tree_
        left = tree_.children_left
        right = tree_.children_right
        threshold = tree_.threshold
        features = tree_.feature
        values = tree_.value  # Valori delle predizioni nei nodi foglia

        def recurse(node, bounds):
            if left[node] == -1 and right[node] == -1:  # Nodo foglia
                # Determina l'etichetta del nodo foglia
                label = np.argmax(values[node])  # Assumendo che sia un classificatore
                return [(bounds, label)]

            regions = []
            if features[node] != -2:  # Nodo di split
                feature = features[node]
                threshold_value = threshold[node]

                # Ottieni i limiti della feature corrente
                min_bound = data_min[feature]
                max_bound = data_max[feature]

                # Dividi i confini in base alla soglia
                left_bounds = bounds + [(feature, min_bound, threshold_value)]
                right_bounds = bounds + [(feature, threshold_value, max_bound)]

                # Ricorsione sui figli
                regions.extend(recurse(left[node], left_bounds))
                regions.extend(recurse(right[node], right_bounds))

            return regions

        # Inizializza la ricorsione
        initial_bounds = []
        regions_and_labels = recurse(0, initial_bounds)

        # Dividi regioni e etichette
        regions, labels = zip(*regions_and_labels)
        return list(regions), list(labels)

    def neuralnetwork_rescheduling(self, trees_and_weights, df_predictions):

        trees = [tree for tree, _ in trees_and_weights]
        weights = [weight for _, weight in trees_and_weights]

        predictions = df_predictions.to_numpy()        
        
        def build_tree_nn(input_dim, output_dim, learning_rate=0.01):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(50, activation='relu'),
                tf.keras.layers.Dense(output_dim, activation='softmax')
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        tree_nns = []

        for i, tree in enumerate(trees):
            y_tree_pred = predictions[:, i].reshape(-1, 1)
            y_tree_pred_ohe = np.eye(self.n_classes)[y_tree_pred.flatten()]

            # Split del dataset in training e validation set
            X_train, X_val, y_train, y_val = train_test_split(predictions, y_tree_pred_ohe, test_size=0.2, random_state=42)
            
            #print(f"Tree {i}: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
            # Callback per early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
            )

            # Creazione e training del modello
            nn = build_tree_nn(X_train.shape[1], y_train.shape[1], learning_rate=0.01)
            nn.fit(
                X_train, y_train,
                epochs=200, batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            tree_nns.append(nn)

        nn_weights = [np.concatenate([w.flatten() for w in nn.get_weights()]) for nn in tree_nns]
        similarity_matrix = cosine_similarity(nn_weights)
        
        reordered_indices = []
        visited = set()

        max_similarity_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        reordered_indices.append(max_similarity_indices[0])
        reordered_indices.append(max_similarity_indices[1])
        visited.update(max_similarity_indices)

        while len(visited) < len(trees):
            last_tree = reordered_indices[-1]
            remaining_trees = set(range(len(trees))) - visited
            next_tree = max(remaining_trees, key=lambda x: similarity_matrix[last_tree, x])
            reordered_indices.append(next_tree)
            visited.add(next_tree)

        reordered_trees_and_weights = [(trees[i], weights[i]) for i in reordered_indices]

        return reordered_trees_and_weights

    def chaincorrelation_rescheduling(self, trees_and_weights, df_predictions_val):

        similarity = self.similarity_percent(df_predictions_val)
        ordered_indices = self.order_trees_by_similarity(similarity)
        trees_and_weights_ordered = [trees_and_weights[i] for i in ordered_indices]

        return trees_and_weights_ordered

    def similarity_percent(self, df_predictions_val):

        temp = np.zeros((df_predictions_val.shape[1], df_predictions_val.shape[1]))
        df_predictions_val_np = df_predictions_val.to_numpy()
        for i in range(df_predictions_val.shape[1]):
            for j in range(df_predictions_val.shape[1]):
                a = df_predictions_val_np[:,i]==df_predictions_val_np[:,j]
                temp[i,j] = np.mean(a)

        return temp
    
    def order_trees_by_similarity(self, similarity_matrix):
        n = similarity_matrix.shape[0]
        visited = np.zeros(n, dtype=bool)  # Array per tracciare gli alberi già visitati
        order = []  # Lista per mantenere l'ordine finale degli alberi

        # Trova gli indici dei due alberi con la percentuale di corrispondenza più alta
        i, j = np.unravel_index(np.argmax(np.triu(similarity_matrix, 1)), similarity_matrix.shape)
        order.extend([i, j])  # Aggiungi i due alberi iniziali all'ordine
        visited[i] = visited[j] = True  # Segna questi alberi come visitati

        # Costruisci l'ordine andando a catena con l'albero più correlato disponibile
        while len(order) < n:
            last_tree = order[-1]
            # Trova il prossimo albero con la massima similarità tra quelli non visitati
            next_tree = np.argmax(similarity_matrix[last_tree] * ~visited)
            
            # Aggiungi l'albero trovato all'ordine e segna come visitato
            order.append(next_tree)
            visited[next_tree] = True

        return order

    def extract_trees_and_weights(self):
        trees = self.extract_trees()
        conv_layer = self.model.get_layer('conv1D')
        weights, _ = conv_layer.get_weights()
        #Jacopo - normalizzo i pesi per il numero di alberi presenti nella RF di depth corrispondente
        weights = weights.ravel() / self.estimator_for_depth
        
        return trees, weights


    def predict_proba(self,X_test):
        samples = []
        for i in range(len(X_test)):
            sample = []
            predictions = []
            for classifier in self.classifiers.values():
                prediction = classifier.predict_proba(X_test[i:i+1])[0]
                predictions.append(prediction)
            predictions = np.array(predictions)

            for i in range(len(predictions[0])):
                for j in range(len(predictions)):
                    sample.append(predictions[j,i])
            
            samples.append(sample)
        
        samples = np.array(samples)
        predictions_nn = self.model.predict(samples)
        return predictions_nn
    
    #Jacopo - funzione per estrarre i trees dalle RF
    def extract_trees(self):
        trees = []
        for classifier in self.classifiers.values():
            trees.extend(classifier.estimators_)
        return trees
    
    #Jacopo - funzione che implementa l'early stopping
    def predict_proba_jacopo(self, X_test):
        #Jacopo - Somma dei pesi totali
        total_weights = np.sum([np.abs(tw[1]) for tw in self.trees_and_weights]) 
        early_stopping_idx = []

        all_predictions = []

        for i in range(len(X_test)):
            predictions = np.zeros(self.n_classes)
            current_weight_sum = 0

            for idx, (tree, weight) in enumerate(self.trees_and_weights):
                prediction = tree.predict_proba(X_test[i:i+1])[0] * weight
                predictions += prediction
                current_weight_sum += np.abs(weight)

                #Jacopo - Fermo l'esecuzione se il massimo non può essere superato
                sorted_predictions = np.sort(predictions)
                current_max = sorted_predictions[-1]
                second_max = sorted_predictions[-2]
                remaining_weight_sum = total_weights - current_weight_sum

                if current_max >= (second_max + remaining_weight_sum):
                    early_stopping_idx.append(idx + 1)
                    break

            all_predictions.append(predictions)

        return all_predictions, early_stopping_idx
    
    def predict(self,X_test):
        predictions, idx_tree = self.predict_proba_jacopo(X_test)
        return np.argmax(predictions, axis=1), idx_tree
    
    def score(self,X_test,y_test):
        predictions, idx_tree = self.predict(X_test)
        return np.sum(predictions==y_test)/len(predictions), idx_tree, self.reordered_clusters, self.trees_and_weights, self.df_predictions, self.df_predictions_val
