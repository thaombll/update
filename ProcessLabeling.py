import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.cluster import KMeans
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
part1 = pd.read_csv('/home/tranthanhthao/update/1.csv')
part2 = pd.read_csv('/home/tranthanhthao/update/2.csv')
part6 = pd.read_csv('/home/tranthanhthao/update/3.csv')
pd.concat([part1, part2, part3],ignore_index=True).to_csv('agnews_explain_embedding.csv', index = False)
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size1, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.4)
        self.relu1 = nn.ReLU()
        
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.dropout2 = nn.Dropout(0.3)
        # self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_size1, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        
        # x = self.fc2(x)
        # x = self.dropout2(x)
        # x = self.relu2(x)
        
        x = self.fc3(x)
        return x
    

def train_model(train_loader, model, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # pri(train_loader)}")
    return model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

def update_df(labeled_df, unlabeled_df, move_to_labeled):
    new_labeled_data = unlabeled_df.iloc[move_to_labeled]
    labeled_df = pd.concat([labeled_df, new_labeled_data], ignore_index=True)
    unlabeled_df = unlabeled_df.drop(move_to_labeled).reset_index(drop=True)
    return labeled_df, unlabeled_df

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.cluster import KMeans

import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)

hidden_size = 256
hidden_size1 = 128 
criterion = nn.CrossEntropyLoss()
epochs = 10


class Labeling:
    def __init__(self, labeled_path, unlabeled_path, labels, task_domain, feature_columns, 
                 label_columns, dataset_name=None, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                 measure_uncertainty="entropy", confidence_threshold=0.5,
                 k_selection=1, cost=150, num_init=150
                 ,keyword_embedding_path = None,
                 model = None):
        self.labeled_path = labeled_path
        self.unlabeled_path = unlabeled_path
        self.labels = labels
        self.task_domain = task_domain
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.dataset_name = dataset_name
        self.model_id = model_id
        self.measure_uncertainty = measure_uncertainty
        self.confidence_threshold = confidence_threshold
        self.k_selection = k_selection
        self.cost = cost
        self.num_init = num_init
        self.keyword_embedding_path = keyword_embedding_path
        self.model = model

    def caculate_confidence_score(self, model, data_loader):
        scores = []
        model.eval()  
        
        with torch.no_grad(): 
            for batch in data_loader:
                inputs = batch[0]  

                inputs = inputs.to(device)  
                outputs = model(inputs)  
                
                if self.measure_uncertainty == "entropy":
                    probabilities = F.softmax(outputs, dim=1)
                    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                    scores.extend(entropy.cpu().numpy())
                
                elif self.measure_uncertainty == "min_margin":
                    probabilities = F.softmax(outputs, dim=1)
                    pmax, psecond = torch.topk(probabilities, 2, dim=1)
                    margin = (pmax[:, 0] - psecond[:, 1]).cpu().numpy()  # Ensure the correct indices
                    scores.extend(margin)
                
                elif self.measure_uncertainty == "least_confidence":
                    probabilities = F.softmax(outputs, dim=1)
                    least_confidence = 1 - torch.max(probabilities, dim=1)[0]
                    scores.extend(least_confidence.cpu().numpy())
        
        return scores

    
    def human_labeling(self, indice, data_loader):
        row = data_loader.iloc[indice]
        return row[self.label_columns]

    def active_learning(self, model, unlabeled_df):
        features = unlabeled_df[self.feature_columns].values
        unlabeled_dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32)

        scores = self.caculate_confidence_score(model, unlabeled_loader)
        
        score_df = pd.DataFrame({
            'index': range(len(scores)),
            'confidence_score': scores
        })

        selected_indices = score_df.nlargest(self.k_selection, 'confidence_score')['index'].values

        if (self.cost < self.k_selection):
            selected_indices = score_df.nlargest(self.cost, 'confidence_score')['index'].values
            self.cost = 0
        else:
            self.cost -= self.k_selection

        labels = []
        for index in selected_indices:
            label = self.human_labeling(index, unlabeled_df)  
            labels.append(label)

        return selected_indices, labels
    
    def bootstrap(self, unlabeled_df, num_init, save_path="labeled.csv"):
        features = unlabeled_df[self.feature_columns].values
        
        kmeans = KMeans(n_clusters=num_init, random_state=42)
        kmeans.fit(features)
        
        cluster_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        selected_indices = []

        for cluster_idx in range(num_init):
            cluster_points = np.where(cluster_labels == cluster_idx)[0]  
            cluster_data = features[cluster_points]  
            
            distances = np.linalg.norm(cluster_data - centroids[cluster_idx], axis=1)
            
            selected_idx = cluster_points[np.argmin(distances)]
            selected_indices.append(selected_idx)

        labeled_data = unlabeled_df.iloc[selected_indices]
        
        labeled_data.to_csv(save_path, index=False)
        print(f"Saved {len(selected_indices)} initial labeled datapoints to {save_path}.")

        return labeled_data
    
    def get_model_label(sefl, model, current_datapoint):
        return model(torch.tensor(current_datapoint, dtype=torch.float32).to(device)).argmax().item()
    
    def get_llm_label(sefl, i, unlabeled_df):
        llm_label = -1
        # print(unlabeled_df.iloc[i])
        if i < len(unlabeled_df) and "LLM_label" in unlabeled_df.columns:
            llm_label = unlabeled_df.iloc[i]["LLM_label"]
        return llm_label
    
    def remove_keyword (self, text, word):
        new_text = text
        regex = re.compile(re.escape(word), re.IGNORECASE)
        new_text = regex.sub('', new_text)
        new_text = new_text.replace('  ', ' ')
        return new_text

    def check_explain_LLM(self, point_data, point_unlabeled_df, model,keyword_embedding_df):
        inputs = torch.tensor(point_data, dtype=torch.float32).to(device)
        # inputs = point_data
        with torch.no_grad():  
            outputs = model(inputs) 
            prob_origin = torch.softmax(outputs, dim=0)

        label_text_origin_predict = prob_origin.argmax(dim=0).cpu().numpy() 
         
        prob_max_origin = prob_origin.max(dim=0)[0].cpu().numpy()

        if point_unlabeled_df['Keyword_Explain'] == '':
            return 0
        elif pd.isna(point_unlabeled_df['Keyword_Explain']):
            return 0
        else:
            for keyword in point_unlabeled_df['Keyword_Explain'].split(', '):
                new_text = self.remove_keyword(point_unlabeled_df['text'], keyword)
                # print(f'bug: {keyword_embedding_df['text']}')
                check = keyword_embedding_df[keyword_embedding_df['text'] == new_text]
                
                if not check.empty:
                    inputs = torch.tensor(check[features].values, dtype=torch.float32).to(device)
                    with torch.no_grad():  
                        outputs = model(inputs)  
                        prob_explain = torch.softmax(outputs, dim=0)
                    
                    label_text_explain_predict = prob_explain.argmax(dim=0).cpu().numpy()  
                    prob_max_explain = prob_explain.max(dim=0)[0].cpu().numpy()

                    if (label_text_explain_predict == label_text_origin_predict).any() and (prob_max_origin <= prob_max_explain).any():
                        return 0
            return 1
        
    def update_data(sefl, labeled_df, unlabeled_df, new_labeled_data):
        # new_labeled_data = unlabeled_df.iloc[move_to_labeled]
        labeled_df = pd.concat([labeled_df, new_labeled_data], ignore_index=True)
        unlabeled_df = unlabeled_df.drop(new_labeled_data.index).reset_index(drop=True)
        return labeled_df, unlabeled_df
    
    def no_cost_condition(self,  model_label, llm_label, confidence, unlabeled_df, error):
        move_to_labeled = []
        if (model_label == llm_label and confidence <= self.confidence_threshold):
            print(f"        Use both Label with fully condition")
            # predicted_labels.append(model_label)

            if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                error += 1

            unlabeled_df.loc[unlabeled_df.index[i], self.label_columns] = model_label
            move_to_labeled.append(i)
        elif (model_label == llm_label):
            print(f"        Use both Label with fully condition")
            # predicted_labels.append(model_label)

            if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                error += 1

            unlabeled_df.loc[unlabeled_df.index[i], self.label_columns] = model_label
            move_to_labeled.append(i)
        elif confidence <= self.confidence_threshold:
            print(f"        Use Model Label with fully condition")
            # predicted_labels.append(model_label)

            if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                error += 1

            unlabeled_df.loc[unlabeled_df.index[i], self.label_columns] = model_label
            move_to_labeled.append(i)
        elif (llm_label in range(len(self.labels))):
            print(f"        Use LLM Label with fully condition")
            unlabeled_df.loc[unlabeled_df.index[i], self.label_columns] = llm_label

            if (llm_label != unlabeled_df.iloc[i][self.label_columns]):
                error += 1

            move_to_labeled.append(i)
        else:
            print(f"        Use Model Label")
            unlabeled_df.loc[unlabeled_df.index[i], self.label_columns] = model_label

            if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                error += 1

            move_to_labeled.append(i)

        return move_to_labeled, unlabeled_df, error


    def process(self):        
        unlabeled_df = pd.read_csv(self.unlabeled_path)
        labeled_df = self.bootstrap(unlabeled_df, self.num_init, save_path="labled_df.csv")
        keyword_embedding_df = pd.read_csv(self.keyword_embedding_path)
        unlabeled_df = unlabeled_df[~unlabeled_df.index.isin(labeled_df.index)]

        total_error = 0
        iteration = 0
        cost = 0
        print(f"Begin process {self.dataset_name}:")
        print(f"Cost {self.cost}:")

        while not unlabeled_df.empty:
            iteration += 1
            print(f"Iteration {iteration}:")

            X_train = labeled_df[self.feature_columns].values
            y_train = labeled_df[self.label_columns].values
            X_test = unlabeled_df[self.feature_columns].values
            y_test = unlabeled_df[self.label_columns].values


            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

            input_size = X_train.shape[1]
            num_classes = len(self.labels) 
            
            model = MLPModel(input_size, hidden_size1, num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model = train_model(train_loader, model, criterion, optimizer, epochs)
            test_model(model, test_loader)
            unlabeled_features = unlabeled_df[self.feature_columns].values
            unlabeled_dataset = TensorDataset(torch.tensor(unlabeled_features, dtype=torch.float32))
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)

            # predicted_labels = []
            confidence_scores = self.caculate_confidence_score(model, unlabeled_loader)
            print(f"min confidence_scores: {min(confidence_scores)}")
            print(f"max confidence_scores: {max(confidence_scores)}")
            count_overfit = 0
            count_llm_error = 0
            count_confidence = 0
            count_consensus = 0
            count_consensus_error = 0
            for i, confidence in enumerate(confidence_scores):
                current_datapoint = unlabeled_features[i]
                model_label = self.get_model_label(model=model, current_datapoint=current_datapoint)
                llm_label = self.get_llm_label(i, unlabeled_df)
                if confidence <= self.confidence_threshold:
                    count_confidence += 1
                    if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                        count_overfit += 1
                    if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                        count_llm_error += 1
                    # print(f"Label Model: {model_label}, Label LLM: {llm_label}")
                    if (model_label == llm_label):
                        count_consensus += 1
                        if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                            count_consensus_error += 1
            
            print(f"Mumber error in confidence score: {count_overfit} / {count_confidence}")
            print(f"Number LLM error: {count_llm_error}")
            print(f"Number consensus error: {count_consensus_error} / {count_consensus}")
            len_labeled_point = 0
            move_to_labeled = []  
            error = 0

            if self.cost >= len(unlabeled_df):
                    selected_indices = list(unlabeled_df.index) 
                    human_labels = [self.human_labeling(index, unlabeled_df) for index in selected_indices]
                    
                    new_labeled_data = unlabeled_df.copy()
                    new_labeled_data[self.label_columns] = human_labels  
                    labeled_df, unlabeled_df = self.update_data(labeled_df, unlabeled_df, new_labeled_data)

                    cost += len(selected_indices)
                    print(f"All remaining {len(selected_indices)} data points have been labeled with human labeling.")
            count_datapoint_confident = 0
            for i, confidence in enumerate(confidence_scores):
                # print(f"min confidence_scores: {min(confidence_scores)}")
                # print(f"max confidence_scores: {max(confidence_scores)}")

                current_datapoint = unlabeled_features[i]
                model_label = self.get_model_label(model=model, current_datapoint=current_datapoint)
                llm_label = self.get_llm_label(i, unlabeled_df)
                check_explain = self.check_explain_LLM(unlabeled_features[i], unlabeled_df.iloc[i], model, keyword_embedding_df)


                if (self.cost > 0):
                    if (count_datapoint_confident == 20):
                        break
                else:
                    if (count_datapoint_confident >= 100):
                        break

                if confidence <= self.confidence_threshold:
                    count_datapoint_confident += 1
                    if llm_label == model_label and check_explain == 1:
                        # predicted_labels.append(model_label)
                        print(f"confidence: {confidence}")
                        if (model_label != unlabeled_df.iloc[i][self.label_columns]):
                            error += 1
                        unlabeled_df.loc[unlabeled_df.index[i], self.label_columns] = model_label
                        move_to_labeled.append(i)
                        len_labeled_point += 1
                        break
            
            if len_labeled_point > 0:
                new_labeled_data = unlabeled_df.iloc[move_to_labeled]
                # print(f"new labeled {new_labeled_data}")
                labeled_df, unlabeled_df = self.update_data(labeled_df, unlabeled_df, new_labeled_data)
                print(f"    Process label: Moved {len_labeled_point} data points to labeled_df.")
            else:
                if (self.cost > 0):
                    selected_indices, human_labels = self.active_learning(model, unlabeled_df)
                    if (self.cost >= self.k_selection):
                        cost += self.k_selection
                    else:
                        cost += self.cost

                    new_labeled_data = unlabeled_df.iloc[selected_indices].copy()
                    new_labeled_data[self.label_columns] = human_labels 
                    labeled_df, unlabeled_df = self.update_data(labeled_df, unlabeled_df, new_labeled_data)
                    print(f"    Active-learning: Selected {len(selected_indices)} data points for manual labeling.")

                else:
                    move_to_labeled = []  
                    for i, confidence in enumerate(confidence_scores):
                        current_datapoint = unlabeled_features[i]
                        model_label = self.get_model_label(model=model, current_datapoint=current_datapoint)
                        llm_label = self.get_llm_label(i, unlabeled_df)
                        move_to_labeled, unlabeled_df, error = self.no_cost_condition(model_label, llm_label, confidence, unlabeled_df, error)
                        break

                    new_labeled_data = unlabeled_df.iloc[move_to_labeled]
                    labeled_df, unlabeled_df = self.update_data(labeled_df, unlabeled_df, new_labeled_data)
                    print(f"    Force Cost: Moved {len(move_to_labeled)} data points to labeled_df.")
                        # break

            print(f"    Cost use {cost}.")
            print(f"    Remaining unlabeled points: {len(unlabeled_df)}")
            print(f"    Error Iteration: {error}")
            total_error += error
            print(f"    Error Total: {total_error} / {len(labeled_df)}")
            print(f"------------------------------------------------------")
            if unlabeled_df.empty:
                print(" All data points have been labeled.")
                break

        print(" Final Labeled Data:", labeled_df)

        return labeled_df


    
if __name__ == "__main__":
    features = []
    for i in range(768):
        features.append(str(i))
    labeling = Labeling(labeled_path=None, 
                    unlabeled_path="/home/tranthanhthao/update/agnews_origin_embedding.csv", 
                    # keyword_embedding_path = '/home/tranthanhthao/DALAB/click_bait/Keyword_embedding.csv',                    labels=['yes', 'no'],
                    task_domain="Agnews detection",
                    labels=[0, 1, 2,3],
                    feature_columns=features, 
                    label_columns="label", 
                    dataset_name="Agnews",
                    cost=100, confidence_threshold=0.05,
                    num_init=150,
                    keyword_embedding_path ="/home/tranthanhthao/update/agnews_explain_embedding.csv",
                    )
    labeling.process()
