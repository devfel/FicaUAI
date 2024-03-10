import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import warnings


class DataModel:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
        self.accuracy = 0
        self.loss = 0
        self.predictions = None
        self.prob_predictions = None
        self.selected_covariates = []
        self.feature_importances = {}
        self.n_hidden_layers = 0
        self.total_hidden_neurons = 0

    def load_data(self, caminho_arquivo):
        self.data = pd.read_excel(caminho_arquivo)
        return self.data

    def prepare_data(self, x_columns, y_column):
        self.X = self.data[x_columns]
        self.y = self.data[y_column]
        self.X = self.scaler.fit_transform(self.X)

    def split_data(self, test_size=0.3, seed=None):
        if seed is not None:
            random_seed = seed
        else:
            random_seed = np.random.randint(0, 2147483647)  # 2^31 - 1

        print(f"Random seed: {random_seed}")
        # Use the generated seed for splitting
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_seed
        )

        # Now you can replicate this exact split by using the printed seed
        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        X_train,
        y_train,
        activation="tanh",
        solver="adam",
        max_iter=10000,
        random_state=42,
    ):
        # Calculate the number of features (covariates)
        n_features = X_train.shape[1]

        # Calculate the number of unique outputs
        n_outputs = len(np.unique(y_train))  # Using numpy's unique function
        # AAnother way of calculating the number of unique outputs
        # n_outputs = len(unique_labels(y_train))

        # Calculate the number of neurons for the hidden layers
        n_neurons = int(2 / 3 * n_features + n_outputs)

        # Create a tuple that represents the hidden layer structure, only one hidden layer
        hidden_layer_sizes = (int(n_neurons),)
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.total_hidden_neurons = sum(hidden_layer_sizes)
        # Another way, creating multiple hidden layers.
        # Example: for 10 neurons, (10, 10, 10) would create 3 hidden layers, each with 10 neurons
        # hidden_layer_sizes = (n_neurons,) * n_neurons # This would create n_neurons hidden layers, each with n_neurons neurons

        print("Training model...")
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        self.predictions = self.model.predict(X_test)
        self.prob_predictions = self.model.predict_proba(X_test)
        self.accuracy = accuracy_score(y_test, self.predictions)
        self.loss = log_loss(y_test, self.prob_predictions)
        return self.accuracy, self.loss

    def save_probabilities(self, caminho_novo_arquivo):
        probabilidades_evasao = self.model.predict_proba(self.X)[:, 1]
        self.data["Probabilidade_Evasao"] = probabilidades_evasao
        self.data.to_excel(caminho_novo_arquivo, index=False, engine="openpyxl")

    def add_dropout_probabilities_to_dataframe(self):
        X_all_scaled = self.scaler.transform(self.data[self.selected_covariates])
        probabilities_evasao = self.model.predict_proba(X_all_scaled)[:, 1]
        self.data["Probabilidade_Evasao"] = probabilities_evasao

    def predict_new_student(self, novo_aluno):
        novo_aluno_scaled = self.scaler.transform(novo_aluno)
        probabilidade_evasao = self.model.predict_proba(novo_aluno_scaled)[:, 1]
        return probabilidade_evasao[0]

    def calculate_feature_importances(self, X, y, feature_names):
        r = permutation_importance(self.model, X, y, n_repeats=10)
        for i, feature_name in enumerate(feature_names):
            self.feature_importances[feature_name] = r.importances_mean[i]

    def normalize_feature_importances(self):
        # Calculate the sum of the absolute values of all importances
        total_importance = sum(
            abs(value) for value in self.feature_importances.values()
        )
        # Normalize each feature's importance based on the sum of absolute values
        for feature in self.feature_importances:
            self.feature_importances[feature] = (
                abs(self.feature_importances[feature]) / total_importance
            ) * 100

    def relative_feature_importances(self):
        # Find the maximum absolute importance value
        max_importance = max(abs(value) for value in self.feature_importances.values())
        # Calculate each feature's importance relative to the maximum absolute value
        for feature in self.feature_importances:
            relative_importance = (
                abs(self.feature_importances[feature]) / max_importance
            ) * 100
            # Retain the original sign of the importance value
            self.feature_importances[feature] = relative_importance * np.sign(
                self.feature_importances[feature]
            )

    def detailed_confusion(self, y_true, y_pred, set_name):
        cm = confusion_matrix(y_true, y_pred)
        percent_correct_0 = cm[0, 0] / cm[0, :].sum() * 100
        percent_correct_1 = cm[1, 1] / cm[1, :].sum() * 100
        overall_percent = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100

        results_str = (
            f"{set_name}:\n"
            "\t\tPrevisão\n"
            "Valor Real\t0\t1\tPercentual de Acerto\n"
            f"0         \t\t{cm[0, 0]}\t{cm[0, 1]}\t{percent_correct_0:.1f}%\n"
            f"1         \t\t{cm[1, 0]}\t{cm[1, 1]}\t{percent_correct_1:.1f}%\n"
            f" \n"
            f"Percentual de Acerto no {set_name} é: {overall_percent:.1f}%\n"
            f" \n"
            f"──────────────────────────────────\n"
            f" \n"
        )
        return results_str


# filter out UserWarnings from sklearn to avoid cluttering the output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Example usage:
# data_model = DataModel()
# data_model.load_data("path/to/data.xlsx")
# data_model.prepare_data(x_columns=["column1", "column2"], y_column="target")
# X_train, X_test, y_train, y_test = data_model.split_data()
# data_model.train_model(X_train, y_train)
# accuracy, loss = data_model.evaluate_model(X_test, y_test)
# print(f"Acurácia: {accuracy}, Perda de Log: {loss}")
