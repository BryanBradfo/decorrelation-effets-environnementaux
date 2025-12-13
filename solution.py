import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set_theme(style="whitegrid")


class SensorDecorrelator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.model = None
        self.features = ["temperature", "ensoleillement"]
        self.target = "deplacement"

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, sep=None, engine="python")

            self.df.columns = self.df.columns.str.strip()

            self.df["TIMESTAMP"] = pd.to_datetime(self.df["TIMESTAMP"], errors="coerce")

            self.df.set_index("TIMESTAMP", inplace=True)

            cols_to_clean = self.features + [self.target]
            for col in cols_to_clean:
                # Force la conversion en nombre, remplace les erreurs par NaN
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            initial_rows = len(self.df)
            self.df = self.df.dropna()
            clean_rows = len(self.df)
            print(self.df.head())

        except Exception as e:
            print(f"ERREUR CRITIQUE lors du chargement : {e}")
            raise e

    def explore_data(self):
        if self.df is None:
            return

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.title("Matrice de Corrélation")
        plt.show()

        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        self.df[self.target].plot(ax=ax[0], color="blue", title="Déplacement mesuré")
        self.df[self.features[0]].plot(ax=ax[1], color="red", title="Température")
        self.df[self.features[1]].plot(ax=ax[2], color="orange", title="Ensoleillement")
        plt.tight_layout()
        plt.show()

    def train_decorrelation_model(self):
        X = self.df[self.features]
        y = self.df[self.target]

        self.model = LinearRegression()
        self.model.fit(X, y)

        score = self.model.score(X, y)
        print(f"Modèle entraîné. R² score : {score:.4f}")
        print(f"Coefficients : {self.model.coef_}")

    def apply_decorrelation(self):
        if self.model is None:
            print("Modèle non entraîné.")
            return

        X = self.df[self.features]
        self.df["Environment_Effect"] = self.model.predict(X)

        self.df["Deplacement_Decorrele"] = (
            self.df[self.target] - self.df["Environment_Effect"]
        )

    def save_and_plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df[self.target], label="Brut (Mesuré)", alpha=0.5)
        plt.plot(
            self.df.index,
            self.df["Deplacement_Decorrele"],
            label="Décorrélé (Structurel)",
            color="green",
            linewidth=2,
        )
        plt.legend()
        plt.title("Comparaison : Données Brutes vs Données Décorrélées")
        plt.show()

        self.df.to_csv("resultats_decorreles.csv")
        print("Résultats sauvegardés dans 'resultats_decorreles.csv'")


if __name__ == "__main__":
    file_path = "dataset.dat"

    analysis = SensorDecorrelator(file_path)
    analysis.load_data()

    analysis.explore_data()
    analysis.train_decorrelation_model()
    analysis.apply_decorrelation()
    analysis.save_and_plot_results()
