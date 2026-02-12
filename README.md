# Optimisation d'IA Embarquée : Classification Fashion-MNIST

Ce projet explore les stratégies d'optimisation pour le déploiement de modèles de Deep Learning sur des systèmes aux ressources limitées (frugalité numérique). L'étude compare systématiquement différentes architectures pour identifier le meilleur compromis entre précision, vitesse et consommation.

## 📌 Problématique & Objectif
* **Problématique** : Comment concilier performance IA et ressources limitées des systèmes embarqués ?
* **Objectif** : Identifier l'architecture optimale pour la classification sur matériel contraint.

## 📊 Méthodologie
L'étude repose sur la comparaison de deux types d'architectures : les réseaux convolutifs (CNN) et les perceptrons multicouches (MLP).

* **Dataset** : Fashion-MNIST (70 000 images en niveaux de gris, 28x28 pixels).
* **Protocole de test** : Évaluation de la classification selon trois scénarios :
    1. Images originales.
    2. Images reconstruites par Autoencodeur (AE).
    3. Mélange d'images originales et reconstruites (Robustesse).
* **Optimisation** : Utilisation de la fonction d'activation **LeakyReLU** pour stabiliser l'apprentissage et améliorer la qualité de reconstruction par rapport au ReLU standard.

## 🚀 Résultats Clés
L'analyse montre que le choix de la dimension latente optimale se situe à **60**, offrant le meilleur compromis entre fidélité de reconstruction (SSIM élevé) et compression.

### Comparaison des performances (Architecture MLP vs CNN)
| Métrique | Classifieur CNN | Classifieur MLP | Gain |
| :--- | :--- | :--- | :--- |
| **Précision** | [cite_start]89 - 90%  | 88 - 89%  | -1%  |
| **Vitesse Entraînement** | 156s (moyenne)  | 13s (moyenne)  | **x12.4 plus rapide**  |
| **Vitesse Inférence** | 2.4s (moyenne)  | 0.73s (moyenne)  | **x3.3 plus rapide**  |

## 💡 Conclusion
Pour un système embarqué utilisant le dataset Fashion-MNIST, l'architecture la plus rationnelle est un **classifieur MLP direct**. L'utilisation d'un Autoencodeur n'apporte pas de bénéfice significatif pour la classification et génère une surcharge computationnelle inutile pour le matériel frugal.

## 🛠️ Configuration de test
* **Logiciels** : Python, Google Colab, Spyder.
* **Matériel** : AMD Ryzen 3 3200U @ 2.60 GHz, 8 Go RAM.

## 🛠️ Installation et Utilisation

### Prérequis
* Python 3.8+
* Environnement virtuel (recommandé)

### Installation
1. Clonez le dépôt :
   ```bash
   git clone [https://github.com/Majidghne/Optimisation-IA-Embarqu-e.git](https://github.com/Majidghne/Optimisation-IA-Embarqu-e.git)
   cd Optimisation-IA-Embarqu-e
   pip install -r requirements.txt
